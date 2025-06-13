# -*- coding: utf-8 -*-
"""
Created on Friday May 30 20:11:20 2025

@author: Giuseppe
"""

# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination, HandoffTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.teams import Swarm
from autogen_core.tools import FunctionTool
from typing import Sequence
from typing import Any, Dict, List


import playwright
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import OpenAI
from pathlib import Path
from IPython.display import Audio, display, HTML
from pydub import AudioSegment
from pydub.playback import play  # ✅ Import play() to actually play audio
import time
import pprint

import gc

import json
import re

import uuid

from fastapi.responses import FileResponse

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from collections import deque

# Create the FastAPI app
app = FastAPI()
# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the index.html at root
@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_path = Path("static/index.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


#####################################################################################################
#global user_proxy, team, loaded_team_state, agents, agent_list, model_client_openai, model_client_gemini

################################ Create OpenAI model client   #############################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
 
 
model_client_openai = OpenAIChatCompletionClient(
    model="gpt-4o-2024-08-06",
    api_key=OPENAI_API_KEY # Optional if you have an OPENAI_API_KEY env variable set.
)

model_client_gemini = OpenAIChatCompletionClient(
    model="gemini-1.5-flash",
    api_key=GOOGLE_API_KEY
)

#####################################################################################################
image_url = None
#########################################################################################################
################################## Initialize variables   ##################################################
CONFIG_FILE = "agent_config.json"
agents = {}
agent_list = []
team = None
loaded_team_state = None  # Will hold config if loaded before team is created
task1 ="This is a debate on ethics and AI"
print("✅ Environment cleared.")

############################ TEXT TO SPEECH  #########################################
# Globals
processed_messages = set()
stop_execution = False
speech_queue = asyncio.Queue()
user_message_queue = asyncio.Queue()
spontaneous_queue = asyncio.Queue()
prioritized_agents = deque()

client1 = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
os.makedirs("audio", exist_ok=True)  # Folder to serve audio files

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    filepath = os.path.join("audio", filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="audio/mpeg")
    return {"error": "File not found"}

async def speak_worker(websocket):
    global stop_execution

    AGENT_VOICES = {
        "moderator_agent": "onyx",
        "expert_1_agent": "nova",
        "expert_2_agent": "shimmer",
        "hilarious_agent": "alloy",
        "image_agent": "alloy",
        "describe_agent": "nova",
        "creative_agent": "onyx",
        "user": "alloy"
    }

    while True:
        item = await speech_queue.get()
        agent_name, content = item

        if item == ("system", "TERMINATE"):
            print("🛑 speak_worker terminated")
            speech_queue.task_done()
            stop_execution = True
            break

        if not content.strip():
            speech_queue.task_done()
            continue

        # Clean message
        text = content.rsplit("XYZ", 1)[0].strip()
        text = re.sub(r'\[.*?\]\(https?://\S+\)', 'You can find the image at the link', text)
        text = re.sub(r'https?://\S+', 'You can find the image at the link', text)

        try:
            voice = AGENT_VOICES.get(agent_name, "onyx")
            response = client1.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )

            filename = f"{uuid.uuid4().hex}.mp3"
            filepath = os.path.join("audio", filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)

            print(f"🔊 Audio saved to {filepath}")
            
            await asyncio.sleep(1.0)
            
            if websocket:
                await websocket.send_text(f"__AUDIO_URL__/audio/{filename}")

        except Exception as e:
            print("❌ Error in speak_worker:", e)

        speech_queue.task_done()

##########################################################################################################
################################# Build Agents from configuration  ####################################

model_clients_map = {
    "openai": model_client_openai,
    "gemini": model_client_gemini
}
##########################################################################################################
################################# Load default configuration    ##########################################
with open(CONFIG_FILE, "r") as f:
    agent_config = json.load(f)
    
##########################################################################################################
################################# Build name_to_agent_skill for introducing Agents #######################
def extract_agent_skills(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    skills = []
    for agent_id, cfg in config.items():
        description = cfg.get("description", "")
        words = description.split()
        name = words[1] if len(words) > 1 else agent_id  # crude fallback
        skills.append(f"{name} ({description.split(',', 1)[-1].strip()})")

    #skills.append("Giuseppe (user proxy)")
    return ", ".join(skills)

##########################################################################################################
################################# Build Agents from configuration  #######################################

tool_lookup = {
}
##########################################################################################################
################################# Build Agents from configuration  #######################################
def build_agents_from_config(config_path, name_to_agent_skill, model_clients_map):
    global task1
    with open(config_path, "r") as f:
        config = json.load(f)

    agents = {}
    for name, cfg in config.items():
        if name == "proxy_agent":
            continue  # Skip creating an AssistantAgent for the user_proxy
        sys_msg = (
            cfg["system_message"]
            .replace("{task1}", task1)
            .replace("{name_to_agent_skill}", name_to_agent_skill)
        )

        client_key = cfg.get("model_client", "openai")
        model_client = model_clients_map.get(client_key)
        if model_client is None:
            raise ValueError(f"Model client '{client_key}' not found for agent '{name}'")
        print("MODEL_CLIENT", model_client)

        tool_names = cfg.get("tools", [])
        if isinstance(tool_names, str):
            import ast
            tool_names = ast.literal_eval(tool_names)  # Safe way to convert string to list

        tool_list = [tool_lookup[t] for t in tool_names] if tool_names else []

        agent = AssistantAgent(
            name=name,
            description=cfg["description"],
            system_message=sys_msg,
            model_client=model_client,
            tools=tool_list)

        # Ensure llm_config exists
        if not hasattr(agent, "llm_config") or agent.llm_config is None:
            agent.llm_config = {}
        agent.llm_config["temperature"] = cfg.get("temperature", 0.7)

        agents[name] = agent
        print(f"✅ Initialized {len(agents)} agents for debate topic: {task1}")
    return agents


#print(f"✅ Initialized {len(agents)} agents for debate topic: {task1}")
    

##########################################################################################################
################################# Termination  ####################################
text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=80)
termination = text_mention_termination | max_messages_termination

##########################################################################################################
################################# Selector Prompt Function    ############################################
selector_prompt = """
You are the Selector agent following strictly the instructions of the moderator and of the selector_func
"""

def dynamic_selector_func(thread):
    global agent_id, prioritized_agents
    
    # 🧠 Force agent turn if someone is in the priority queue
    if prioritized_agents:
        next_priority = prioritized_agents.popleft()
        print(f"🎯 Prioritizing agent: {next_priority}")
        return next_priority
    
    last_msg = thread[-1]
    last_message = last_msg.content.lower().strip()
    sender = last_msg.source.lower()

    name_to_agent = {
        "alice": "expert_1_agent",
        "bob": "expert_2_agent",
        "charlie": "hilarious_agent",
        "alan": "moderator_agent",
        "albert": "creative_agent",
        "giuseppe": "user_proxy",
    }

    # 🔹 First user interaction → go to moderator
    if sender == "user":
        print("👤 User input detected. Moderator takes over.")
        return "moderator_agent"

    # 🔹 AGENT (not moderator) just spoke
    if sender != "moderator_agent":
        if last_message.endswith("xyz"):
            focus_area = last_message.rsplit("xyz", 1)[0].strip()
            pattern = r'\b(' + '|'.join(map(re.escape, name_to_agent.keys())) + r')\b'
            matches = re.findall(pattern, focus_area)
            unique_mentions = set(matches)

            if len(unique_mentions) == 1:
                mentioned = matches[0]
                if name_to_agent[mentioned] == sender:
                    print(f"🔁 Agent '{sender}' mentioned only themselves. Returning to Moderator.")
                    return "moderator_agent"
                else:
                    print(f"📣 Agent '{sender}' mentioned another agent ('{mentioned}'). Moderator should intervene.")
                    return "moderator_agent"
            elif len(unique_mentions) > 1:
                print(f"📣 Agent '{sender}' mentioned multiple agents. Moderator should intervene.")
                return "moderator_agent"
        # No 'xyz' or no mentions → let agent continue
        print(f"⏭ Agent '{sender}' keeps the floor.")
        return sender

    # 🔹 MODERATOR just spoke
    if not last_message.endswith("xyz"):
        print("⚠️ Moderator message incomplete (no 'xyz'). Staying with Moderator.")
        return "moderator_agent"

    focus_area = last_message.rsplit("xyz", 1)[0].strip()
    pattern = r'\b(' + '|'.join(map(re.escape, name_to_agent.keys())) + r')\b'
    matches = list(re.finditer(pattern, focus_area))

    if not matches:
        print("⚠️ No agent mentioned by moderator. Staying with Moderator.")
        return "moderator_agent"

    # ✅ Find the last valid agent mentioned
    for match in matches:

        if len(matches) == 1:
            match = matches[0]
            name = match.group(1)
            agent_id = name_to_agent.get(name)
            if agent_id and agent_id != "moderator_agent":
                print(f"✅ Moderator selected '{name}'. Routing to {agent_id}.")
                return agent_id

        if len(matches) >= 2:
            match = matches[1]
            name = match.group(1)
            agent_id = name_to_agent.get(name)
            if agent_id and agent_id != "moderator_agent":
                print(f"✅ Moderator selected '{name}'. Routing to {agent_id}.")
                return agent_id
    print("⚠️ Moderator mentioned only user or moderator. Staying with Moderator.")
    return "moderator_agent"

#print(dir(agents["user_proxy"]))
#print(dir (team))
#print(help (team))


####################################################################################################
# === Globals ===
user_conversation = []
gradio_input_buffer = {"message": None}
agent_config_ui = {}

################## SET TOPIC        ###########################################################
def set_task_only(task_text):
            global task1
            task1 = task_text
            return "✅ Debate topic set." if task_text else "❌ Topic cannot be empty."

##########################################################################################################
################################# Configuration File    ###################################################=
def load_agent_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_agent_config(*args):
    updated = {}
    idx = 0
    for name in agent_config_ui:
        updated[name] = {
            "description": args[idx],
            "system_message": args[idx + 1],
            "temperature": args[idx + 2],
            "model_client": args[idx + 3],
            "tools": args[idx + 4]
        }
        idx += 5
    with open(CONFIG_FILE, "w") as f:
        json.dump(updated, f, indent=2)
    return "✅ Configuration saved."


##########################################################################################################
################################# SAVE Config    ###################################################
# === Save config ===
async def save_config():
    global team
    team_state = await team.save_state()
    with open("coding/team_state.json", "w") as f:
        json.dump(team_state, f)
    print("✅ Config saved.")
    return "✅ Config saved to disk."

def sync_save_config():
    return asyncio.run(save_config())  # 'team' must be globally accessible


##########################################################################################################
################################# LOAD Config   ###################################################
def sync_load_config():
    global loaded_team_state
    with open("coding/team_state.json", "r") as f:
        loaded_team_state = json.load(f)
    print("🟢 Config loaded and pending application.")
    return "🟢 Config loaded. Ready to apply when system starts."

##########################################################################################################
################################# User Intervention    ###################################################
# === User interaction ===
def handle_user_message(message):
    global user_conversation, gradio_input_buffer
    gradio_input_buffer["message"] = message
    user_conversation.append(f"🧑 USER: {message}")
    return "\n".join(user_conversation)

def intervene_now(user_input):
    return handle_user_message(user_input)


##########################################################################################################
################################# loop for debate  #######################################################
async def run_chat(team, websocket=None):
    global stop_execution, image_url, task1, gradio_input_buffer

    print("🚀 Starting debate with streaming...")

    async for result in team.run_stream(task=task1):
        if stop_execution:
            break
        
        if hasattr(result, "content") and isinstance(result.content, str):
            text = result.content
            agent_name = result.source

            if agent_name == "user_proxy":
                print("🧑 Giuseppe (user_proxy) has responded.")

            if not hasattr(team, "_message_history"):
                team._message_history = []
            team._message_history.append({"sender": agent_name, "content": text})

            print(f"👤 sender: {agent_name}")
            print(f"📝 content: {text}")
            print(f"🖼️ image_url: {image_url}")

            prefix = "🧑" if "user" in agent_name.lower() else "🤖"
            user_conversation.append(f"{prefix} {agent_name.upper()}: {text}")

            await speech_queue.put((agent_name, text))

            if "TERMINATE" in text:
                stop_execution = True
                await speech_queue.put(("system", "TERMINATE"))
                print("✅ Chat terminated.")
                break
            
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "message": "Service is running"}

import traceback

#########

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global team, agents, agent_list, stop_execution, loaded_team_state, task1, user_message_queue

    team = None
    stop_execution = False
    task1 = None  # 🆕 Debate topic will be set by user

    async def flush_queue(queue: asyncio.Queue):
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                break

    await flush_queue(user_message_queue)
    await flush_queue(speech_queue)

    await websocket.accept()
    try:
        # ✳️ Wait for user to submit task1 before anything else
        await websocket.send_text("📝 Please enter the debate topic and click 'Set Topic' before continuing.")
        while task1 is None:
            init_msg = await websocket.receive_text()
            if init_msg.startswith("__SET_TASK1__:"):
                task1 = init_msg.replace("__SET_TASK1__:", "").strip()
                print(f"🟢 Debate topic set: {task1}")
                await websocket.send_text(f"✅ Debate topic received:\n{task1}")
            elif init_msg == "__ping__":
                continue
            else:
                await websocket.send_text("⚠️ Please use the 'Set Topic' button to begin.")

        # 🔧 Load agents
        name_to_agent_skill = extract_agent_skills(CONFIG_FILE)
        agents = build_agents_from_config(CONFIG_FILE, name_to_agent_skill, model_clients_map)

        # 🎤 User input handler
        
        
#####################################################################################################
        async def websocket_listener(websocket):
            global user_message_queue, spontaneous_queue
            while True:
                print("PLUTO2")                
                data = await websocket.receive_text()
                if data == "__ping__":
                    print("PLUTO2")
                    continue
                if data == "__USER_PROXY_TURN__":
                    print("🎙️ Moderator gave the floor to user.")
                    # Wait for user input (the next message will go to the queue)
                
                elif data.startswith("__SPONTANEOUS__"):
                    message = data.replace("__SPONTANEOUS__", "").strip()
                    print("⚡ Spontaneous input received:", message)
                    await spontaneous_queue.put(message)
                    prioritized_agents.append("user_proxy")  # 🔥 Ask selector to prioritize user
                    continue
                
                else:
                    print("👤 User responded:", data)
                    print("PLUTO4")
                    await user_message_queue.put(data)
        
        """
        async def wrapped_input_func(*args, **kwargs):
                global  user_message_queue
            
            
                # 🧑 Moderator gave the floor
                print("⏳ Waiting for user response...")
                while True:
                    msg = await user_message_queue.get()
                    if msg and msg.strip():
                        return msg
        """
       
        async def wrapped_input_func(*args, **kwargs):
                global spontaneous_queue, user_message_queue
            
                if not spontaneous_queue.empty():
                    msg = await spontaneous_queue.get()
                    print("⚡ Using spontaneous message:", msg)
                    return msg
            
                print("⏳ Waiting for user input (moderator turn)...")
                while True:
                    msg = await user_message_queue.get()
                    if msg and msg.strip():
                        return msg         


        agents["user_proxy"] = UserProxyAgent(name="user_proxy", input_func=wrapped_input_func)

        agent_list = [
            agents["moderator_agent"],
            agents["expert_1_agent"],
            agents["expert_2_agent"],
            agents["hilarious_agent"],
            agents["creative_agent"],
            agents["user_proxy"],
        ]

        team = SelectorGroupChat(
            agent_list,
            model_client=model_client_openai,
            selector_func=dynamic_selector_func,
            termination_condition=termination,
            allow_repeated_speaker=True,
        )

        if loaded_team_state:
            await team.load_state(loaded_team_state)
            loaded_team_state = None
        
        asyncio.create_task(websocket_listener(websocket))
        
        asyncio.create_task(speak_worker(websocket))
        
        await run_chat(team, websocket=websocket)
        
        await speech_queue.join()

        stop_execution = True
        return "✅ Debate complete."

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
        await speech_queue.put(("system", "TERMINATE"))
        stop_execution = True
        team = None

    except Exception as e:
        traceback.print_exc()
        await websocket.send_text("⚠️ Internal server error during debate.")