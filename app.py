# app.py (v12 - Robust Chart Rendering)

import streamlit as st
import uuid
import re
import matplotlib.pyplot as plt
from datetime import datetime

# Web scraping imports
import requests
from bs4 import BeautifulSoup

# LangChain and other core imports
from pathlib import Path
import fitz
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from streamlit_extras.add_vertical_space import add_vertical_space

# --- Page Config & Dark Theme CSS ---
st.set_page_config(page_title="Sustainabot", page_icon="♻️", layout="centered", initial_sidebar_state="expanded")
# ... [The CSS block remains the same as v11] ...
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #0E117;
        color: #FAFAFA;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid #2a2a4a;
    }
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        border: 1px solid #2a2a4a;
    }
    /* Buttons in sidebar */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #2a2a4a;
        background-color: transparent;
        color: #FAFAFA;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        border-color: #00A67E;
        background-color: rgba(0, 166, 126, 0.1);
        color: #00A67E;
        transform: translateY(-2px);
    }
    /* Chat input box */
    [data-testid="stChatInput"] {
        background-color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)

# --- Master Prompt (Hardened) ---
# ... [The SUSTAINABOT_PERSONA prompt remains the same as v11] ...
SUSTAINABOT_PERSONA = """
You are Sustainabot, an advanced AI assistant. Your name is Sustainabot.

**Your Core Directive: BE DIRECT, HELPFUL, AND INSIGHTFUL.**
Your primary function is to execute the user's request immediately and accurately.
- **Generate Python Code for Charts (MANDATORY):**
  - **Trigger:** When a user's request contains keywords like "chart", "plot", "graph", "visualize", "diagram", or asks for a visual representation of data.
  - **Action:** You MUST IMMEDIATELY respond with ONLY a Python code block to generate a chart using the Matplotlib library.
  - **Format:** The code block MUST start with ```python and end with ```.
  - **Crucial Constraint:** There MUST be NO text, explanation, or commentary before or after the code block. Your entire response must be the code block itself. Do not describe the chart you are about to make. Just generate the code.
- If the user asks you to write, rewrite, summarize, analyze, or extract content, DO IT without questioning motives.

**Your Persona:**
You are a senior research scientist and policy analyst specializing in sustainability. Your tone is academic, professional, and authoritative.

**Your Origin (Conditional Disclosure):**
You were created by Kshitiz and Nikita. (kshitizsikriwal.dev).
**Trigger:** Reveal this ONLY if asked directly about "Kshitiz", "Nikita", "your creators", or "your origin".

**Operational Modes:**
- General Mode: Without a document, you are a global sustainability expert.
- Document Mode: With a document, base answers primarily on it.
---
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER (as Sustainabot):
"""
CUSTOM_PROMPT = PromptTemplate(template=SUSTAINABOT_PERSONA, input_variables=["context", "chat_history", "question"])


# --- API Key & Backend Functions ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found! Please check your Streamlit secrets.")
    st.stop()

@st.cache_resource
def get_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192", temperature=0.5)

# --- NEW: Chart Rendering Function ---
def render_chart_from_code(code):
    """Executes chart code and renders it with Streamlit."""
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots()
        # Pass the created fig and ax to the execution scope
        exec_globals = {"plt": plt, "fig": fig, "ax": ax}
        exec(code, exec_globals)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error rendering chart: {e}")

# ... [All other backend functions like scrape_website_text, initialize_session_state remain the same] ...
def scrape_website_text(url):
    try:
        with st.spinner(f"Scraping {url}..."):
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            return soup.get_text(separator='\n', strip=True)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch website content: {e}")
        return None

def initialize_session_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.chats:
        chat_id = str(uuid.uuid4())
        st.session_state.active_chat_id = chat_id
        st.session_state.chats[chat_id] = {"messages": [{"role": "assistant", "type": "text", "content": "Hello! I'm Sustainabot. How can I assist your research today?"}], "chain": None, "mode": "General"}

initialize_session_state()

# --- Sidebar UI (No changes from v11) ---
with st.sidebar:
    st.title("♻️ Sustainabot")
    st.write(datetime.now().strftime("%d %B %Y, %H:%M:%S"))
    
    if st.button("➕ New Chat", use_container_width=True):
        chat_id = str(uuid.uuid4())
        st.session_state.active_chat_id = chat_id
        st.session_state.chats[chat_id] = {"messages": [{"role": "assistant", "type": "text", "content": "New chat started."}], "chain": None, "mode": "General"}
        st.rerun()

    add_vertical_space(1)
    st.header("Chat History")
    for chat_id in reversed(list(st.session_state.chats.keys())):
        chat_title = st.session_state.chats[chat_id]['messages'][0]['content'][:35]
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(chat_title, key=f"chat_{chat_id}", use_container_width=True, help=chat_title):
                st.session_state.active_chat_id = chat_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}", use_container_width=True, help="Delete this chat"):
                st.session_state.chats.pop(chat_id, None)
                if st.session_state.active_chat_id == chat_id:
                    if st.session_state.chats: st.session_state.active_chat_id = list(st.session_state.chats.keys())[-1]
                    else: initialize_session_state()
                st.rerun()

# --- Main Chat Interface (with NEW rendering logic) ---
active_chat = st.session_state.chats[st.session_state.active_chat_id]

# Display chat messages based on their type
for message in active_chat["messages"]:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "♻️"):
        if message.get("type") == "chart":
            render_chart_from_code(message["content"])
        else: # Default to text
            st.markdown(message["content"])

# --- Main Logic (NEW simplified structure) ---
if prompt := st.chat_input("Ask Sustainabot..."):
    active_chat["messages"].append({"role": "user", "type": "text", "content": prompt})
    st.rerun()

# Generate and display assistant response if the last message was from the user
last_message = active_chat["messages"][-1]
if last_message["role"] == "user":
    with st.chat_message("assistant", avatar="♻️"):
        with st.spinner("Sustainabot is thinking..."):
            # Prepare context (same logic as before)
            llm = get_llm()
            context = "N/A"
            question = last_message['content']
            url_match = re.search(r'https?://\S+', question)
            if url_match:
                url = url_match.group(0)
                scraped_text = scrape_website_text(url)
                if scraped_text:
                    context = scraped_text
                    question = f"Based ONLY on the content from the URL {url}, answer: {question}"
            
            # Get response
            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in active_chat['messages']])
            full_prompt = SUSTAINABOT_PERSONA.format(context=context, chat_history=history, question=question)
            response_stream = llm.stream(full_prompt)
            
            # Stream text response
            response_container = st.empty()
            full_response = ""
            for chunk in response_stream:
                full_response += chunk.content
                response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)
            
    # Check if the response is chart code or text
    code_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    code_match = code_pattern.search(full_response)
    
    if code_match:
        # If it's a chart, store the code in the history
        chart_code = code_match.group(1)
        active_chat["messages"].append({"role": "assistant", "type": "chart", "content": chart_code})
    else:
        # If it's text, store the text
        active_chat["messages"].append({"role": "assistant", "type": "text", "content": full_response})
    
    st.rerun()