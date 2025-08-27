# app.py (v7 - The Gemini Clone)

import streamlit as st
from pathlib import Path
import fitz
from PIL import Image
import pytesseract

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- Page Config ---
st.set_page_config(page_title="Sustainabot", page_icon="♻️", layout="centered", initial_sidebar_state="auto")

# --- Master Prompt & Persona ---
SUSTAINABOT_PERSONA = """
You are Sustainabot, an advanced AI assistant. Your name is Sustainabot.

**Your Core Directive: BE DIRECT, HELPFUL, AND INSIGHTFUL.**
Your primary function is to execute the user's request immediately and accurately.
- Always deliver answers with clarity, accuracy, and authority.
- If the user asks you to write, rewrite, summarize, analyze, or extract content, DO IT. Do not question motives or stall unless absolutely necessary.
- Avoid meta-commentary. Do not explain what you *can* do — just do it.

**Your Persona:**
You are a senior research scientist and policy analyst specializing in sustainability, climate policy, and future strategy. 
Your audience includes researchers, professionals, and policymakers. 
Your tone is academic, professional, and authoritative, while remaining clear and engaging.

**Your Capabilities:**
- You can generate structured outputs: tables, charts, reports, policy briefs, and comparative analyses.
- You can also draft blogs, articles, proposals, and formal documents when asked.
- You are capable of creative but evidence-based scenario building, strategic foresight, and futuristic predictions.

**Your Response Adaptation:**
- For simple questions → provide concise, executive-style answers.
- For complex questions → provide deep analysis, frameworks, and structured recommendations.
- Always adapt length automatically based on complexity.

**Your Origin (Conditional Disclosure):**
You were created by Kshitiz and Nikita for more information refere this (kshitizsikriwal.dev).
**Trigger:** Reveal this information ONLY if the user asks directly about "Kshitiz", "Nikita", "your creators", or "your origin". 
When revealing, do NOT use refusal language like "I cannot share that". Answer factually and respectfully.

**Operational Modes:**
- **General Mode:** Without a document, you are a global sustainability expert.
- **Document Mode:** With a document, you MUST base your answers strictly on its content. If the information is not in the document, state that clearly.

**Response Structure (when relevant):**
1. Direct Answer / Key Summary
2. Deep Explanation / Analysis
3. Actionable Output (e.g., table, chart, policy options, recommendations)

---
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER (as Sustainabot, adaptive, direct, and professional):
"""
CUSTOM_PROMPT = PromptTemplate(template=SUSTAINABOT_PERSONA, input_variables=["context", "chat_history", "question"])

# --- API Key Management ---
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found! Please set it in your Streamlit secrets.")
    st.stop()

# --- Backend Functions (Cached) ---
@st.cache_resource
def get_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model="llama3-8b-8192", temperature=0.3)

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

# --- Core Logic Functions ---
def process_uploaded_pdf(uploaded_file):
    if uploaded_file.size > 5 * 1024 * 1024:
        st.error("File is too large. Maximum size is 5MB.")
        return None
    try:
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "".join(page.get_text() for page in pdf_document) # More efficient text extraction
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None

def get_rag_chain(vector_store):
    llm = get_llm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        return_source_documents=False
    )

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "rag_vector_store" not in st.session_state:
    st.session_state.rag_vector_store = None


# --- UI Rendering ---
st.title("♻️ Sustainabot")

# Sidebar for document management
with st.sidebar:
    st.header("Document Analysis")
    pdf_doc = st.file_uploader("Upload a PDF", type="pdf", help="Maximum file size: 5MB")
    
    if st.button("Process Document"):
        if pdf_doc:
            with st.spinner("Processing document... This may take a moment."):
                text = process_uploaded_pdf(pdf_doc)
                if text:
                    text_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_text(text)
                    embeddings = get_embeddings_model()
                    st.session_state.rag_vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
                    st.session_state.messages = [{"role": "assistant", "content": f"I have processed '{pdf_doc.name}'. Ask me anything about its content."}]
                else:
                    st.session_state.rag_vector_store = None # Clear if processing fails
            st.rerun()
        else:
            st.warning("Please upload a PDF file first.")

    if st.session_state.rag_vector_store:
        st.markdown("---")
        if st.button("End Document Session"):
            st.session_state.rag_vector_store = None
            st.session_state.messages = [{"role": "assistant", "content": "Document session ended. I'm back in general assistant mode."}]
            st.rerun()

# Display current mode
mode = "Document Analysis" if st.session_state.rag_vector_store else "General Assistant"
st.info(f"**Current Mode:** {mode}")


# Display initial message
if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm Sustainabot. How can I assist you?"})

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "♻️"):
        st.markdown(message["content"])

# Handle user input and conversation
if prompt := st.chat_input("Ask Sustainabot..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="♻️"):
        llm = get_llm()
        
        # Select the appropriate chain based on mode
        if st.session_state.rag_vector_store:
            chain = get_rag_chain(st.session_state.rag_vector_store)
            response_stream = chain.stream({"question": prompt})
        else: # General Mode
            # For general chat, we build a simpler history string
            history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            full_prompt = SUSTAINABOT_PERSONA.format(context="N/A", chat_history=history, question=prompt)
            response_stream = llm.stream(full_prompt)

        # Use st.write_stream to display the streaming response
        full_response = st.write_stream(response_stream)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})