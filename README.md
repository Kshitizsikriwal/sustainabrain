# Sustainabrain 🧠🌱

Sustainabrain is a **Streamlit-based AI assistant** that combines **Retrieval-Augmented Generation (RAG)** with **LangChain’s short-term memory** to provide context-aware, intelligent conversations.  

<img width="1280" height="800" alt="Screenshot 2025-08-28 at 2 30 36 AM" src="https://github.com/user-attachments/assets/9a6cc74c-931c-41da-b8b8-170d9eb42b02" />


It is designed to answer user queries with relevant, grounded responses while maintaining conversational flow across multiple turns.  

🔗 Live Demo: [sustainabrain.streamlit.app](https://sustainabrain.streamlit.app)  

---

## 🚀 Features
- **RAG (Retrieval-Augmented Generation)** → Retrieves domain-relevant chunks from a vector store and grounds responses.  
- **Short-Term Memory** → Maintains conversational context with LangChain memory components (`ConversationBufferMemory`).  
- **Streamlit UI** → Interactive, lightweight, and user-friendly web interface.  
- **Context-Aware Responses** → Answers adapt to ongoing dialogue instead of being isolated to single queries.  

---

## 🏗️ Architecture Overview
1. **Data Ingestion & Chunking**  
   Documents are loaded and split into smaller chunks for efficient embedding.  

2. **Vector Store**  
    Chunks are stored in a semantic index (e.g., FAISS, Pinecone) for fast retrieval.  

3. **Retriever + RAG Chain**  
    User queries retrieve top relevant chunks, which are combined with the query for LLM responses.  

4. **Short-Term Memory**  
    LangChain’s memory stores recent dialogue, enabling continuity across multiple user turns.  

5. **Streamlit Frontend**  
   Clean, interactive web app for querying and visualizing results.  

---

## 🖥️ How to Use
1. Open the app in your browser after running with Streamlit.  
2. Ask a question or upload content (if enabled).  
3. Get **context-aware responses** that use both retrieval and memory.  
4. Continue the conversation naturally – the AI remembers recent exchanges.  

---

## 🔑 Key Concepts
- **RAG (Retrieval-Augmented Generation)** → Enhances LLM responses with external knowledge.  
- **LangChain Memory** → Keeps track of short-term conversation context.  
- **Streamlit Session State** → Manages user sessions and interaction flow.  

---

## 📸 Future Enhancements
- Add support for more document formats (PDF, CSV, etc.).  
- Improve long-term memory (summarization/entity memory).  
- Support multimodal RAG (text + images).  
- Deploy with containerization (Docker).  

---

## 🤝 Contributing
Pull requests and suggestions are welcome!  
