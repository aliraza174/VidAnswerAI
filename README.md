# 🎥 VidAnswerAI — AI-Powered YouTube Video Q&A App

VidAnswerAI is an intelligent, AI-powered question-answering application that allows users to query YouTube videos by asking natural language questions. The system automatically fetches the video’s transcript, translates it if needed, vectorizes the text using sentence embeddings, and retrieves the most relevant context to accurately answer user queries.

Built using **Streamlit** for the interactive web interface and **LangChain** for managing vector databases and AI chains, VidAnswerAI provides a seamless way to extract meaningful information from any video — without watching the whole thing.

---

## 📸 Demo

![App Screenshot](screenshot.png)

---

## 🚀 Features

✅ Fetch YouTube video transcripts (manual or auto-generated)  
✅ Translate transcripts automatically when needed  
✅ Split transcripts into clean chunks for processing  
✅ Convert text chunks into vector embeddings using Sentence Transformers  
✅ Store embeddings in a local Chroma vector database  
✅ Retrieve relevant context via semantic similarity search  
✅ Use Groq’s Llama 3 model for natural, context-based question answering  
✅ Chat-style Streamlit web interface for interactive Q&A  

---

## 📦 Tech Stack

- **Python 3**
- **Streamlit**
- **LangChain**
- **LangChain Community + Groq**
- **YouTube Transcript API**
- **Deep Translator**
- **Sentence-Transformers**
- **Chroma Vector Store**
- **Dotenv for API key management**

---

## 📂 Project Structure

