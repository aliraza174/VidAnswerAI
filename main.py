import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnableLambda, RunnableParallel
from dotenv import load_dotenv
from deep_translator import GoogleTranslator, exceptions as dt_exceptions
from urllib.parse import urlparse, parse_qs
import os
import time

# Load API Key
load_dotenv()
api_key = os.getenv("KEY")

# Extract Video ID from URL
def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    elif parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        query = parse_qs(parsed_url.query)
        return query.get('v', [None])[0]
    else:
        return None

# Fetch YouTube Transcript & Translate
def get_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL.")

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_manually_created_transcript(['en'])
        except:
            generated_languages = list(transcript_list._generated_transcripts.keys())
            if not generated_languages:
                raise NoTranscriptFound("No available transcript found.")
            transcript = transcript_list.find_generated_transcript(generated_languages)

        transcript_chunks = transcript.fetch()
        full_text = " ".join(t.text for t in transcript_chunks)

        # Translate in safe chunks
        chunk_size = 2000
        translated_text = ""

        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size]
            for attempt in range(3):
                try:
                    translated_piece = GoogleTranslator(source='auto', target='en').translate(chunk)
                    translated_text += translated_piece + " "
                    break
                except dt_exceptions.RequestError:
                    if attempt < 2:
                        time.sleep(2)
                        continue
                    else:
                        raise ValueError("Could not fetch translation after multiple attempts.")

        return translated_text.strip()

    except Exception as e:
        raise ValueError(f"Could not fetch transcript: {e}")

# Split Text
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# Store Embeddings
def store_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_texts(chunks, embedding=embeddings)
    return vector_store

# Combine everything into vector store
def video_to_vector(video_url):
    text = get_transcript(video_url)
    chunks = split_text(text)
    vector_store = store_embeddings(chunks)
    return vector_store

# Retrieve context from vector DB
def retrieve_context(inputs):
    retriever = inputs["vector_store"].as_retriever()
    docs = retriever.get_relevant_documents(inputs["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    return {"context": context, "question": inputs["question"]}

# LLM to answer questions
def answer_with_llm(inputs):
    llm = ChatGroq(api_key=api_key, model="llama3-8b-8192")
    prompt = f"""
    You are a helpful assistant. Answer the question based on the video transcript context.

    Context:
    {inputs["context"]}

    Question:
    {inputs["question"]}

    If context doesn't provide an answer, reply "Sorry, no relevant info found in the video."
    """
    response = llm.invoke(prompt)
    return response.content

# Build complete QA pipeline
def build_pipeline(video_url):
    vector_store_runnable = RunnableLambda(lambda _: video_to_vector(video_url))
    parallel_inputs = RunnableParallel({
        "vector_store": vector_store_runnable,
        "question": RunnableLambda(lambda x: x)
    })
    pipeline = parallel_inputs | RunnableLambda(retrieve_context) | RunnableLambda(answer_with_llm)
    return pipeline

# ---------------- Streamlit UI -----------------
st.set_page_config(page_title="YouTube Video Q&A", page_icon="🎥")

if "page" not in st.session_state:
    st.session_state.page = "url_input"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# Page 1: Enter YouTube URL
if st.session_state.page == "url_input":
    st.title("🎥 YouTube Video Q&A")
    video_url = st.text_input("Enter YouTube video URL:")
    if st.button("Load Video"):
        try:
            with st.spinner("Processing video transcript and building vector store..."):
                pipeline = build_pipeline(video_url)
                st.session_state.pipeline = pipeline
                st.session_state.page = "chat"
                st.session_state.messages = []
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# Page 2: Chat Interface
elif st.session_state.page == "chat":
    st.title("🤖 Ask Questions About the Video")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about the video transcript:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            answer = st.session_state.pipeline.invoke(user_input)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    st.divider()
    if st.button("🔙 Back to Video URL"):
        st.session_state.page = "url_input"
        st.rerun()
