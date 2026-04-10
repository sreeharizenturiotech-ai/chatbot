import whisper
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gtts import gTTS
import gradio as gr
import os

# =========================
# LOAD MODELS
# =========================

stt_model = whisper.load_model("tiny")

with open("rag_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

texts = [doc["text"] for doc in documents]

# =========================
# CHUNKING
# =========================
def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

chunks = []
for t in texts:
    chunks.extend(chunk_text(t))

# =========================
# EMBEDDINGS + FAISS
# =========================
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

embeddings = embedder.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# =========================
# RETRIEVE
# =========================
def retrieve(question, k=3):
    query_embedding = embedder.encode([question]).astype("float32")
    D, I = index.search(query_embedding, k)
    return [chunks[i] for i in I[0]]

# =========================
# SIMPLE ANSWER
# =========================
def generate_answer(context, question):
    if not context:
        return "I don't know based on the provided documents."
    return context[0][:200]

# =========================
# MAIN FUNCTION
# =========================
def voice_chat(audio):
    if audio is None:
        return "No audio", None

    result = stt_model.transcribe(audio)
    question = result["text"]

    docs = retrieve(question)
    context = "\n".join(docs)

    answer = generate_answer(context, question)

    tts = gTTS(answer)
    tts.save("response.mp3")

    return answer, "response.mp3"

# =========================
# GRADIO UI
# =========================
interface = gr.Interface(
    fn=voice_chat,
    inputs=gr.Audio(type="filepath"),
    outputs=["text", "audio"],
    title="🎤 Voice RAG Assistant"
)

interface.launch()
