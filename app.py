import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_pdf_text, split_text

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Retriever (TF-IDF based)
def find_relevant_chunks(question, chunks, top_n=3):
    if not chunks:
        return []
    texts = chunks + [question]
    try:
        vectorizer = TfidfVectorizer().fit_transform(texts)
        cosine_sim = cosine_similarity(vectorizer[-1], vectorizer[:-1]).flatten()
        top_indices = cosine_sim.argsort()[-top_n:][::-1]
        return [chunks[i] for i in top_indices]
    except Exception as e:
        st.error(f"Error in finding relevant chunks: {str(e)}")
        return []

# Gemini Answer Generator
def ask_gemini(context, question):
    if not context:
        return "No relevant context found in the PDF to answer the question."
    prompt = f"Based on the following context, answer the question:\n\n{context}\n\nQuestion: {question}"
    try:
        response = model.generate_content([{"parts": [{"text": prompt}]}])
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating content: {str(e)}")
        return "Failed to generate an answer due to an API error."

# UI
st.set_page_config(page_title="PDF Q&A  By Zohaib", layout="centered")
st.title("📄 PDF Q&A Bot By Zohaib")

pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf:
    with st.spinner("Reading PDF..."):
        raw_text = load_pdf_text(pdf)
        if raw_text.startswith("Error"):
            st.error(raw_text)
        else:
            chunks = split_text(raw_text)
            if not chunks:
                st.error("No text chunks were created. The PDF may be empty or too short.")
            else:
                st.success(f"PDF loaded and processed! ({len(chunks)} chunks created)")

    user_q = st.text_input("Ask a question about the PDF:")
    if user_q:
        if not raw_text or not chunks:
            st.error("Cannot process the question because no text was extracted from the PDF.")
        else:
            with st.spinner("Thinking..."):
                relevant_chunks = find_relevant_chunks(user_q, chunks)
                context = "\n\n".join(relevant_chunks)
                answer = ask_gemini(context, user_q)
                st.markdown("### 💬 Answer:")
                st.write(answer)
                if relevant_chunks:
                    with st.expander("View Relevant Chunks"):
                        for i, chunk in enumerate(relevant_chunks, 1):
                            st.write(f"**Chunk {i}**: {chunk[:200]}...")
