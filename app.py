# streamlit_app.py

import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Set up embeddings and Qdrant client
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_retriever(type_filter):
    retrievers = []
    # Set k to a large number to get all chunks
    all_chunks_k = 5
    if type_filter in ["both", "laws"]:
        retrievers.append(Qdrant(
            client=qdrant_client,
            collection_name="ksa_laws",
            embeddings=embedding_model
        ).as_retriever(search_kwargs={"k": all_chunks_k}))
    if type_filter in ["both", "cases"]:
        retrievers.append(Qdrant(
            client=qdrant_client,
            collection_name="ksa_cases",
            embeddings=embedding_model
        ).as_retriever(search_kwargs={"k": all_chunks_k}))
    if len(retrievers) == 1:
        return retrievers[0]
    else:
        from langchain.retrievers import EnsembleRetriever
        return EnsembleRetriever(retrievers=retrievers)

def get_llm():
    return AzureChatOpenAI(
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        openai_api_version="2024-12-01-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_key=AZURE_OPENAI_API_KEY,
        temperature=0.3
    )

def get_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
        max_len=8
    )

def build_chain(memory, retriever, llm, prompt):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=False
    )

# --- Streamlit UI ---
st.set_page_config(page_title="⚖️ KSA Legal Chatbot", layout="centered")
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>⚖️ KSA Legal RAG Chatbot</h1>
    <p style='text-align: center; color: #555;'>Ask legal questions about KSA Laws and Judgments.</p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User question input
with st.form(key="question_form"):
    user_input = st.text_input("💬 Write your legal question:", "")
    col1, col2 = st.columns([1, 2])
    with col1:
        type_filter = st.radio(
            "Select document source:",
            ["both", "laws", "cases"],
            horizontal=True,
            index=0,
        )
    with col2:
        submit = st.form_submit_button("Ask", use_container_width=True)

if submit and user_input.strip():
    # (Re)initialize chain if filter changed or not present
    if "chain_type" not in st.session_state or st.session_state.chain_type != type_filter:
        llm = get_llm()
        memory = get_memory()
        retriever = get_retriever(type_filter)
        system_template = """
        You are a KSA legal assistant. Your role:
        - Answer questions about KSA laws and cases (Sharia, Traffic, Basic Law of Governance).
        - Use ONLY the provided context (laws/cases).
        - Cite sources in brackets like [LAW: Contract Law 2004] or [CASE: Smith v Jones].
        - If unsure, say "I don’t know based on available information."
        - Be concise but accurate; include detailed explanations when asked.

        Context:
        {context}
        """
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=system_template + "\nQuestion: {question}\nAnswer:"
        )
        st.session_state.chain = build_chain(memory, retriever, llm, qa_prompt)
        st.session_state.chain_type = type_filter

    try:
        result = st.session_state.chain.invoke({"question": user_input})
        answer = result["answer"]
        sources = result.get("source_documents", [])

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", answer))
        st.session_state.chat_history.append(("Sources", sources))
    except Exception as e:
        st.error(f"Error: {e}")

# Display chat history
for idx, (role, msg) in enumerate(st.session_state.chat_history):
    if role == "Sources":
        if msg:
            with st.expander("📚 Sources", expanded=False):
                for doc in msg:
                    meta = doc.metadata
                    # Filter sources according to the selected filter
                    if st.session_state.chain_type == "cases" and meta.get('type', '').lower() != "case":
                        continue
                    if st.session_state.chain_type == "laws" and meta.get('type', '').lower() != "law":
                        continue
                    st.markdown(f"- [{meta.get('type', 'UNKNOWN').upper()}] {meta.get('source_id', 'N/A')}")
    else:
        bubble_color = "#E3F2FD" if role == "You" else "#F1F8E9"
        st.markdown(
            f"<div style='background-color: {bubble_color}; border-radius: 10px; padding: 10px; margin-bottom: 5px;'><b>{role}:</b> {msg}</div>",
            unsafe_allow_html=True,
        )

# Add a reset button at the bottom
st.markdown("---")
if st.button("🔄 Reset Chat"):
    st.session_state.chat_history = []
    if "chain_type" in st.session_state:
        del st.session_state.chain_type
    if "chain" in st.session_state:
        del st.session_state.chain
