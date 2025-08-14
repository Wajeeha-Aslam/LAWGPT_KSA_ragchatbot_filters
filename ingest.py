# ingest.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import Qdrant

from pymongo import MongoClient
from qdrant_client import QdrantClient

import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")
LAWS_PATH = os.getenv("LAWS_PATH", "pdfs") 

embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
qdrant_client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY, 
    timeout=360,
    prefer_grpc=True  
)

from qdrant_client.http import models

# # Ensure collection exists before uploading
qdrant_client.recreate_collection(
    collection_name="ksa_cases",
    vectors_config=models.VectorParams(
        size=384,  # Match your embedding size
        distance=models.Distance.COSINE
    )
    
)
qdrant_client.recreate_collection(
    collection_name="ksa_laws",
    vectors_config=models.VectorParams(
        size=384,  # Match your embedding size
        distance=models.Distance.COSINE
    )
)


# --- Ingest Laws (PDFs) ---
def ingest_laws():
    print("🔄Loading laws from PDF folder...")
    all_docs = []
    for fname in os.listdir(LAWS_PATH):
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(LAWS_PATH, fname))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_id"] = fname
                doc.metadata["type"] = "law"
            all_docs.extend(docs)

    docs_split = text_splitter.split_documents(all_docs)
    print(f"🚀 Uploading {len(docs_split)} law chunks to Qdrant...")
    
    vector_store = Qdrant(
    client=qdrant_client,
    collection_name="ksa_laws",
    embeddings=embedding_model,
    )
    

    vector_store.add_documents(docs_split)
    


def ingest_cases():
    
    print("🔄 Connecting to MongoDB...")
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    print("📥 Fetching cases from MongoDB...")
    documents = []

    for case in collection.find():
        text_parts = []

        # Extract data from nested fields
        original = case.get("original_case", {})
        extracted = case.get("extracted_schema", {})

        # Case-level fields
        case_number = extracted.get("CaseNumber", original.get("Case Number", "Unknown"))
        text_parts.append(f"case_number: {case_number}")

        judgment_text = original.get("Judgment Text")
        if judgment_text:
            text_parts.append(f"judgment_text: {judgment_text}")

        verdict = extracted.get("Verdict")
        if verdict:
            text_parts.append(f"verdict: {verdict}")

        legal_basis = extracted.get("LegalBasis", {})
        rejection_reason = None
        rejection_ref = None
        if isinstance(legal_basis, dict):
            rejection_reason = legal_basis.get("RejectedReason")
            rejection_ref = legal_basis.get("RejectionReference")
        elif isinstance(legal_basis, list) and legal_basis:
            # Use the first item or join all reasons if needed
            first_basis = legal_basis[0]
            if isinstance(first_basis, dict):
                rejection_reason = first_basis.get("RejectedReason")
                rejection_ref = first_basis.get("RejectionReference")
        if rejection_reason:
            text_parts.append(f"rejection_reason: {rejection_reason}")
        if rejection_ref:
            text_parts.append(f"rejection_reference: {rejection_ref}")

        # Appealed Judgment metadata
        appealed = extracted.get("AppealedJudgment", {})
        if appealed:
            appealed_info = (
                f"appealed_court: {appealed.get('Court', '')}\n"
                f"appeal_circuit: {appealed.get('AppealCircuit', '')}\n"
                f"appealed_judgment_number: {appealed.get('JudgmentDocumentNumber', '')}\n"
                f"appealed_judgment_date: {appealed.get('JudgmentDateHijri', '')}"
            )
            text_parts.append(appealed_info)

        text = "\n".join(text_parts).strip()
        if not text:
            continue

        metadata = {
            "source_id": case_number,
            "judgment_date": extracted.get("JudgmentDateHijri", ""),
            "judgment_authority": extracted.get("JudgmentAuthority", ""),
            "city": original.get("City", ""),
            "court_name": original.get("Court Name", ""),
            "type": "case",
            "judgment_text": judgment_text,
            "verdict": verdict,
            "rejection_reason": rejection_reason,
            "rejection_reference": rejection_ref,
            "appealed_court": appealed.get("Court", ""),
            "appeal_circuit": appealed.get("AppealCircuit", ""),
        }

        doc = Document(page_content=text, metadata=metadata)
        documents.append(doc)

    docs_split = text_splitter.split_documents(documents)
    print(f"🚀 Uploading {len(docs_split)} case chunks to Qdrant...")

    vector_store = Qdrant(
        client=qdrant_client,
        collection_name="ksa_cases",
        embeddings=embedding_model,
    )
    vector_store.add_documents(docs_split)

# # --- Ingest Cases (MongoDB) ---
# def ingest_cases():
#     print("[*] Connecting to MongoDB...")
#     mongo_client = MongoClient(MONGO_URI)
#     db = mongo_client[MONGO_DB]
#     collection = db[MONGO_COLLECTION]

#     print("[*] Fetching cases from MongoDB...")
#     documents = []
#     for case in collection.find():
#         text_parts = []
#         if case.get("case_number"):
#             text_parts.append(f"case_number: {case['case_number']}")
#         if case.get("case_title"):
#             text_parts.append(f"case_title: {case['case_title']}")
#         if case.get("judgment"):
#             text_parts.append(f"judgment: {case['judgment']}")
#         if case.get("verdict"):
#             text_parts.append(f"verdict: {case['verdict']}")
#         if case.get("judgment_text"):
#             text_parts.append(f"judgment_text: {case['judgment_text']}")

#         text = "\n".join(text_parts) if text_parts else ""

#         if not text.strip():
#             continue

#         metadata = {
#             "source_id": case.get("case_number", "Unknown"),
#             "case_title": case.get("case_title", ""),
#             "claimant": case.get("claimant", ""),
#             "defendant": case.get("defendant", ""),
#             "type": "case"
#         }

#         doc = Document(page_content=text, metadata=metadata)
#         documents.append(doc)

#     docs_split = text_splitter.split_documents(documents)
#     print(f"[+] Uploading {len(docs_split)} case chunks to Qdrant...")
#     vector_store = Qdrant(
#         client=qdrant_client,
#         collection_name="cases",
#         embeddings=embedding_model,
#     )
#     vector_store.add_documents(docs_split)
if __name__ == "__main__":
    ingest_laws()      
    ingest_cases()
    print("✅ All embeddings uploaded.")
