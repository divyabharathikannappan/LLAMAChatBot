# data_loader.py

import pandas as pd
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Define constants
CSV_PATH = "dataset.csv"
PERSIST_DIR = "embeddings"

def create_documents_from_csv(csv_path):
    df = pd.read_csv(csv_path).fillna("N/A")
    documents = []

    for _, row in df.iterrows():
        content = f"""
        Grant Name: {row['GrantName']}
        Description: {row['GrantDescription']}
        Amount: {row['Amount']}
        Deadline: {row['Deadline']}
        Type: {row['GrantType']}
        Province: {row['Province']}
        Country: {row['Country']}
        Information Link: {row['InformationLink']}
        Focus Area: {row['FocusArea']}
        Eligibility Criteria: {row['EligibilityCriteria']}
        """
        documents.append(Document(page_content=content.strip(), metadata={
            "country": row["Country"].lower(),
            "province": row["Province"].lower()
        }))
    return documents

def generate_chroma_db():
    print("[INFO] Loading dataset and generating embeddings...")
    documents = create_documents_from_csv(CSV_PATH)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectordb = Chroma.from_documents(documents, embedding, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print(f"[SUCCESS] Chroma DB saved to `{PERSIST_DIR}`")

if __name__ == "__main__":
    generate_chroma_db()
