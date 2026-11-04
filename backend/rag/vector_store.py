import chromadb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_DB_PATH = BASE_DIR / "vector_db"


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_PATH)
        )

        self.collection = self.client.get_or_create_collection(
            name="company_knowledge",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, ids, documents, embeddings, metadatas):
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query_embedding, top_k=3):
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results

    def count(self):
        return self.collection.count()

    def get_documents_by_source(self, source: str):
        return self.collection.get(
            where={"source": source},
            include=["metadatas"]
        )

    def get_all_metadata(self):
        return self.collection.get(
            include=["metadatas"]
        )

    def delete_by_source(self, source: str):
        self.collection.delete(
            where={"source": source}
        )