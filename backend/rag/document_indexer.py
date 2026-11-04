
import hashlib

from backend.rag.knowledge_base_loader import load_knowledge_base
from backend.rag.document_chunker import split_document
from backend.rag.embedding_service import generate_embedding
from backend.rag.vector_store import VectorStore


def calculate_content_hash(content: str) -> str:
    """Generate a SHA-256 fingerprint for document content."""

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def index_documents() -> int:

    # 1. Load all company documents
    documents = load_knowledge_base()

    # Connect to ChromaDB even if there are no documents,
    # because we may need to remove deleted documents.
    store = VectorStore()

    total_chunks = 0
    current_filenames = set()

    # 2. Process each document
    for document in documents:

        filename = document["filename"]
        content = document["content"]

        current_filenames.add(filename)

        print(f"\nProcessing: {filename}")

        # 3. Calculate document fingerprint
        content_hash = calculate_content_hash(content)

        # 4. Check existing records
        existing = store.get_documents_by_source(filename)

        existing_metadatas = existing["metadatas"] or []

        # Check whether the document is unchanged
        if (
            existing_metadatas
            and all(
                metadata.get("content_hash") == content_hash
                for metadata in existing_metadatas
            )
        ):
            print("Document unchanged. Skipping...")
            total_chunks += len(existing["ids"])
            continue

        print("Document is new or changed. Re-indexing...")

        # 5. Split document into chunks
        chunks = split_document(
            content,
            chunk_size=200
        )

        # 6. Generate all new embeddings BEFORE deleting old records
        ids = []
        embeddings = []
        metadatas = []

        for index, chunk in enumerate(chunks):

            chunk_id = f"{filename}_chunk_{index}"

            print(
                f"Generating embedding for chunk {index + 1}..."
            )

            embedding = generate_embedding(chunk)

            ids.append(chunk_id)
            embeddings.append(embedding)

            metadatas.append({
                "source": filename,
                "chunk_index": index,
                "content_hash": content_hash
            })

        # 7. Replace old records only after embedding succeeds
        store.delete_by_source(filename)

        if chunks:
            store.add_documents(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )

        total_chunks += len(chunks)

        print(
            f"Indexed {len(chunks)} chunks from {filename}"
        )

    # 8. Find documents that were deleted from the knowledge base
    all_records = store.get_all_metadata()

    indexed_filenames = {
        metadata["source"]
        for metadata in (all_records["metadatas"] or [])
    }

    deleted_filenames = indexed_filenames - current_filenames

    for filename in deleted_filenames:
        print(f"\nRemoving deleted document: {filename}")
        store.delete_by_source(filename)

    print("\nIndexing completed!")
    print("Current chunks indexed:", total_chunks)
    print("Total records in ChromaDB:", store.count())

    return total_chunks


if __name__ == "__main__":

    index_documents()