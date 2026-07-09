import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb


KNOWLEDGE_CATEGORIES = {
    "business_rules": "Business Rule",
    "sample_test_cases": "Sample Test Case",
    "requirements": "Requirement",
    "defects": "Defect",
    "standards": "Standard",
    "api_docs": "API Documentation"
}


def reset_vector_store(vector_store_path="vector_store"):
    """
    Deletes the ChromaDB vector store folder. On Windows, ChromaDB's
    PersistentClient holds file locks on the .bin files inside the
    vector store (PermissionError WinError 32). We must explicitly
    reset the client and force garbage collection to release those
    locks before attempting to delete the folder, otherwise the delete
    silently fails and the stale vector store remains, causing the next
    generation to retrieve content from the previous project.
    """

    if not os.path.exists(vector_store_path):
        return

    # Release any open ChromaDB file handles before deleting.
    try:
        import gc
        client = chromadb.PersistentClient(path=vector_store_path)
        try:
            client.reset()
        except Exception:
            pass
        del client
        gc.collect()
    except Exception:
        pass

    # Try to delete - retry a few times on Windows where file handles
    # may take a moment to release after gc.collect().
    import time

    for attempt in range(5):
        try:
            shutil.rmtree(vector_store_path)
            return
        except PermissionError:
            if attempt < 4:
                time.sleep(0.5)
            else:
                # Last resort: delete file by file, skipping locked ones
                # and leaving the folder structure so the app doesn't
                # crash, but at least removing as much stale data as
                # possible.
                for root, dirs, files in os.walk(vector_store_path, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except Exception:
                            pass
                try:
                    os.rmdir(vector_store_path)
                except Exception:
                    pass


def clear_knowledge_base(
    knowledge_folder="knowledge_base",
    vector_store_path="vector_store"
):

    if os.path.exists(knowledge_folder):
        shutil.rmtree(knowledge_folder)

    ensure_knowledge_folders(knowledge_folder)
    reset_vector_store(vector_store_path)


def ensure_knowledge_folders(knowledge_folder="knowledge_base"):

    os.makedirs(knowledge_folder, exist_ok=True)

    for category in KNOWLEDGE_CATEGORIES.keys():
        os.makedirs(
            os.path.join(knowledge_folder, category),
            exist_ok=True
        )


def split_text_into_chunks(text, chunk_size=800, overlap=100):

    chunks = []
    text = text.strip()

    if not text:
        return chunks

    start = 0

    while start < len(text):

        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

        if start <= 0:
            start = end

        if start >= len(text):
            break

    return chunks


def recreate_collection(
    vector_store_path="vector_store",
    collection_name="advanced_enterprise_qa_knowledge"
):

    client = chromadb.PersistentClient(
        path=vector_store_path
    )

    try:
        client.delete_collection(
            name=collection_name
        )
    except Exception:
        pass

    return client.get_or_create_collection(
        name=collection_name
    )


def get_collection(
    vector_store_path="vector_store",
    collection_name="advanced_enterprise_qa_knowledge"
):

    client = chromadb.PersistentClient(
        path=vector_store_path
    )

    return client.get_or_create_collection(
        name=collection_name
    )


def build_knowledge_base(
    knowledge_folder="knowledge_base",
    vector_store_path="vector_store"
):

    ensure_knowledge_folders(knowledge_folder)

    collection = recreate_collection(
        vector_store_path=vector_store_path
    )

    documents = []
    ids = []
    metadatas = []

    doc_index = 0

    for category, category_label in KNOWLEDGE_CATEGORIES.items():

        category_folder = os.path.join(
            knowledge_folder,
            category
        )

        for file_name in os.listdir(category_folder):

            if file_name.endswith(".txt"):

                file_path = os.path.join(
                    category_folder,
                    file_name
                )

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    content = file.read()

                chunks = split_text_into_chunks(
                    content,
                    chunk_size=800,
                    overlap=100
                )

                for chunk_number, chunk in enumerate(chunks):

                    documents.append(chunk)

                    ids.append(
                        f"{category}_{doc_index}_{chunk_number}_{file_name}"
                    )

                    metadatas.append({
                        "source": file_name,
                        "category": category,
                        "category_label": category_label,
                        "chunk_number": chunk_number
                    })

                    doc_index += 1

    if documents:

        collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    return collection


def search_by_category(
    collection,
    query,
    category,
    top_k=1
):

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where={
            "category": category
        }
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    chunks = []

    for doc, metadata in zip(documents, metadatas):

        source = metadata.get("source", "unknown")
        label = metadata.get("category_label", category)
        chunk_number = metadata.get("chunk_number", 0)

        chunks.append({
            "category": category,
            "category_label": label,
            "source": source,
            "chunk_number": chunk_number,
            "content": doc
        })

    return chunks


PRIORITY_CATEGORIES = [
    "business_rules",
    "sample_test_cases",
    "requirements",
    "standards",
    "defects",
    "api_docs"
]


def search_advanced_knowledge_base(
    query,
    knowledge_folder="knowledge_base",
    vector_store_path="vector_store",
    top_k_per_category=1
):
    """
    Queries all 6 knowledge categories concurrently instead of one at a
    time. Each category query is an independent ChromaDB call, so this
    is a direct equivalent of the test-case-generation concurrency fix:
    wall-clock time becomes roughly max(single category query) instead
    of sum(all 6 categories queried one after another).

    ChromaDB's PersistentClient is safe to share read-only queries across
    threads, so this is a low-risk speedup (no concurrent writes happen
    here, only reads).
    """

    ensure_knowledge_folders(knowledge_folder)

    collection = get_collection(
        vector_store_path=vector_store_path
    )

    all_chunks = []

    with ThreadPoolExecutor(max_workers=len(PRIORITY_CATEGORIES)) as executor:

        future_to_category = {
            executor.submit(
                search_by_category,
                collection,
                query,
                category,
                top_k_per_category,
            ): category
            for category in PRIORITY_CATEGORIES
        }

        results_by_category = {}

        for future in as_completed(future_to_category):

            category = future_to_category[future]

            try:
                results_by_category[category] = future.result()
            except Exception:
                # One category failing to query shouldn't block the
                # others - just return no chunks for that category.
                results_by_category[category] = []

    # Preserve the original priority ordering in the merged output,
    # even though the queries themselves ran out of order.
    for category in PRIORITY_CATEGORIES:
        all_chunks.extend(results_by_category.get(category, []))

    return all_chunks


def format_rag_context(retrieved_chunks, max_chars=900):

    formatted_chunks = []
    current_length = 0

    for chunk in retrieved_chunks:

        text = (
            f"Knowledge Type: {chunk['category_label']}\n"
            f"Source: {chunk['source']}\n"
            f"Chunk: {chunk['chunk_number']}\n"
            f"{chunk['content']}"
        )

        if current_length + len(text) > max_chars:
            break

        formatted_chunks.append(text)
        current_length += len(text)

    return "\n\n---\n\n".join(formatted_chunks)


def group_chunks_by_category(retrieved_chunks):

    grouped = {}

    for chunk in retrieved_chunks:

        label = chunk["category_label"]

        if label not in grouped:
            grouped[label] = []

        grouped[label].append(chunk)

    return grouped