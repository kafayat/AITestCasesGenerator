import os
import shutil
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

    if os.path.exists(vector_store_path):
        shutil.rmtree(vector_store_path)


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


def search_advanced_knowledge_base(
    query,
    knowledge_folder="knowledge_base",
    vector_store_path="vector_store",
    top_k_per_category=1
):

    ensure_knowledge_folders(knowledge_folder)

    collection = get_collection(
        vector_store_path=vector_store_path
    )

    all_chunks = []

    priority_categories = [
        "business_rules",
        "sample_test_cases",
        "requirements",
        "standards",
        "defects",
        "api_docs"
    ]

    for category in priority_categories:

        chunks = search_by_category(
            collection,
            query,
            category,
            top_k=top_k_per_category
        )

        all_chunks.extend(chunks)

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