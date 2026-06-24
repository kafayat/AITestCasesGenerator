import os
import shutil
import chromadb


def reset_vector_store(vector_store_path="vector_store"):

    if os.path.exists(vector_store_path):
        shutil.rmtree(vector_store_path)


def load_knowledge_base(
    knowledge_folder="knowledge_base",
    vector_store_path="vector_store"
):

    client = chromadb.PersistentClient(
        path=vector_store_path
    )

    collection = client.get_or_create_collection(
        name="enterprise_qa_knowledge"
    )

    documents = []
    ids = []
    metadatas = []

    if not os.path.exists(knowledge_folder):
        os.makedirs(knowledge_folder, exist_ok=True)

    for index, file_name in enumerate(os.listdir(knowledge_folder)):

        if file_name.endswith(".txt"):

            file_path = os.path.join(
                knowledge_folder,
                file_name
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                content = file.read()

            if content.strip():

                documents.append(content)
                ids.append(f"doc_{index}_{file_name}")
                metadatas.append({
                    "source": file_name,
                    "type": "enterprise_knowledge"
                })

    if documents:

        collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

    return collection


def search_knowledge_base(
    query,
    knowledge_folder="knowledge_base",
    vector_store_path="vector_store",
    top_k=1
):

    collection = load_knowledge_base(
        knowledge_folder,
        vector_store_path
    )

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    retrieved_chunks = []

    for doc, metadata in zip(documents, metadatas):

        source = metadata.get("source", "unknown")

        retrieved_chunks.append(
            f"Source: {source}\n{doc}"
        )

    return "\n\n".join(retrieved_chunks)