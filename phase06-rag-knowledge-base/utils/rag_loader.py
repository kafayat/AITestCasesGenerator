# Import os module for reading folders and files
import os

# Import ChromaDB vector database
import chromadb


# Load knowledge documents into ChromaDB
def load_knowledge_base():

    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(
        path="vector_store"
    )

    # Create or get collection
    collection = client.get_or_create_collection(
        name="qa_knowledge_base"
    )

    # Folder where knowledge documents are stored
    knowledge_folder = "knowledge_base"

    # Store document text
    documents = []

    # Store document IDs
    ids = []

    # Loop through knowledge files
    for index, file_name in enumerate(
        os.listdir(knowledge_folder)
    ):

        # Only process .txt files
        if file_name.endswith(".txt"):

            # Build file path
            file_path = os.path.join(
                knowledge_folder,
                file_name
            )

            # Read file content
            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                content = file.read()

            # Add document content
            documents.append(content)

            # Add unique document ID
            ids.append(
                f"doc_{index}"
            )

    # Store documents in ChromaDB
    if documents:

        collection.upsert(
            documents=documents,
            ids=ids
        )

    # Return collection
    return collection


# Search knowledge base using user requirement
def search_knowledge_base(query):

    # Load collection
    collection = load_knowledge_base()

    # Search top 3 relevant documents
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    # Extract documents
    documents = results.get(
        "documents",
        [[]]
    )[0]

    # Combine documents into one context
    rag_context = "\n\n".join(
        documents
    )

    # Return RAG context
    return rag_context