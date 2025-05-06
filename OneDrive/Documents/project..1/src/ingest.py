import chromadb
from langchain_community.embeddings import OpenAIEmbeddings  # Update import for OpenAIEmbeddings
from langchain_community.vectorstores import Chroma  # Update import for Chroma

# Set up Chroma connection with persistent storage
client = chromadb.PersistentClient(path="chroma_db")

# Try deleting the old collection and create a new clean collection
collection_name = "youtube-video-qa"

try:
    if collection_name in [col.name for col in client.list_collections()]:
        client.delete_collection(collection_name)  # Delete old collection to avoid "Collection already exists"
    collection = client.create_collection(collection_name)
except Exception as e:
    print(f"❌ Error handling the database: {e}")

def ingest_text_to_chroma(text):
    """
    Convert text to vector representation using OpenAIEmbeddings and add to Chroma database.
    """
    if not text or len(text.strip()) == 0:
        print("⚠️ The entered text is empty or invalid.")
        return

    try:
        # Ensure collection exists or create a new one
        collection_name = "static_collection_name"  # استخدم اسم ثابت للمجموعة أو اسم ديناميكي
        collection = client.get_collection(collection_name) if client.get_collection(collection_name) else client.create_collection(collection_name)

        # Convert the text to vector representation
        embedding_model = OpenAIEmbeddings()
        embeddings = embedding_model.embed_query(text)

        # Check for duplicate text in the collection by comparing embeddings or another method
        existing_docs = collection.get()  # Get all documents in the collection (or use other checks if necessary)
        
        # Simple duplicate check based on the content (this can be improved)
        if any(doc["documents"] == text for doc in existing_docs["documents"]):
            print("⚠️ Text already exists in the database, avoiding duplication.")
            return

        # Add the text to Chroma database
        collection.add(
            documents=[text],
            embeddings=[embeddings],
            metadatas=[{"source": "video", "source_info": "Video transcript"}],
            ids=[str(collection.count() + 1)]  # Use dynamic ID to avoid duplication
        )

        print("✅ Text successfully added to Chroma database.")
    
    except Exception as e:
        print(f"❌ Error adding data to Chroma: {e}")
