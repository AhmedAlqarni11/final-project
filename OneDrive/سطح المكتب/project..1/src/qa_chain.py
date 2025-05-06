import chromadb
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from ingest import get_collection_for_video
# Set up Chroma client with persistent storage
client = chromadb.PersistentClient(path="chroma_db")

# Check for collection existence or create it if not present
collection_name = "youtube-video-qa"

try:
    if collection_name in [col.name for col in client.list_collections()]:
        collection = client.get_collection(collection_name)
    else:
        collection = client.create_collection(collection_name)
except Exception as e:
    print(f"❌ Error setting up the database: {e}")
    collection = None

# Code that handles retrieving the answer from Chroma
def get_answer(user_input: str, url: str) -> str:
    collection = get_collection_for_video(url)  # Get the collection based on the URL
    if collection is None:
        return "⚠️ Error connecting to the database."

    try:
        # Set up Chroma retriever with OpenAI embeddings
        embedding_model = OpenAIEmbeddings()
        vectorstore = Chroma(client=client, embedding_function=embedding_model)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Set up LangChain model for searching and answering
        qa_model = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(),
            retriever=retriever
        )

        # Run the search and return the answer
        answer = qa_model.run(user_input)

        if "❌" in answer:
            model = ChatOpenAI()
            response = model.run(f"What is the Generative Fill feature in Photoshop and how does it help create realistic environments for video?")
            return response

        return answer

    except Exception as e:
        print(f"❌ Error while searching for the answer: {e}")
        return "⚠️ Error while retrieving the answer, try again later."
