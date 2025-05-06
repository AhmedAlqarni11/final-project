from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import chromadb

# Set up Chroma connection with persistent storage
client = chromadb.PersistentClient(path="chroma_db")

# Load the collection or create it if it doesn't exist
collection_name = "youtube-video-qa"
try:
    existing_collections = [col.name for col in client.list_collections()]
    if collection_name in existing_collections:
        collection = client.get_collection(collection_name)
    else:
        collection = client.create_collection(collection_name)
except Exception as e:
    print(f"❌ Error setting up the database: {e}")
    collection = None

def answer_question(user_input: str):
    """
    Retrieve the answer based on the user's question using LangChain and ChromaDB.
    The answer should only be based on the content available in the database.
    """
    if collection is None:
        return "⚠️ Error connecting to the database."

    try:
        # Load embedding model
        embedding_model = OpenAIEmbeddings()

        # Create vectorstore using Chroma
        vectorstore = Chroma(client=client, embedding_function=embedding_model)

        # Set up the retriever for efficient searching
        retriever = vectorstore.as_retriever()

        # Define the tool used to retrieve answers
        tools = [
            Tool(
                name="Answer Question",
                func=retriever.get_relevant_documents,  # Retrieve relevant documents based on the question
                description="Search the database and retrieve relevant documents based on the user's input."
            )
        ]

        # Set up the Agent using ChatOpenAI for smarter responses
        agent = initialize_agent(tools, ChatOpenAI(), agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

        # Run the search and return the answer
        answer = agent.run(user_input)

        # If no answer is found, use GPT to generate one
        if "❌" in answer:
            model = ChatOpenAI()
            response = model.run(f"What is the Generative Fill feature in Photoshop and how does it help create realistic environments for video?")
            return response

        return answer

    except Exception as e:
        return f"❌ Error retrieving the answer: {e}"
