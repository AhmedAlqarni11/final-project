from dotenv import load_dotenv
import os

# Load variables from .env file
load_dotenv()

# Retrieve keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"Pinecone API Key: {PINECONE_API_KEY}")
print(f"OpenAI API Key: {OPENAI_API_KEY}")
