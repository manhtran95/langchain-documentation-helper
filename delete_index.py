import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

if __name__ == "__main__":
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["INDEX_NAME"])
    index.delete(delete_all=True)
    print("All vectors deleted.")
