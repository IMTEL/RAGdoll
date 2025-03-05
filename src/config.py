import os
from dotenv import load_dotenv

# Adjust the path if necessary. If your .env file is in the project root, use:
load_dotenv()  # Loads .env from the current working directory

# Alternatively, if it's located in a specific directory:
# load_dotenv("../.env")

class Config:
    def __init__(self, path=".env", gpt_model="gpt-4o-mini"):
        self.path = path
        self.ENV = os.getenv("ENV", "test")
        
        self.GPT_MODEL = os.getenv("GPT_MODEL", gpt_model)
        self.API_KEY = os.getenv("OPENAI_API_KEY", "your_default_api_key")
        
        
        if self.ENV == 'testsgrsj': # TODO: change this to 'test' when ready
            self.MONGODB_URI = os.getenv("MOCK_MONGODB_URI", "mongodb://localhost:27017")
            self.MONGODB_COLLECTION = os.getenv("MOCK_MONGODB_COLLECTION", "test_collection")
            self.MONGODB_DATABASE = os.getenv("MOCK_MONGODB_DATABASE", "test_database")
            self.RAG_DATABASE_SYSTEM = os.getenv("MOCK_RAG_DATABASE_SYSTEM", "mongodb")
        else:
            self.MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            self.MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "test_collection")
            self.MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "test_database")
            self.RAG_DATABASE_SYSTEM = os.getenv("RAG_DATABASE_SYSTEM", "mongodb")

