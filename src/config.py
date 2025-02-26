import os
from dotenv import load_dotenv

"""
manage configuration variables
"""

load_dotenv("../.env") # .env? file path er vanskelig 


class Config:
    def __init__(self, path=".env", gpt_model="gpt-4o-mini"):  # TODO make sure 
        self.path = path
        self.GPT_MODEL = os.getenv(key="GPT_MODEL", default=gpt_model)
        self.API_KEY = os.getenv("OPENAI_API_KEY")
        self.MONGODB_URI = os.getenv("MONGODB_URI")
        self.MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION")
        self.MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
        self.RAG_DATABASE_SYSTEM = os.getenv("RAG_DATABASE_SYSTEM", "mongodb")
        self.BASE_URL_FRONTEND = os.getenv("BASE_URL_FRONTEND", "http://localhost:8080")
