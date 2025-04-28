import os
from dotenv import load_dotenv
# import whisper # No longer needed here

load_dotenv()  


class Config:
    def __init__(self, path=".env", gpt_model="gpt-4o-mini", gemini_model="gemini-2.0-flash-lite"):
        self.path = path
        self.ENV = os.getenv("ENV", "dev")
        
        self.GPT_MODEL = os.getenv("GPT_MODEL", gpt_model)
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", gemini_model)
        self.API_KEY = os.getenv("OPENAI_API_KEY", "your_default_api_key")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_default_gemini_api_key")
        
        # Faster-Whisper Configuration
        # Options: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v1, large-v2, large-v3, distil-large-v3 etc.
        self.whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "base") 
        # Options: "cuda", "cpu"
        self.whisper_device = os.getenv("WHISPER_DEVICE", "cpu") 
        # Options for CPU: "int8", "float32"
        # Options for CUDA: "float16", "int8_float16", "int8"
        default_compute_type = "int8" if self.whisper_device == "cpu" else "float16"
        self.whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", default_compute_type)

        # Remove direct model loading from config
        # try:
        #     self.whisper_model = whisper.load_model("base").to("cuda") # TODO: load model on ping or keep container warm
        # except:
        #     self.whisper_model = whisper.load_model("base")
        
        if self.ENV == 'dev': # TODO: change this to 'dev' when ready
            self.MONGODB_URI = os.getenv("MOCK_MONGODB_URI", "mongodb://localhost:27017")
            self.MONGODB_COLLECTION = os.getenv("MOCK_MONGODB_COLLECTION", "test_collection")
            self.MONGODB_DATABASE = os.getenv("MOCK_MONGODB_DATABASE", "test_database")
            self.RAG_DATABASE_SYSTEM = os.getenv("MOCK_RAG_DATABASE_SYSTEM", "mongodb")
        else:
            self.MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            self.MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "test_collection")
            self.MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "test_database")
            self.RAG_DATABASE_SYSTEM = os.getenv("RAG_DATABASE_SYSTEM", "mongodb")

