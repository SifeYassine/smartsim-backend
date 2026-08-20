from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///db/smartsim.db"
    
    DB_FILE_PATH: str = "./db"
    STATIC_FILES_PATH: str = "./static/images"
    
    MODEL_NAME: str = "openai/clip-vit-base-patch16"
    MODEL_CACHE_DIR: str = "./cache"
    
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    API_TITLE: str = "Image Similarity API"
    API_VERSION: str = "1.0.0"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()