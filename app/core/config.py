from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:@localhost:5432/smartsimdb"
    
    STATIC_FILES_PATH: str = "./static/images"
    
    CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    CLIP_CACHE_DIR: str = "./cache"
    
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    
    API_TITLE: str = "Image Similarity API"
    API_VERSION: str = "1.0.0"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()