from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from app.core.config import get_settings
from app.db.base import init_db
from app.api import images, search

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

def create_application() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION, lifespan=lifespan)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    os.makedirs(settings.STATIC_FILES_PATH, exist_ok=True)
    app.mount("/static", StaticFiles(directory=settings.STATIC_FILES_PATH), name="static")

    app.include_router(images.router)
    app.include_router(search.router)
    
    @app.get("/")
    async def root():
        return {"message": "Image Similarity API is running!"}
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    return app

app = create_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)