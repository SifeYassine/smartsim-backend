from sqlalchemy.ext.asyncio import AsyncSession
from app.services.embedding_service import EmbeddingService
from app.crud.image import ImageCRUD

class SimilarityService:    
    def __init__(self, embedding_service: EmbeddingService, image_crud: ImageCRUD):
        self.embedding_service = embedding_service
        self.image_crud = image_crud
    
    async def search_by_text(self, db: AsyncSession, query_text: str, top_k: int = 5, similarity_threshold: float = 0.12) -> list[tuple[str, float]]:
        query_features = self.embedding_service.extract_text_features(query_text)
        if query_features is None:
            return []
        
        results = await self.image_crud.vector_search(db, query_features, top_k, similarity_threshold)
        return results
    
    async def search_by_image(self, db: AsyncSession, image_path: str, top_k: int = 5, similarity_threshold: float = 0.2) -> list[tuple[str, float]]:
        query_features = self.embedding_service.extract_image_features(image_path)
        if query_features is None:
            return []
        
        results = await self.image_crud.vector_search(db, query_features, top_k, similarity_threshold)
        return results