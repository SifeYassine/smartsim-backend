from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, func, bindparam
from typing import Optional
import numpy as np
from app.db.types import Vector
from app.models.image import Image

class ImageCRUD:    
    async def get_all_filenames(self, db: AsyncSession) -> set[str]:
        result = await db.execute(select(Image.filename))
        return {row.filename for row in result.fetchall()}

    async def get_by_filename(self, db: AsyncSession, filename: str) -> Optional[Image]:
        result = await db.execute(select(Image).where(Image.filename == filename))
        return result.scalar_one_or_none()
    
    async def get_all(self, db: AsyncSession) -> list[Image]:
        result = await db.execute(select(Image))
        return list(result.scalars().all())
    
    async def get_all_with_embeddings(self, db: AsyncSession) -> list[dict]:
        result = await db.execute(
            select(Image.id, Image.filename, Image.file_path, Image.embedding)
            .where(Image.embedding.isnot(None))
        )
        
        return [{'id': row.id, 'filename': row.filename, 'file_path': row.file_path, 'embedding': row.embedding} for row in result]
    
    async def vector_search(self, db: AsyncSession, query_embedding: list[float], top_k: int, similarity_threshold: float) -> list[tuple[str, float]]:
        query_vector = bindparam("query_vector", value=query_embedding, type_=Vector())
        distance = func.vec_distance_cosine(Image.embedding, query_vector)
        similarity = 1 - distance

        result = await db.execute(
            select(Image.filename, similarity.label("similarity"))
            .where(Image.embedding.isnot(None))
            .where(similarity >= similarity_threshold)
            .order_by(distance)
            .limit(top_k)
        )

        return [(row.filename, row.similarity) for row in result.fetchall()]
    
    async def create(self, db: AsyncSession, filename: str, file_path: str, embedding: list[float]) -> Image:
        result = await db.execute(insert(Image).values(filename=filename, file_path=file_path, embedding=embedding).returning(Image))
        return result.scalar_one()
    
    async def delete_all(self, db: AsyncSession) -> None:
        await db.execute(delete(Image))
    
    async def update_cluster_info(self, db: AsyncSession, image_ids: list[int], cluster_id: int, similarity_threshold: Optional[float], max_clusters: Optional[int]) -> None:
        await db.execute(
            update(Image)
            .where(Image.id.in_(image_ids))
            .values(cluster_id=cluster_id, similarity_threshold=similarity_threshold, max_clusters=max_clusters, clustered_at=func.now())
        )
    
    async def clear_cluster_info(self, db: AsyncSession) -> None:
        await db.execute(update(Image).values(cluster_id=None,similarity_threshold=None, max_clusters=None, clustered_at=None))