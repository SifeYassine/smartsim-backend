from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete, text
from sqlalchemy.sql import func
from typing import Optional
import numpy as np
from app.models.image import Image

class ImageCRUD:    
    async def get_all_filenames(self, db: AsyncSession) -> set[str]:
        result = await db.execute(select(Image.filename))
        return {row.filename for row in result.fetchall()}
    
    async def get_all(self, db: AsyncSession) -> list[Image]:
        result = await db.execute(select(Image))
        return list(result.scalars().all())
    
    async def get_all_with_embeddings(self, db: AsyncSession) -> list[dict]:
        sql_query = text("""SELECT id, filename, file_path, embedding::text FROM images WHERE embedding IS NOT NULL""")
        result = await db.execute(sql_query)
        return [{'id': row.id, 'filename': row.filename, 'file_path': row.file_path, 'embedding': eval(row.embedding)} for row in result.fetchall()]
    
    async def vector_search(self, db: AsyncSession, query_embedding: np.ndarray, top_k: int, similarity_threshold: float) -> list[tuple[str, float]]:
        query_vector = query_embedding.tolist()
        
        sql_query = text("""
            SELECT filename, file_path, 1 - (embedding <=> :query_vector) as similarity
            FROM images
            WHERE 1 - (embedding <=> :query_vector) >= :threshold
            ORDER BY embedding <=> :query_vector
            LIMIT :limit
        """)
        
        result = await db.execute(sql_query, {"query_vector": str(query_vector), "threshold": similarity_threshold, "limit": top_k})
        return [(row.filename, row.similarity) for row in result.fetchall()]
    
    async def create(self, db: AsyncSession, filename: str, file_path: str, embedding: list[float]) -> Image:
        result = await db.execute(insert(Image).values(filename=filename, file_path=file_path, embedding=embedding).returning(Image))
        return result.scalar_one()
    
    async def delete_all(self, db: AsyncSession) -> None:
        await db.execute(delete(Image))
        await db.commit()
    
    async def update_cluster_info(self, db: AsyncSession, image_ids: list[int], cluster_id: int, similarity_threshold: Optional[float], max_clusters: Optional[int]) -> None:
        await db.execute(
            update(Image)
            .where(Image.id.in_(image_ids))
            .values(cluster_id=cluster_id, similarity_threshold=similarity_threshold, max_clusters=max_clusters, clustered_at=func.now())
        )
    
    async def clear_cluster_info(self, db: AsyncSession) -> None:
        await db.execute(update(Image).values(cluster_id=None,similarity_threshold=None, max_clusters=None, clustered_at=None))