from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.db.base import Base

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, unique=True)
    file_path = Column(String(500), nullable=False)
    embedding = Column(Vector(512))
    
    cluster_id = Column(Integer, nullable=True)
    similarity_threshold = Column(Float, nullable=True)
    max_clusters = Column(Integer, nullable=True)
    clustered_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())