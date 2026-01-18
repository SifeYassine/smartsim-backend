from pydantic import BaseModel, field_validator
from typing import Optional
from fastapi import Form

class SearchByTextRequest(BaseModel):
    query: str
    top_k: int = 5
    similarity_threshold: float = 0.12
    
    @field_validator('query')
    @classmethod
    def query_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query text cannot be empty")
        return v.strip()
    
    @field_validator('top_k')
    @classmethod
    def top_k_must_be_valid(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError("top_k must be between 1 and 100")
        return v
    
    @field_validator('similarity_threshold')
    @classmethod
    def threshold_must_be_valid(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        return v
    
    @classmethod
    def as_form(cls, query: str = Form(...), top_k: int = Form(5), similarity_threshold: float = Form(0.12)):
        return cls(query=query, top_k=top_k, similarity_threshold=similarity_threshold)


class SearchByImageRequest(BaseModel):
    image_path: str
    top_k: int = 5
    similarity_threshold: float = 0.12
    
    @field_validator('image_path')
    @classmethod
    def path_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Image path cannot be empty")
        return v.strip()
    
    @field_validator('top_k')
    @classmethod
    def top_k_must_be_valid(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError("top_k must be between 1 and 100")
        return v
    
    @field_validator('similarity_threshold')
    @classmethod
    def threshold_must_be_valid(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        return v
    
    @classmethod
    def as_form(cls, image_path: str = Form(...), top_k: int = Form(5), similarity_threshold: float = Form(0.12)):
        return cls(image_path=image_path, top_k=top_k, similarity_threshold=similarity_threshold)


class SearchResult(BaseModel):
    filename: str
    similarity: float

class SearchResponse(BaseModel):
    results: list[SearchResult]
    count: int

class TextSearchResponse(SearchResponse):
    query: str

class ImageSearchResponse(SearchResponse):
    query_image: str

class ClusterRequest(BaseModel):
    max_num_clusters: Optional[int] = None
    similarity_threshold: Optional[float] = None
    min_images_per_cluster: Optional[int] = None
    max_images_per_cluster: Optional[int] = None
    
    @field_validator('max_num_clusters')
    @classmethod
    def max_clusters_must_be_valid(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 2:
            raise ValueError("max_num_clusters must be at least 2")
        return v
    
    @field_validator('similarity_threshold')
    @classmethod
    def threshold_must_be_valid(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        return v
    
    @field_validator('min_images_per_cluster', 'max_images_per_cluster')
    @classmethod
    def cluster_size_must_be_valid(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("Cluster size must be at least 1")
        return v
    
    @classmethod
    def as_form(
        cls,
        max_num_clusters: Optional[int] = Form(None),
        similarity_threshold: Optional[float] = Form(None),
        min_images_per_cluster: Optional[int] = Form(None),
        max_images_per_cluster: Optional[int] = Form(None)
    ):

        return cls(
            max_num_clusters=max_num_clusters,
            similarity_threshold=similarity_threshold,
            min_images_per_cluster=min_images_per_cluster,
            max_images_per_cluster=max_images_per_cluster
        )


class ClusterResponse(BaseModel):
    clusters: dict[int, list[str]]
    cluster_count: int
    total_images: int