from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.deps import get_similarity_service, get_clustering_service
from app.services.similarity_service import SimilarityService
from app.services.clustering_service import ClusteringService
from app.schemas.search import SearchByTextRequest, SearchByImageRequest, TextSearchResponse, ImageSearchResponse, ClusterRequest, ClusterResponse, SearchResult

router = APIRouter(prefix="/search", tags=["search"])

@router.post("/by-text", response_model=TextSearchResponse)
async def search_by_text(
    request: SearchByTextRequest = Depends(SearchByTextRequest.as_form),
    db: AsyncSession = Depends(get_db),
    similarity_service: SimilarityService = Depends(get_similarity_service)
):
    results = await similarity_service.search_by_text(db, request.query, request.top_k, request.similarity_threshold)
    
    return TextSearchResponse(
        query=request.query,
        results=[SearchResult(filename=filename, similarity=similarity) for filename, similarity in results],
        count=len(results)
    )


@router.post("/by-image", response_model=ImageSearchResponse)
async def search_by_image(
    request: SearchByImageRequest = Depends(SearchByImageRequest.as_form),
    db: AsyncSession = Depends(get_db),
    similarity_service: SimilarityService = Depends(get_similarity_service)
):
    results = await similarity_service.search_by_image(db, request.image_path, request.top_k, request.similarity_threshold)
    
    return ImageSearchResponse(
        query_image=request.image_path,
        results=[SearchResult(filename=filename, similarity=similarity) for filename, similarity in results],
        count=len(results)
    )


@router.post("/cluster", response_model=ClusterResponse)
async def cluster_images(
    request: ClusterRequest = Depends(ClusterRequest.as_form),
    db: AsyncSession = Depends(get_db),
    clustering_service: ClusteringService = Depends(get_clustering_service)
):
    clusters = await clustering_service.cluster_images(
        db,
        max_num_clusters=request.max_num_clusters,
        similarity_threshold=request.similarity_threshold,
        min_images_per_cluster=request.min_images_per_cluster,
        max_images_per_cluster=request.max_images_per_cluster
    )
    
    return ClusterResponse(clusters=clusters, cluster_count=len(clusters), total_images=sum(len(images) for images in clusters.values()))