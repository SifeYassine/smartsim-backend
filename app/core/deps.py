from functools import lru_cache
from fastapi import Depends

from app.core.config import Settings, get_settings
from app.crud.image import ImageCRUD

from app.services.embedding_service import EmbeddingService
from app.services.similarity_service import SimilarityService
from app.services.clustering_service import ClusteringService
from app.services.file_storage_service import FileStorageService
from app.services.image_upload_service import ImageUploadService


@lru_cache()
def get_image_crud() -> ImageCRUD:
    return ImageCRUD()


@lru_cache()
def get_embedding_service(settings: Settings = Depends(get_settings)) -> EmbeddingService:
    return EmbeddingService(model_name=settings.CLIP_MODEL_NAME, cache_dir=settings.CLIP_CACHE_DIR)


@lru_cache()
def get_file_storage_service(settings: Settings = Depends(get_settings)) -> FileStorageService:
    return FileStorageService(static_path=settings.STATIC_FILES_PATH)


def get_similarity_service(embedding_service: EmbeddingService = Depends(get_embedding_service), image_crud: ImageCRUD = Depends(get_image_crud)) -> SimilarityService:
    return SimilarityService(embedding_service=embedding_service, image_crud=image_crud)


def get_clustering_service(image_crud: ImageCRUD = Depends(get_image_crud)) -> ClusteringService:
    return ClusteringService(image_crud=image_crud)


def get_image_upload_service(
    file_storage_service: FileStorageService = Depends(get_file_storage_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    image_crud: ImageCRUD = Depends(get_image_crud)
) -> ImageUploadService:
    return ImageUploadService(file_storage_service=file_storage_service, embedding_service=embedding_service, image_crud=image_crud)