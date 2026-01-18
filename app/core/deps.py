from functools import lru_cache
from app.core.config import Settings, get_settings
from app.crud.image import ImageCRUD
from app.services.embedding_service import EmbeddingService
from app.services.similarity_service import SimilarityService
from app.services.clustering_service import ClusteringService
from app.services.file_storage_service import FileStorageService
from app.services.image_upload_service import ImageUploadService

class ServiceContainer:    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.image_crud = None
        self.embedding_service = None
        self.similarity_service = None
        self.clustering_service = None
        self.file_storage_service = None
        self.image_uplaod_service = None
    
    @property
    def image_crud(self) -> ImageCRUD:
        if self.image_crud is None:
            self.image_crud = ImageCRUD()
        return self.image_crud
    
    @property
    def embedding_service(self) -> EmbeddingService:
        if self.embedding_service is None:
            self.embedding_service = EmbeddingService(model_name=self.settings.CLIP_MODEL_NAME, cache_dir=self.settings.CLIP_CACHE_DIR)
        return self.embedding_service
    
    @property
    def similarity_service(self) -> SimilarityService:
        if self.similarity_service is None:
            self.similarity_service = SimilarityService(self.embedding_service)
        return self.similarity_service
    
    @property
    def clustering_service(self) -> ClusteringService:
        if self.clustering_service is None:
            self.clustering_service = ClusteringService()
        return self.clustering_service
    
    @property
    def file_storage_service(self) -> FileStorageService:
        if self.file_storage_service is None:
            self.file_storage_service = FileStorageService(static_path=self.settings.STATIC_FILES_PATH)
        return self.file_storage_service
    
    @property
    def image_upload_service(self) -> ImageUploadService:
        if self.image_upload_service is None:
            self.image_upload_service = ImageUploadService(file_storage_service=self.file_storage_service, embedding_service=self.embedding_service)
        return self.image_upload_service
    
@lru_cache()
def get_service_container() -> ServiceContainer:
    settings = get_settings()
    return ServiceContainer(settings)