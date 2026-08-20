from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from app.crud.image import ImageCRUD

class ClusteringService:
    def __init__(self, image_crud: ImageCRUD):
        self.image_crud = image_crud

    async def cluster_images(
        self,
        db: AsyncSession,
        max_num_clusters: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        min_images_per_cluster: Optional[int] = None,
        max_images_per_cluster: Optional[int] = None
    ) -> dict[int, list[str]]:

        images_data = await self.image_crud.get_all_with_embeddings(db)
        
        if len(images_data) < 2:
            return {}
        
        embeddings = np.array([np.array(img['embedding']) for img in images_data])
        filenames = [img['filename'] for img in images_data]
        image_ids = [img['id'] for img in images_data]
        
        labels = self.perform_clustering(embeddings, max_num_clusters, similarity_threshold)
        clusters, cluster_image_ids = self.group_clusters(labels, filenames, image_ids)
        clusters = self.filter_clusters(clusters, min_images_per_cluster, max_images_per_cluster)
        
        await self.save_clustering_results(db, cluster_image_ids, similarity_threshold, max_num_clusters, clusters)
        
        return { int(k): v for k, v in clusters.items() }
    
    def perform_clustering(self, embeddings: np.ndarray, max_num_clusters: Optional[int], similarity_threshold: Optional[float]) -> list[int]:
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
        
        if max_num_clusters is not None and similarity_threshold is None:
            clustering = AgglomerativeClustering(n_clusters=max_num_clusters, metric='precomputed', linkage='average')

        elif max_num_clusters is None and similarity_threshold is not None:
            clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=1 - similarity_threshold)

        else:
            clustering = AgglomerativeClustering(n_clusters=10, metric='precomputed', linkage='average')
        
        labels = clustering.fit_predict(distance_matrix)
        return [ int(label) for label in labels ]
    
    def group_clusters(self, labels: np.ndarray, filenames: list[str], image_ids: list[int]) -> tuple[dict[int, list[str]], dict[int, list[int]]]:
        clusters = defaultdict(list)
        cluster_image_ids = defaultdict(list)
        
        for i, label in enumerate(labels):
            clusters[label].append(filenames[i])
            cluster_image_ids[label].append(image_ids[i])
        
        return clusters, cluster_image_ids
    
    def filter_clusters(self, clusters: dict[int, list[str]], min_images_per_cluster: Optional[int], max_images_per_cluster: Optional[int]) -> dict[int, list[str]]:
        filtered = clusters.copy()
        
        if min_images_per_cluster is not None:
            filtered = { key: value for key, value in filtered.items() if len(value) >= min_images_per_cluster }
        
        if max_images_per_cluster is not None:
            filtered = { key: value for key, value in filtered.items() if len(value) <= max_images_per_cluster }
        
        return filtered
    
    async def save_clustering_results(
        self,
        db: AsyncSession,
        cluster_image_ids: dict[int, list[int]],
        similarity_threshold: Optional[float],
        max_clusters: Optional[int],
        filtered_clusters: dict[int, list[str]]
    ) -> None:

        await self.image_crud.clear_cluster_info(db)
        
        for cluster_id, image_ids in cluster_image_ids.items():
            if cluster_id in filtered_clusters:
                await self.image_crud.update_cluster_info(db, image_ids, cluster_id, similarity_threshold, max_clusters)
        
        await db.commit()