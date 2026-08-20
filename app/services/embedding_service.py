import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import Optional
from tqdm import tqdm

class EmbeddingService:    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self) -> None:
        print("Loading CLIP model...")

        self.model = CLIPModel.from_pretrained(self.model_name, use_safetensors=True, cache_dir=self.cache_dir)
        self.processor = CLIPProcessor.from_pretrained(self.model_name, use_safetensors=True, use_fast=True, cache_dir=self.cache_dir)
        self.model.eval()
        self.model = self.model.float()

        print("CLIP model loaded successfully!")
    
    def extract_images_features(self, image_paths: list[str], batch_size: int = 16) -> list[np.ndarray]:
        all_features = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting image features"):
            batch_paths = image_paths[i:i + batch_size]
            batch_features = self.extract_batch_features(batch_paths)
            all_features.extend(batch_features)
        
        return all_features
    
    def extract_batch_features(self, image_paths: list[str]) -> list[np.ndarray]:
        batch_images = []
        
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                batch_images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        if not batch_images:
            return []
        
        inputs = self.processor(images=batch_images, return_tensors="pt")
        
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = F.normalize(features, p=2, dim=1)
            features = features.cpu().numpy()
        
        return [feat for feat in features]
    
    def extract_text_features(self, text: str) -> Optional[np.ndarray]:
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                features = F.normalize(features, p=2, dim=1)
                features = features.cpu().numpy().reshape(-1)
            
            return features
        except Exception as e:
            print(f"Error extracting text features: {e}")
            return None
    
    def extract_image_features(self, image_path: str) -> Optional[np.ndarray]:
        try:
            img = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=img, return_tensors="pt")
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = F.normalize(features, p=2, dim=1)
                features = features.cpu().numpy().reshape(-1)
            
            return features
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None