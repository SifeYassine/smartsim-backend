from typing import Optional

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image
from tokenizers import Tokenizer
from tqdm import tqdm

# CLIP ViT-B/16 as ONNX (Xenova's export of openai/clip-vit-base-patch16) and OpenAI's tokenizer,
# pinned to the commits whose outputs match the PyTorch model.
ONNX_REPO = "Xenova/clip-vit-base-patch16"
ONNX_REVISION = "342fdf2f67aded64d138ff074745fb4a5d2bba5f"
TOKENIZER_REVISION = "5ef227a78de3f75873f373246dac80def63b0003"

IMAGE_SIZE = 224
MAX_TEXT_TOKENS = 77
END_OF_TEXT_TOKEN = 49407
IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class EmbeddingService:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.vision_session = None
        self.text_session = None
        self.tokenizer = None
        self.load_model()

    def load_model(self) -> None:
        print("Loading CLIP model...")

        vision_path = hf_hub_download(ONNX_REPO, "onnx/vision_model.onnx", revision=ONNX_REVISION, cache_dir=self.cache_dir)
        text_path = hf_hub_download(ONNX_REPO, "onnx/text_model.onnx", revision=ONNX_REVISION, cache_dir=self.cache_dir)
        tokenizer_path = hf_hub_download(self.model_name, "tokenizer.json", revision=TOKENIZER_REVISION, cache_dir=self.cache_dir)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.vision_session = ort.InferenceSession(vision_path, options, providers=["CPUExecutionProvider"])
        self.text_session = ort.InferenceSession(text_path, options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        print("CLIP model loaded successfully!")

    def extract_images_features(self, image_paths: list[str], batch_size: int = 16) -> list[Optional[np.ndarray]]:
        all_features = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting image features"):
            batch_paths = image_paths[i:i + batch_size]
            batch_features = self.extract_batch_features(batch_paths)
            all_features.extend(batch_features)

        return all_features

    def extract_batch_features(self, image_paths: list[str]) -> list[Optional[np.ndarray]]:
        """One entry per path, None where the image could not be read, so callers can pair
        results with their files."""
        batch_pixels = []

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                batch_pixels.append(self.preprocess(img))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                batch_pixels.append(None)

        loaded = [pixels for pixels in batch_pixels if pixels is not None]
        if not loaded:
            return [None] * len(image_paths)

        features = self.vision_session.run(["image_embeds"], {"pixel_values": np.stack(loaded)})[0]
        features = iter(self.normalize(features))

        return [None if pixels is None else next(features) for pixels in batch_pixels]

    def extract_text_features(self, text: str) -> Optional[np.ndarray]:
        try:
            ids = self.tokenizer.encode(text).ids
            if len(ids) > MAX_TEXT_TOKENS:
                ids = ids[:MAX_TEXT_TOKENS - 1] + [END_OF_TEXT_TOKEN]
            input_ids = np.array([ids], dtype=np.int64)
            inputs = {"input_ids": input_ids}
            if any(i.name == "attention_mask" for i in self.text_session.get_inputs()):
                inputs["attention_mask"] = np.ones_like(input_ids)

            features = self.text_session.run(["text_embeds"], inputs)[0]
            return self.normalize(features).reshape(-1)
        except Exception as e:
            print(f"Error extracting text features: {e}")
            return None

    def extract_image_features(self, image_path: str) -> Optional[np.ndarray]:
        try:
            features = self.extract_batch_features([image_path])
            return features[0] if features else None
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    @staticmethod
    def preprocess(img: Image.Image) -> np.ndarray:
        """CLIPProcessor's steps: shortest side to 224 (bicubic), 224 center crop, CLIP normalization."""
        width, height = img.size
        scale = IMAGE_SIZE / min(width, height)
        resized = img.resize((max(IMAGE_SIZE, int(width * scale)), max(IMAGE_SIZE, int(height * scale))), Image.Resampling.BICUBIC)
        left = (resized.width - IMAGE_SIZE) // 2
        top = (resized.height - IMAGE_SIZE) // 2
        cropped = resized.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
        pixels = (np.asarray(cropped, dtype=np.float32) / 255.0 - IMAGE_MEAN) / IMAGE_STD
        return pixels.transpose(2, 0, 1)

    @staticmethod
    def normalize(features: np.ndarray) -> np.ndarray:
        return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
