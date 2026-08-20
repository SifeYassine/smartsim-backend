from pathlib import Path
from fastapi import UploadFile
import aiofiles
import shutil
import os

class FileStorageService:    
    VALID_EXTENSIONS = ('png', 'jpg', 'jpeg', 'jfif', 'webp')
    
    def __init__(self, static_path: str):
        self.static_path = Path(static_path)
        self.static_path.mkdir(parents=True, exist_ok=True)
    
    def validate_image_file(self, file: UploadFile) -> tuple[bool, str]:
        if not file.content_type or not file.content_type.startswith('image/'):
            return False, "File must be an image"
        
        file_ext = Path(file.filename).suffix.lower().lstrip('.')
        if file_ext not in self.VALID_EXTENSIONS:
            return False, f"Invalid extension. Allowed: {self.VALID_EXTENSIONS}"
        
        return True, ""
    
    def get_unique_filepath(self, filename: str) -> Path:
        file_path = self.static_path / filename
        if not file_path.exists():
            return file_path
        
        counter = 1
        stem = file_path.stem
        suffix = file_path.suffix
        
        while file_path.exists():
            file_path = self.static_path / f"{stem}_{counter}{suffix}"
            counter = counter + 1
            
        return file_path
    
    async def save_file(self, file: UploadFile, file_path: Path) -> None:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
    
    def delete_file(self, file_path: str | Path) -> None:
        try:
            os.remove(file_path)
        except Exception:
            pass
    
    def clear_directory(self) -> None:
        if self.static_path.exists():
            shutil.rmtree(self.static_path)
            self.static_path.mkdir(parents=True, exist_ok=True)