from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import UploadFile
import os

from app.services.file_storage_service import FileStorageService
from app.services.embedding_service import EmbeddingService
from app.crud.image import ImageCRUD

class ImageUploadService:    
    def __init__(self, file_storage_service: FileStorageService, embedding_service: EmbeddingService, image_crud: ImageCRUD):
        self.file_storage_service = file_storage_service
        self.embedding_service = embedding_service
        self.image_crud = image_crud
    
    async def upload_folder(self, db: AsyncSession, files: list[UploadFile]) -> dict:
        validation_result = self.validate_files(files)
        if validation_result['error']:
            return validation_result
        
        valid_files = validation_result['valid_files']
        invalid_files = validation_result['invalid_files']
        
        filter_result = await self.filter_existing_files(db, valid_files)
        new_files = filter_result['new_files']
        already_exists = filter_result['already_exists']
        
        if not new_files:
            return {
                'success': True,
                'message': "All uploaded images already exist in the database",
                'data': {
                    'total_uploaded': 0,
                    'total_valid': len(valid_files),
                    'invalid_files': invalid_files,
                    'failed_saves': [],
                    'new_images': 0,
                    'already_indexed': len(already_exists)
                }
            }
        
        save_result = await self.save_files(new_files)
        saved_files = save_result['saved_files']
        failed_saves = save_result['failed_saves']
        
        if not saved_files:
            return {
                'success': False,
                'error': f"Failed to save any files. Errors: {failed_saves}"
            }
        
        index_result = await self.index_images(db, saved_files)
        
        if not index_result['success']:
            for file_path in saved_files:
                self.file_storage_service.delete_file(file_path)
            return index_result
        
        return {
            'success': True,
            'message': f"Successfully uploaded and indexed {index_result['indexed_count']} images",
            'data': {
                'total_uploaded': len(saved_files),
                'total_valid': len(valid_files),
                'invalid_files': invalid_files,
                'failed_saves': failed_saves,
                'new_images': index_result['indexed_count'],
                'already_indexed': len(already_exists)
            }
        }
    
    async def upload_single_image(self, file: UploadFile) -> dict:
        is_valid, error = self.file_storage_service.validate_image_file(file)
        if not is_valid:
            return {'success': False, 'error': error}
        
        filename = os.path.basename(file.filename)
        file_path = self.file_storage_service.get_unique_filepath(filename, subdirectory="uploads")
        
        if file_path.exists():
            return {
                'success': True,
                'data': {
                    'message': "File already exists",
                    'filename': filename,
                    'file_path': str(file_path).replace(os.sep, '/'),
                    'already_existed': True
                }
            }
        
        try:
            await self.file_storage_service.save_file(file, file_path)
        except Exception as e:
            return {'success': False, 'error': f"Failed to save file: {str(e)}"}
        
        return {
            'success': True,
            'data': {
                'message': "File uploaded successfully",
                'filename': filename,
                'file_path': str(file_path).replace(os.sep, '/'),
                'already_existed': False
            }
        }
    
    async def clear_all_images(self, db: AsyncSession) -> dict:
        await self.image_crud.delete_all(db)

        try:
            self.file_storage_service.clear_directory()
        except Exception as e:
            print(f"Warning: Could not clean up static files: {e}")
        
        return {
            'success': True,
            'message': "All images cleared"
        }
    
    def validate_files(self, files: list[UploadFile]) -> dict:
        valid_files = []
        invalid_files = []
        
        for file in files:
            is_valid, error = self.file_storage_service.validate_image_file(file)
            if is_valid:
                valid_files.append(file)
            else:
                invalid_files.append(file.filename)
        
        if not valid_files:
            return {
                'error': f"No valid image files found. Invalid files: {invalid_files}",
                'valid_files': [],
                'invalid_files': invalid_files
            }
        
        return {
            'error': None,
            'valid_files': valid_files,
            'invalid_files': invalid_files
        }
    
    async def filter_existing_files(self, db: AsyncSession, files: list[UploadFile]) -> dict:
        existing_filenames = await self.image_crud.get_all_filenames(db)
        
        new_files = []
        already_exists = []
        
        for file in files:
            filename = os.path.basename(file.filename)
            if filename in existing_filenames:
                already_exists.append(filename)
            else:
                new_files.append(file)
        
        return {
            'new_files': new_files,
            'already_exists': already_exists
        }
    
    async def save_files(self, files: list[UploadFile]) -> dict:
        saved_files = []
        failed_saves = []
        
        for file in files:
            try:
                filename = os.path.basename(file.filename)
                file_path = self.file_storage_service.get_unique_filepath(filename)
                await self.file_storage_service.save_file(file, file_path)
                saved_files.append(str(file_path))
            except Exception as e:
                failed_saves.append(f"{file.filename}: {str(e)}")
        
        return {
            'saved_files': saved_files,
            'failed_saves': failed_saves
        }
    
    async def index_images(self, db: AsyncSession, file_paths: list[str]) -> dict:
        print(f"Indexing {len(file_paths)} new images...")
        
        try:
            features = self.embedding_service.extract_images_features(file_paths)
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to extract features: {str(e)}"
            }
        
        indexed_count = 0
        for img_path, feature in zip(file_paths, features):
            if feature is not None:
                filename = os.path.basename(img_path)
                img_path_normalized = img_path.replace(os.sep, '/')
                await self.image_crud.create(db, filename, img_path_normalized, feature.tolist())
                indexed_count += 1
            else:
                self.file_storage_service.delete_file(img_path)
        
        await db.commit()
        
        return {
            'success': True,
            'indexed_count': indexed_count
        }