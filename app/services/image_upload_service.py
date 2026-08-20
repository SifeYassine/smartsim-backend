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
        valid_files, invalid_files = self.validate_files(files)
        new_files, already_exists = await self.filter_existing_files(db, valid_files)

        if not new_files:
            return {
                'total_uploaded': 0,
                'total_valid': len(valid_files),
                'invalid_files': invalid_files,
                'failed_saves': [],
                'new_images': 0,
                'already_indexed': len(already_exists)
            }

        saved_files, failed_saves = await self.save_files(new_files)
        indexed_count = await self.index_images(db, saved_files)

        return {
            'total_uploaded': len(saved_files),
            'total_valid': len(valid_files),
            'invalid_files': invalid_files,
            'failed_saves': failed_saves,
            'new_images': indexed_count,
            'already_indexed': len(already_exists)
        }

    async def upload_single_image(self, db: AsyncSession, file: UploadFile) -> dict:
        is_valid, error = self.file_storage_service.validate_image_file(file)
        if not is_valid:
            raise ValueError(error)

        filename = os.path.basename(file.filename)
        existing = await self.image_crud.get_by_filename(db, filename)

        if existing:
            return {
                'filename': filename,
                'file_path': existing.file_path,
                'already_existed': True
            }

        file_path = self.file_storage_service.get_unique_filepath(filename)
        await self.file_storage_service.save_file(file, file_path)

        try:
            await self.index_images(db, [str(file_path)])
        except Exception as e:
            self.file_storage_service.delete_file(str(file_path))
            raise RuntimeError(f"Failed to index uploaded image: {e}")

        return {
            'filename': filename,
            'file_path': str(file_path).replace(os.sep, '/'),
            'already_existed': False
        }

    async def clear_all_images(self, db: AsyncSession) -> dict:
        await self.image_crud.delete_all(db)
        await db.commit()

        try:
            self.file_storage_service.clear_directory()
        except Exception as e:
            print(f"Warning: Could not clean up static files: {e}")

        return {'message': "All images cleared"}

    def validate_files(self, files: list[UploadFile]) -> tuple[list[UploadFile], list[str]]:
        valid_files, invalid_files = [], []

        for file in files:
            is_valid, error = self.file_storage_service.validate_image_file(file)
            if is_valid:
                valid_files.append(file)
            else:
                invalid_files.append(file.filename)

        if not valid_files:
            raise ValueError(f"No valid image files found. Invalid files: {invalid_files}")

        return valid_files, invalid_files

    async def filter_existing_files(self, db: AsyncSession, files: list[UploadFile]) -> tuple[list[UploadFile], list[str]]:
        existing_filenames = await self.image_crud.get_all_filenames(db)

        new_files, already_exists = [], []

        for file in files:
            filename = os.path.basename(file.filename)
            if filename in existing_filenames:
                already_exists.append(filename)
            else:
                new_files.append(file)

        return new_files, already_exists

    async def save_files(self, files: list[UploadFile]) -> tuple[list[str], list[str]]:
        saved_files, failed_saves = [], []

        for file in files:
            try:
                filename = os.path.basename(file.filename)
                file_path = self.file_storage_service.get_unique_filepath(filename)
                await self.file_storage_service.save_file(file, file_path)
                saved_files.append(str(file_path))
            except Exception as e:
                failed_saves.append(f"{file.filename}: {str(e)}")

        if not saved_files:
            raise RuntimeError(f"Failed to save any files. Errors: {failed_saves}")

        return saved_files, failed_saves

    async def index_images(self, db: AsyncSession, file_paths: list[str]) -> int:
        print(f"Indexing {len(file_paths)} new images...")

        try:
            features = self.embedding_service.extract_images_features(file_paths)
        except Exception as e:
            raise RuntimeError(f"Failed to extract features: {str(e)}")

        indexed_count = 0
        for img_path, feature in zip(file_paths, features):
            if feature is not None:
                filename = os.path.basename(img_path)
                img_path_normalized = img_path.replace(os.sep, '/')
                await self.image_crud.create(db, filename, img_path_normalized, feature.tolist())
                indexed_count = indexed_count + 1
            else:
                self.file_storage_service.delete_file(img_path)

        await db.commit()

        return indexed_count