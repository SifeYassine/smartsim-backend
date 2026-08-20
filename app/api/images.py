from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.deps import get_image_crud, get_image_upload_service
from app.crud.image import ImageCRUD
from app.services.image_upload_service import ImageUploadService
from app.schemas.image import ImageListResponse, ImageResponse, UploadFolderResponse, UploadImageResponse, ClearImagesResponse

router = APIRouter(prefix="/images", tags=["images"])

@router.post("/upload-folder", response_model=UploadFolderResponse)
async def upload_folder(files: list[UploadFile] = File(...), db: AsyncSession = Depends(get_db), image_upload_service: ImageUploadService = Depends(get_image_upload_service)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        result = await image_upload_service.upload_folder(db, files)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result['new_images'] == 0:
        message = "All uploaded images already exist in the database"
    else:
        message = f"Successfully uploaded and indexed {result['new_images']} images"

    return UploadFolderResponse(message=message, **result)


@router.post("/upload-image", response_model=UploadImageResponse)
async def upload_image(file: UploadFile = File(...), db: AsyncSession = Depends(get_db), image_upload_service: ImageUploadService = Depends(get_image_upload_service)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        result = await image_upload_service.upload_single_image(db, file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if result['already_existed']:
        message = "File already exists in the database"
    else:
        message = "File uploaded successfully"

    return UploadImageResponse(message=message, **result)


@router.delete("/clear", response_model=ClearImagesResponse)
async def clear_images(db: AsyncSession = Depends(get_db), image_upload_service: ImageUploadService = Depends(get_image_upload_service)):
    result = await image_upload_service.clear_all_images(db)
    return ClearImagesResponse(**result)