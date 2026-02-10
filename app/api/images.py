from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.deps import get_image_crud, get_image_upload_service
from app.crud.image import ImageCRUD
from app.services.image_upload_service import ImageUploadService
from app.schemas.image import ImageListResponse, ImageResponse, UploadFolderResponse, UploadImageResponse, ClearImagesResponse

router = APIRouter(prefix="/images", tags=["images"])

@router.post("/upload-folder", response_model=UploadFolderResponse)
async def upload_folder(
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    image_upload_service: ImageUploadService = Depends(get_image_upload_service)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    result = await image_upload_service.upload_folder(db, files)
    
    if not result['success']:
        raise HTTPException(status_code=400 if 'error' in result else 500, detail=result.get('error'))
    
    return UploadFolderResponse(message=result['message'], **result['data'])


@router.post("/upload-image", response_model=UploadImageResponse)
async def upload_image(file: UploadFile = File(...), image_upload_service: ImageUploadService = Depends(get_image_upload_service)):
    result = await image_upload_service.upload_single_image(file)
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return UploadImageResponse(**result['data'])


@router.get("/list", response_model=ImageListResponse)
async def list_images(db: AsyncSession = Depends(get_db), image_crud: ImageCRUD = Depends(get_image_crud)):
    images = await image_crud.get_all(db)
    
    return ImageListResponse(images=[ImageResponse(filename=img.filename, file_path=img.file_path) for img in images], count=len(images))


@router.delete("/clear", response_model=ClearImagesResponse)
async def clear_images(db: AsyncSession = Depends(get_db), image_upload_service: ImageUploadService = Depends(get_image_upload_service)):
    result = await image_upload_service.clear_all_images(db)
    
    return ClearImagesResponse(message=result['message'])