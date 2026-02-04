from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.deps import get_service_container, ServiceContainer
from app.schemas.image import ImageListResponse, ImageResponse, UploadFolderResponse, UploadImageResponse, ClearImagesResponse

router = APIRouter(prefix="/images", tags=["images"])

@router.post("/upload-folder", response_model=UploadFolderResponse)
async def upload_folder(files: list[UploadFile] = File(...), db: AsyncSession = Depends(get_db), services: ServiceContainer = Depends(get_service_container)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    result = await services.image_upload_service.upload_folder(db, files)
    
    if not result['success']:
        raise HTTPException(status_code=400 if 'error' in result else 500, detail=result.get('error'))
    
    return UploadFolderResponse(message=result['message'], **result['data'])


@router.get("/list", response_model=ImageListResponse)
async def list_images(db: AsyncSession = Depends(get_db), services: ServiceContainer = Depends(get_service_container)):
    images = await services.image_crud.get_all(db)
    
    return ImageListResponse(images=[ImageResponse(filename=img.filename, file_path=img.file_path) for img in images], count=len(images))


@router.delete("/clear", response_model=ClearImagesResponse)
async def clear_images(db: AsyncSession = Depends(get_db), services: ServiceContainer = Depends(get_service_container)):
    result = await services.image_upload_service.clear_all_images(db)
    
    return ClearImagesResponse(message=result['message'])


@router.post("/upload", response_model=UploadImageResponse)
async def upload_image(file: UploadFile = File(...), services: ServiceContainer = Depends(get_service_container)):
    result = await services.image_upload_service.upload_single_image(file)
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return UploadImageResponse(**result['data'])