from pydantic import BaseModel

class ImageResponse(BaseModel):
    filename: str
    file_path: str
    
    class Config:
        from_attributes = True


class ImageListResponse(BaseModel):
    images: list[ImageResponse]
    count: int


class UploadFolderResponse(BaseModel):
    message: str
    total_uploaded: int
    total_valid: int
    invalid_files: list[str]
    failed_saves: list[str]
    new_images: int
    already_indexed: int


class UploadImageResponse(BaseModel):
    message: str
    filename: str
    file_path: str
    already_existed: bool


class ClearImagesResponse(BaseModel):
    message: str