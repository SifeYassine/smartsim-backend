from sqlalchemy.orm import declarative_base
from app.db.session import engine

Base = declarative_base()

async def init_db():    
    from app.models.image import Image
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)