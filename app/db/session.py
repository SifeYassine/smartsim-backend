import sqlite_vec
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import event
from typing import AsyncGenerator
from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool, echo=False)

@event.listens_for(engine.sync_engine, "connect")
def load_sqlite_vec(dbapi_connection, connection_record):
    raw_connection = dbapi_connection._connection._conn
    raw_connection.enable_load_extension(True)
    sqlite_vec.load(raw_connection)
    raw_connection.enable_load_extension(False)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()