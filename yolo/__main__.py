import uvicorn

from .config import get_config

if __name__ == "__main__":
    settings = get_config()
    uvicorn.run(
        "yolo.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
