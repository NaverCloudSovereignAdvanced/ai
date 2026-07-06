from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    database_url: str = Field(default="sqlite:///data/app.db", alias="DATABASE_URL")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="GEMINI_MODEL")
    naver_clova_ocr_invoke_url: str | None = Field(default=None, alias="NAVER_CLOVA_OCR_INVOKE_URL")
    naver_clova_ocr_secret_key: str | None = Field(default=None, alias="NAVER_CLOVA_OCR_SECRET_KEY")

    class Config:
        populate_by_name = True
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "sources").mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "rag").mkdir(parents=True, exist_ok=True)
    return settings
