import os
from pathlib import Path
from typing import Optional
from sqlalchemy import create_engine

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"
DEFAULT_DB_URI = "postgresql+psycopg2://mlops:mlops@postgres:5432/segmentation"


def get_db_uri() -> str:
    return os.environ.get("DB_URI", DEFAULT_DB_URI)


def get_engine(db_uri: Optional[str] = None):
    return create_engine(db_uri or get_db_uri())


def init_db(schema_path: Optional[Path] = None) -> None:
    path = schema_path or SCHEMA_PATH
    sql = path.read_text()
    statements = [s.strip() for s in sql.split(";") if s.strip()]

    engine = get_engine()
    with engine.begin() as conn:
        for stmt in statements:
            conn.exec_driver_sql(stmt)
