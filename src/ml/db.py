from sqlalchemy import create_engine

DB_URI = "postgresql+psycopg2://mlops:mlops@postgres:5432/segmentation"

def get_engine():
    return create_engine(DB_URI)
