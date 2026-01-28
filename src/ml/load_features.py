import os
import pandas as pd
from sqlalchemy import create_engine, text

DEFAULT_DB_URI = "postgresql+psycopg2://mlops:mlops@postgres:5432/segmentation"


def get_db_uri() -> str:
    return os.environ.get("DB_URI", DEFAULT_DB_URI)


RENAME = {
    "CUST_ID": "customer_id",
    "BALANCE": "balance",
    "BALANCE_FREQUENCY": "balance_frequency",
    "PURCHASES": "purchases",
    "ONEOFF_PURCHASES": "oneoff_purchases",
    "INSTALLMENTS_PURCHASES": "installments_purchases",
    "CASH_ADVANCE": "cash_advance",
    "PURCHASES_FREQUENCY": "purchases_frequency",
    "ONEOFF_PURCHASES_FREQUENCY": "oneoff_purchases_frequency",
    "PURCHASES_INSTALLMENTS_FREQUENCY": "purchases_installments_frequency",
    "CASH_ADVANCE_FREQUENCY": "cash_advance_frequency",
    "CASH_ADVANCE_TRX": "cash_advance_trx",
    "PURCHASES_TRX": "purchases_trx",
    "CREDIT_LIMIT": "credit_limit",
    "PAYMENTS": "payments",
    "MINIMUM_PAYMENTS": "minimum_payments",
    "PRC_FULL_PAYMENT": "prc_full_payment",
    "TENURE": "tenure",
}

def load_csv_to_postgres(csv_path: str = "/opt/airflow/data/Customer Data.csv"):
    df = pd.read_csv(csv_path)

    df = df.rename(columns=RENAME)


    df["event_timestamp"] = pd.Timestamp.utcnow()


    cols = ["customer_id", "event_timestamp"] + [v for v in RENAME.values() if v != "customer_id"]
    df = df[cols]

    engine = create_engine(get_db_uri())

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE customer_features;"))
    df.to_sql("customer_features", engine, if_exists="append", index=False, chunksize=2000)
