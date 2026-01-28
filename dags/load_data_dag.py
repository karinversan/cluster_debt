from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.ml.load_features import load_csv_to_postgres

with DAG(
    dag_id="load_data_to_postgres",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    load_task = PythonOperator(
        task_id="load_csv",
        python_callable=load_csv_to_postgres
    )
