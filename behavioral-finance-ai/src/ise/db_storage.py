
#approach 1: In-Memory / Dynamic Execution
import psycopg2
from datetime import datetime

DB_CONFIG = {
    "dbname": "behavioral_finance_db",
    "user": "postgres",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}

def init_db():
    query = """
    CREATE TABLE IF NOT EXISTS micro_savings_logs (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(50),
        log_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        survival_buffer NUMERIC(10, 2),
        surplus NUMERIC(10, 2),
        transfer_amount NUMERIC(10, 2)
    );
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    cur.close()
    conn.close()

def log_savings_transaction(user_id, survival_buffer, surplus, transfer_amount):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO micro_savings_logs (user_id, survival_buffer, surplus, transfer_amount)
        VALUES (%s, %s, %s, %s)
        """,
        (user_id, survival_buffer, surplus, transfer_amount)
    )
    conn.commit()
    cur.close()
    conn.close()