"""
database.py
-----------
Handles all SQLite database operations for the AI-Based Job Recommendation System.

Why SQLite?
- Serverless & Zero-Configuration: SQLite is embedded directly into the application,
  storing the database as a single file (database/jobs.db).
- Built-in to Python: Requires no external database servers (like PostgreSQL or MySQL)
  to be installed or configured, making it ideal for student portfolios and viva demos.
- High Performance & Reliability: Easily handles thousands of records with ACID compliance.
"""

import os
import sqlite3
import pandas as pd

# Define paths relative to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "database", "jobs.db")
DEFAULT_CSV_PATH = os.path.join(BASE_DIR, "data", "jobs.csv")


def get_db_connection(db_path=DEFAULT_DB_PATH):
    """
    Establishes and returns a connection to the SQLite database.
    Sets row_factory to sqlite3.Row to access columns by name.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(csv_path=DEFAULT_CSV_PATH, db_path=DEFAULT_DB_PATH):
    """
    Initializes the SQLite database:
    1. Ensures the database directory exists.
    2. Creates the 'jobs' table if it does not already exist.
    3. Reads 'jobs.csv' and populates the database table.
    """
    # Ensure database folder exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Create the jobs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY,
            job_title TEXT NOT NULL,
            company TEXT NOT NULL,
            location TEXT NOT NULL,
            required_skills TEXT NOT NULL,
            qualification TEXT NOT NULL,
            experience INTEGER NOT NULL,
            job_type TEXT NOT NULL,
            description TEXT NOT NULL
        )
    """)
    conn.commit()

    # Check if table already has records
    cursor.execute("SELECT COUNT(*) AS total FROM jobs")
    count = cursor.fetchone()["total"]

    if count == 0:
        if not os.path.exists(csv_path):
            conn.close()
            raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

        print(f"Loading jobs from '{csv_path}' into SQLite database...")
        df = pd.read_csv(csv_path)

        # Insert records into SQLite
        insert_query = """
            INSERT INTO jobs (
                job_id, job_title, company, location,
                required_skills, qualification, experience,
                job_type, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        records = [
            (
                int(row["job_id"]),
                str(row["job_title"]).strip(),
                str(row["company"]).strip(),
                str(row["location"]).strip(),
                str(row["required_skills"]).strip(),
                str(row["qualification"]).strip(),
                int(row["experience"]),
                str(row["job_type"]).strip(),
                str(row["description"]).strip(),
            )
            for _, row in df.iterrows()
        ]

        cursor.executemany(insert_query, records)
        conn.commit()
        print(f"Successfully inserted {len(records)} job records into SQLite database.")
    else:
        print(f"Database already initialized with {count} job records.")

    conn.close()


def get_all_jobs(db_path=DEFAULT_DB_PATH):
    """
    Retrieves all job records from SQLite database as a Pandas DataFrame.
    If the database is empty or missing, it initializes it from CSV first.
    """
    if not os.path.exists(db_path):
        init_db(db_path=db_path)

    conn = get_db_connection(db_path)
    query = "SELECT * FROM jobs ORDER BY job_id ASC"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # If table is empty, re-initialize
    if df.empty:
        init_db(db_path=db_path)
        conn = get_db_connection(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()

    return df


if __name__ == "__main__":
    print("Initializing SQLite Database...")
    init_db()
    print("Database initialization complete.")
