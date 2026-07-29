import sqlite3
from pathlib import Path


def init_db():
    base_dir = Path(__file__).resolve().parent.parent
    db_dir = base_dir / 'database'
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_dir / 'garbage.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            prediction TEXT,
            confidence REAL,
            latitude REAL,
            longitude REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()
