import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_name="parking.db"):
        self.db_path = os.path.join(os.path.dirname(__file__), db_name)
        self._create_table()

    def _create_table(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_text TEXT,
                    confidence REAL,
                    timestamp TEXT
                )
            """)
            conn.commit()

    def log_detection(self, plate_text, confidence):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detections (plate_text, confidence, timestamp)
                VALUES (?, ?, ?)
            """, (plate_text, confidence, datetime.now().isoformat()))
            conn.commit()

    def get_first_detection_today(self, plate_text):
        # Get start of today in ISO format string comparison
        start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp FROM detections
                WHERE plate_text = ? AND timestamp >= ?
                ORDER BY timestamp ASC
                LIMIT 1
            """, (plate_text, start_of_day))
            row = cursor.fetchone()
            
        if row:
            return datetime.fromisoformat(row[0])
        return None
