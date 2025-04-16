import sqlite3

class Database:
    def __init__(self, db_name='detections.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def insert_detection(self, label, confidence, x, y, width, height):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO detections (label, confidence, x, y, width, height)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (label, confidence, x, y, width, height))
        self.conn.commit()

    def close(self):
        self.conn.close()