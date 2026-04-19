# db.py
"""
Module làm việc với SQLite:
- Khởi tạo database
- Lưu lịch sử phân loại
- Lấy danh sách lịch sử gần nhất
"""

import sqlite3
import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any

# Đường dẫn tới file DB nằm trong thư mục data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Thư mục hiện tại (src/utils)
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  # Thư mục gốc (root)
DATA_DIR = os.path.join(ROOT_DIR, "data") # Thư mục data
DB_PATH = os.path.join(DATA_DIR, "sentiments.db") # Đường dẫn file DB


def get_connection():
    """
    Tạo connection tới SQLite.
    check_same_thread=False để dùng được trong Streamlit (nhiều thread).
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def with_connection(func):
    """
    Hàm tiện ích để xử lý kết nối SQLite.
    - func: Hàm xử lý logic cần thực thi với kết nối.
    """
    conn = None
    try:
        conn = get_connection()
        if not conn:
            raise sqlite3.Error("Không thể kết nối tới cơ sở dữ liệu.")
        return func(conn)  # Thực thi logic với kết nối
    except sqlite3.Error as e:
        print(f"Đã xảy ra lỗi: {e}")
    finally:
        if conn:
            conn.close()

def init_db():
    """
    Tạo bảng nếu chưa có.
    Bảng: sentiments(id, text, sentiment, timestamp)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    def create_table(conn):
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sentiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
            """
        )
        conn.commit()
        print("Bảng đã được khởi tạo.")

    with_connection(create_table)

def save_result(text: str, sentiment: str):
    """
    Lưu một kết quả vào bảng sentiments.
    """
    
    def insert_data(conn):
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            """
            INSERT INTO sentiments (text, sentiment, timestamp)
            VALUES (?, ?, ?);
            """,
            (text, sentiment, timestamp),
        )
        conn.commit()
        print("Lưu kết quả thành công.")

    with_connection(insert_data)

def get_history(limit: int = 50, sentiment: str | None = None) -> List[Dict[str, Any]]:
    """
    Lấy danh sách lịch sử mới nhất.
    - limit: số bản ghi tối đa, mặc định 50
    - sentiment: lọc theo POSITIVE / NEGATIVE / NEUTRAL, hoặc None nếu lấy tất cả
    """
    def fetch_data(conn):
        cursor = conn.cursor()
        query = """
            SELECT text, sentiment, timestamp
            FROM sentiments
        """
        params = []

        if sentiment:
            query += " WHERE sentiment = ?"
            params.append(sentiment)

        query += " ORDER BY id DESC LIMIT ?;"
        params.append(limit)

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        history = [
            {"text": text, "sentiment": sentiment, "timestamp": timestamp}
            for text, sentiment, timestamp in rows
        ]
        print(f"Lấy được bản ghi lịch sử thành công: {len(history)} bản ghi.")
        return history

    return with_connection(fetch_data) or []