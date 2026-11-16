import os
import shutil
import sqlite3
from typing import List, Dict, Any, Union

class SessionManager:
    def __init__(self, db_path: str, base_wav_dir: str, max_history: int = 10):
        self.db_path = db_path
        self.base_wav_dir = base_wav_dir
        self.max_history = max_history
        self._init_db()

    def _init_db(self):
            """初始化数据库和表结构"""
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # 创建消息表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        emotion TEXT,
                        wav_path TEXT,
                        created_at TIMESTAMP DEFAULT (datetime('now', 'localtime'))
                    )
                ''')
                # 为 session_id 创建索引，提升查询和截断速度
                cursor.execute('''CREATE INDEX IF NOT EXISTS idx_session_id ON messages(session_id)''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS session_owners (
                        session_id TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT (datetime('now', 'localtime'))
                    )
                ''')
                cursor.execute('''CREATE INDEX IF NOT EXISTS idx_session_owner_user_id ON session_owners(user_id)''')
                conn.commit()

    def create_session(self, session_id: str, user_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO session_owners (session_id, user_id) VALUES (?, ?)",
                (session_id, user_id),
            )

    def user_owns_session(self, session_id: str, user_id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM session_owners WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
        return row is not None

    def _add_message_user(self, session_id: str, role: str, content: str, emotion: str, wav_path: str):
        """通用插入消息方法"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO messages (session_id, role, content, emotion, wav_path) VALUES (?, ?, ?, ?, ?)''', (session_id, role, content, emotion, wav_path))
            conn.commit()
            return cursor.lastrowid

    def _add_message_assistant(self, session_id: str, role: str, content: str, wav_path: str):
        """通用插入消息方法"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO messages (session_id, role, content, wav_path) VALUES (?, ?, ?, ?)''', (session_id, role, content, wav_path))
            conn.commit()
            return cursor.lastrowid

    def add_user_message(self, session_id: str, text: str, emotion: str, user_wav_path: str):
        msg_id = self._add_message_user(session_id, "user", text, emotion, user_wav_path)
        return msg_id

    def add_assistant_message(self, session_id: str, text: str, assistant_wav_path: str):
        msg_id = self._add_message_assistant(session_id, "assistant", text, assistant_wav_path)
        return msg_id

    def update_session_emotion(self, id: str, emotion: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE messages SET emotion = ? WHERE id = ?",(emotion, id))
            conn.commit()

    def rollback_last_user_message(self, session_id: str):
        """如果 API 调用失败，回滚最后一次用户提问"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 获取该 session 的最后一条记录
            cursor.execute('''SELECT id, role FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1''', (session_id,))
            row = cursor.fetchone()
            
            # 如果存在且最后一条是用户发出的，则删除
            if row and row[1] == "user":
                cursor.execute('DELETE FROM messages WHERE id = ?', (row[0],))
                conn.commit()

    def get_messages(self, session_id: str, is_show=False) -> List[Dict[str, str]]:
        """获取完整对话记录用于 API 请求"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if is_show:
                cursor.execute('''SELECT role, content, emotion, wav_path, created_at FROM messages WHERE session_id = ? ORDER BY id ASC ''', (session_id,))
            else:
                cursor.execute('''SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC  LIMIT ?''', (session_id, self.max_history))
            rows = cursor.fetchall()
            
        messages = []
        if is_show:
            for role, content, emotion, wav_path, created_at in rows:
                messages.append({"role": role, "content": content, "emotion": emotion, "wav_path": wav_path, "created_at": created_at})
        else:
            for role, content in rows:
                messages.append({"role": role, "content": content})
            
        return messages
    
    def get_all_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # 通过 JOIN 获取每个 session 的最后一条用户消息
            cursor.execute('''
                SELECT m.session_id, m.content, m.created_at
                FROM messages m
                JOIN session_owners so ON so.session_id = m.session_id AND so.user_id = ?
                INNER JOIN (
                    SELECT session_id, MAX(created_at) AS max_time
                    FROM messages
                    WHERE role = 'user'
                    GROUP BY session_id
                ) t ON m.session_id = t.session_id 
                AND m.created_at = t.max_time 
                AND m.role = 'user'
                ORDER BY m.created_at DESC
            ''', (user_id,))
            rows = cursor.fetchall()
        return [
            {"session_id": row[0], "last_user_message": row[1], "timestamp": row[2]}
            for row in rows
        ]
        
    def delete_sessions(self, session_ids: Union[str, List[str]], user_id: int):
        if isinstance(session_ids, str):
            session_ids = [session_ids]

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for sid in session_ids:
                owner = cursor.execute(
                    "SELECT 1 FROM session_owners WHERE session_id = ? AND user_id = ?",
                    (sid, user_id),
                ).fetchone()
                if not owner:
                    continue
                # 1. 删除数据库记录
                cursor.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                cursor.execute("DELETE FROM session_owners WHERE session_id = ?", (sid,))
                # 2. 删除音频文件夹
                session_dir = os.path.join(self.base_wav_dir, sid)
                if os.path.isdir(session_dir):
                    try:
                        shutil.rmtree(session_dir, ignore_errors=False)  # 捕获错误
                    except OSError as e:
                        # 可根据项目需要记录日志或继续
                        print(f"删除文件夹 {session_dir} 失败: {e}")
            conn.commit()

    def delete_user_sessions(self, user_id: int):
        with sqlite3.connect(self.db_path) as conn:
            session_ids = [
                row[0] for row in conn.execute(
                    "SELECT session_id FROM session_owners WHERE user_id = ?", (user_id,)
                ).fetchall()
            ]
        self.delete_sessions(session_ids, user_id)
