import os
import shutil
import sqlite3
import uuid
from typing import Optional


class VoicePromptManager:
    def __init__(self, db_path: str, storage_dir: str):
        self.db_path = db_path
        self.storage_dir = storage_dir
        self.default_id = None
        os.makedirs(storage_dir, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS voice_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    wav_path TEXT NOT NULL,
                    language TEXT NOT NULL DEFAULT 'Chinese',
                    created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
                    updated_at TIMESTAMP DEFAULT (datetime('now', 'localtime'))
                )
            ''')
            row = conn.execute(
                "SELECT id FROM voice_prompts ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            self.default_id = row["id"] if row else None

            columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
            if "voice_prompt_id" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN voice_prompt_id INTEGER")
            if self.default_id is not None:
                conn.execute(
                    "UPDATE users SET voice_prompt_id = ? WHERE voice_prompt_id IS NULL",
                    (self.default_id,),
                )

    def list_prompts(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute('''
                SELECT id, name, text, language, created_at, updated_at
                FROM voice_prompts ORDER BY id
            ''').fetchall()
        return [{**dict(row), "is_default": row["id"] == self.default_id} for row in rows]

    def get_prompt(self, prompt_id: int) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM voice_prompts WHERE id = ?", (prompt_id,)).fetchone()
        return dict(row) if row else None

    def get_user_prompt(self, user_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT voice_prompt_id FROM users WHERE id = ?", (user_id,)).fetchone()
        prompt_id = row["voice_prompt_id"] if row and row["voice_prompt_id"] else self.default_id
        prompt = self.get_prompt(prompt_id) if prompt_id is not None else None
        if not prompt and self.default_id is not None:
            prompt = self.get_prompt(self.default_id)
        if not prompt:
            raise RuntimeError("没有可用的音色模板")
        return prompt

    def _store_audio(self, source_path: str) -> str:
        target = os.path.join(self.storage_dir, f"{uuid.uuid4().hex}.wav")
        shutil.move(source_path, target)
        os.chmod(target, 0o644)
        return target

    def create_prompt(self, name: str, text: str, audio_path: str, language: str = "Chinese") -> dict:
        if not name.strip() or not text.strip():
            raise ValueError("音色名称和参考文本不能为空")
        stored_path = self._store_audio(audio_path)
        with self._connect() as conn:
            cursor = conn.execute('''
                INSERT INTO voice_prompts (name, text, wav_path, language)
                VALUES (?, ?, ?, ?)
            ''', (name.strip(), text.strip(), stored_path, language.strip() or "Chinese"))
            prompt_id = cursor.lastrowid
            if self.default_id is None:
                self.default_id = prompt_id
                conn.execute(
                    "UPDATE users SET voice_prompt_id = ? WHERE voice_prompt_id IS NULL",
                    (prompt_id,),
                )
        return self.get_prompt(prompt_id)

    def update_prompt(self, prompt_id: int, name: str, text: str, audio_path: Optional[str] = None) -> dict:
        current = self.get_prompt(prompt_id)
        if not current:
            raise KeyError("音色模板不存在")
        if not name.strip() or not text.strip():
            raise ValueError("音色名称和参考文本不能为空")
        new_path = self._store_audio(audio_path) if audio_path else current["wav_path"]
        with self._connect() as conn:
            conn.execute('''
                UPDATE voice_prompts SET name = ?, text = ?, wav_path = ?, updated_at = datetime('now', 'localtime')
                WHERE id = ?
            ''', (name.strip(), text.strip(), new_path, prompt_id))
        if audio_path and current["wav_path"].startswith(os.path.abspath(self.storage_dir)):
            try:
                os.remove(current["wav_path"])
            except FileNotFoundError:
                pass
        return self.get_prompt(prompt_id)

    def delete_prompt(self, prompt_id: int):
        prompt = self.get_prompt(prompt_id)
        if not prompt:
            raise KeyError("音色模板不存在")
        with self._connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM voice_prompts").fetchone()[0] <= 1:
                raise ValueError("参考音色至少保留一个，当前唯一音色不能删除")
            replacement_id = conn.execute(
                "SELECT id FROM voice_prompts WHERE id != ? ORDER BY created_at, id LIMIT 1", (prompt_id,)
            ).fetchone()["id"]
            conn.execute("UPDATE users SET voice_prompt_id = ? WHERE voice_prompt_id = ?", (replacement_id, prompt_id))
            conn.execute("DELETE FROM voice_prompts WHERE id = ?", (prompt_id,))
        if prompt_id == self.default_id:
            self.default_id = replacement_id
        if prompt["wav_path"].startswith(os.path.abspath(self.storage_dir)):
            try:
                os.remove(prompt["wav_path"])
            except FileNotFoundError:
                pass

    def assign_user_prompt(self, user_id: int, prompt_id: int) -> bool:
        if not self.get_prompt(prompt_id):
            raise ValueError("音色模板不存在")
        with self._connect() as conn:
            cursor = conn.execute("UPDATE users SET voice_prompt_id = ? WHERE id = ?", (prompt_id, user_id))
        return cursor.rowcount > 0
