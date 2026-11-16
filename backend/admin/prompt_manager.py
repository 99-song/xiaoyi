import json
import re
import sqlite3
from typing import Optional


class PromptManager:
    RESERVED_TYPES = {"emotion_detection"}
    TYPE_PATTERN = re.compile(r"^[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9 _-]{0,49}$")

    def __init__(self, db_path: str, prompt_path: str, default_role: str = "old_people", default_system_prompt: str = "emotion_detection"):
        self.db_path = db_path
        self.prompt_path = prompt_path
        self.default_role = default_role
        self.default_system_prompt = default_system_prompt
        self._init_prompt_table()
        self._init_user_role_column()
        if not self.get_prompt(default_role):
            self.default_role = self._first_prompt_type("role_prompts")
        if not self.get_system_prompt(default_system_prompt):
            self.default_system_prompt = self._first_prompt_type("system_prompts")

    def _first_prompt_type(self, table: str) -> str:
        with self._connect() as conn:
            row = conn.execute(f"SELECT type FROM {table} ORDER BY created_at, rowid LIMIT 1").fetchone()
        if not row:
            raise RuntimeError(f"{table} 至少需要一个提示词")
        return row["type"]

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_user_role_column(self):
        with self._connect() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)")}
            if "role_type" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN role_type TEXT")
            if "emotion_prompt_type" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN emotion_prompt_type TEXT")
            conn.execute(
                "UPDATE users SET role_type = ? WHERE role_type IS NULL OR trim(role_type) = ''",
                (self.default_role,),
            )
            conn.execute(
                "UPDATE users SET emotion_prompt_type = ? WHERE emotion_prompt_type IS NULL OR trim(emotion_prompt_type) = ''",
                (self.default_system_prompt,),
            )

    def _read_seed_file(self) -> list[dict]:
        prompts = []
        with open(self.prompt_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item.get("type"), str) or not isinstance(item.get("content"), str):
                    raise ValueError(f"提示词文件第 {line_number} 行格式错误")
                prompts.append({"type": item["type"], "content": item["content"]})
        return prompts

    def _init_prompt_table(self):
        with self._connect() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS role_prompts (
                    type TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
                    updated_at TIMESTAMP DEFAULT (datetime('now', 'localtime'))
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_prompts (
                    type TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
                    updated_at TIMESTAMP DEFAULT (datetime('now', 'localtime'))
                )
            ''')

            # 从旧结构迁移系统提示词，然后从角色表移除。
            role_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(role_prompts)")
            }
            if "assignable" in role_columns:
                conn.execute('''
                    INSERT OR IGNORE INTO system_prompts (type, content, created_at, updated_at)
                    SELECT type, content, created_at, updated_at
                    FROM role_prompts WHERE assignable = 0
                ''')
                conn.execute('''
                    CREATE TABLE role_prompts_new (
                        type TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
                        updated_at TIMESTAMP DEFAULT (datetime('now', 'localtime'))
                    )
                ''')
                conn.execute('''
                    INSERT INTO role_prompts_new (type, content, created_at, updated_at)
                    SELECT type, content, created_at, updated_at
                    FROM role_prompts WHERE assignable = 1
                ''')
                conn.execute("DROP TABLE role_prompts")
                conn.execute("ALTER TABLE role_prompts_new RENAME TO role_prompts")

            total_count = (
                conn.execute("SELECT COUNT(*) FROM role_prompts").fetchone()[0]
                + conn.execute("SELECT COUNT(*) FROM system_prompts").fetchone()[0]
            )
            if total_count == 0:
                for item in self._read_seed_file():
                    table = "system_prompts" if item["type"] in self.RESERVED_TYPES else "role_prompts"
                    conn.execute(
                        f"INSERT INTO {table} (type, content) VALUES (?, ?)",
                        (item["type"], item["content"]),
                    )

    def list_prompts(self) -> list[dict]:
        with self._connect() as conn:
            role_rows = conn.execute('''
                SELECT type, content, created_at, updated_at
                FROM role_prompts
                ORDER BY created_at ASC
            ''').fetchall()
            system_rows = conn.execute('''
                SELECT type, content, created_at, updated_at
                FROM system_prompts
                ORDER BY created_at ASC
            ''').fetchall()
        roles = [{**dict(row), "assignable": True, "category": "role"} for row in role_rows]
        systems = [{**dict(row), "assignable": False, "category": "system"} for row in system_rows]
        return roles + systems

    def get_prompt(self, prompt_type: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM role_prompts WHERE type = ?", (prompt_type,)
            ).fetchone()
            if not row:
                row = conn.execute(
                    "SELECT content FROM system_prompts WHERE type = ?", (prompt_type,)
                ).fetchone()
        return row["content"] if row else None

    def get_system_prompt(self, prompt_type: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM system_prompts WHERE type = ?", (prompt_type,)
            ).fetchone()
        return row["content"] if row else None

    def create_prompt(self, prompt_type: str, content: str, category: str = "role") -> dict:
        prompt_type = prompt_type.strip()
        content = content.strip()
        if not self.TYPE_PATTERN.fullmatch(prompt_type):
            raise ValueError("模板标识需以中文或英文字母开头，可包含中英文、数字、空格、下划线和连字符")
        if not content:
            raise ValueError("提示词不能为空")
        if category not in {"role", "system"}:
            raise ValueError("提示词分类只能是 role 或 system")
        table = "role_prompts" if category == "role" else "system_prompts"
        with self._connect() as conn:
            exists = conn.execute(
                f"SELECT 1 FROM {table} WHERE type = ?", (prompt_type,)
            ).fetchone()
        if exists:
            raise ValueError("提示词标识已存在")
        try:
            with self._connect() as conn:
                conn.execute(
                    f"INSERT INTO {table} (type, content) VALUES (?, ?)",
                    (prompt_type, content),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("提示词标识已存在") from exc
        return {"type": prompt_type, "content": content, "assignable": category == "role", "category": category}

    def update_prompt(self, prompt_type: str, content: str, new_type: Optional[str] = None, category: Optional[str] = None) -> dict:
        content = content.strip()
        new_type = (new_type or prompt_type).strip()
        if not content:
            raise ValueError("提示词不能为空")
        if not self.TYPE_PATTERN.fullmatch(new_type):
            raise ValueError("模板标识需以中文或英文字母开头，可包含中英文、数字、空格、下划线和连字符")
        if category not in {None, "role", "system"}:
            raise ValueError("提示词分类只能是 role 或 system")
        with self._connect() as conn:
            if category is None:
                is_system = conn.execute(
                    "SELECT 1 FROM system_prompts WHERE type = ?", (prompt_type,)
                ).fetchone()
            else:
                is_system = category == "system"
            table = "system_prompts" if is_system else "role_prompts"
            if new_type != prompt_type and conn.execute(
                f"SELECT 1 FROM {table} WHERE type = ?", (new_type,)
            ).fetchone():
                raise ValueError("同一分类中已存在该提示词名称")
            cursor = conn.execute(f'''
                UPDATE {table}
                SET type = ?, content = ?, updated_at = datetime('now', 'localtime')
                WHERE type = ?
            ''', (new_type, content, prompt_type))
            if cursor.rowcount == 0:
                raise KeyError("提示词不存在")
            if new_type != prompt_type:
                user_column = "emotion_prompt_type" if is_system else "role_type"
                conn.execute(
                    f"UPDATE users SET {user_column} = ? WHERE {user_column} = ?",
                    (new_type, prompt_type),
                )
                if not is_system and prompt_type == self.default_role:
                    self.default_role = new_type
                if is_system and prompt_type == self.default_system_prompt:
                    self.default_system_prompt = new_type
            row = conn.execute(
                f"SELECT type, content, created_at, updated_at FROM {table} WHERE type = ?",
                (new_type,),
            ).fetchone()
        result = dict(row)
        result["assignable"] = table == "role_prompts"
        result["category"] = "role" if result["assignable"] else "system"
        return result

    def delete_prompt(self, prompt_type: str, category: Optional[str] = None):
        if category not in {None, "role", "system"}:
            raise ValueError("提示词分类只能是 role 或 system")
        with self._connect() as conn:
            if category is None:
                is_system = conn.execute(
                    "SELECT 1 FROM system_prompts WHERE type = ?", (prompt_type,)
                ).fetchone()
            else:
                is_system = category == "system"
            table = "system_prompts" if is_system else "role_prompts"
            exists = conn.execute(
                f"SELECT 1 FROM {table} WHERE type = ?", (prompt_type,)
            ).fetchone()
            if not exists:
                raise KeyError("提示词不存在")
            if conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] <= 1:
                raise ValueError("每类提示词至少保留一个，当前唯一模板不能删除")
            replacement = conn.execute(
                f"SELECT type FROM {table} WHERE type != ? ORDER BY created_at, rowid LIMIT 1",
                (prompt_type,),
            ).fetchone()["type"]
            user_column = "emotion_prompt_type" if is_system else "role_type"
            conn.execute(
                f"UPDATE users SET {user_column} = ? WHERE {user_column} = ?",
                (replacement, prompt_type),
            )
            cursor = conn.execute(f"DELETE FROM {table} WHERE type = ?", (prompt_type,))
        if not is_system and prompt_type == self.default_role:
            self.default_role = replacement
        if is_system and prompt_type == self.default_system_prompt:
            self.default_system_prompt = replacement

    def get_user_prompt(self, user_id: int) -> tuple[str, str]:
        with self._connect() as conn:
            row = conn.execute("SELECT role_type FROM users WHERE id = ?", (user_id,)).fetchone()
        role_type = row["role_type"] if row and row["role_type"] else self.default_role
        content = self.get_prompt(role_type)
        if content is None:
            role_type = self.default_role
            content = self.get_prompt(role_type) or ""
        return role_type, content

    def get_user_system_prompt(self, user_id: int) -> tuple[str, str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT emotion_prompt_type FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        prompt_type = row["emotion_prompt_type"] if row and row["emotion_prompt_type"] else self.default_system_prompt
        content = self.get_system_prompt(prompt_type)
        if content is None:
            prompt_type = self.default_system_prompt
            content = self.get_system_prompt(prompt_type) or ""
        return prompt_type, content

    def assign_user_role(self, user_id: int, role_type: str) -> bool:
        with self._connect() as conn:
            prompt = conn.execute(
                "SELECT 1 FROM role_prompts WHERE type = ?", (role_type,)
            ).fetchone()
        if not prompt:
            raise ValueError("角色不存在或不可分配")
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET role_type = ? WHERE id = ?", (role_type, user_id)
            )
            return cursor.rowcount > 0

    def assign_user_system_prompt(self, user_id: int, prompt_type: str) -> bool:
        with self._connect() as conn:
            prompt = conn.execute(
                "SELECT 1 FROM system_prompts WHERE type = ?", (prompt_type,)
            ).fetchone()
            if not prompt:
                raise ValueError("情绪判断提示词不存在或不可分配")
            cursor = conn.execute(
                "UPDATE users SET emotion_prompt_type = ? WHERE id = ?", (prompt_type, user_id)
            )
            return cursor.rowcount > 0
