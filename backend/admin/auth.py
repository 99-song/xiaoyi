import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Optional


class AuthManager:
    MIN_USERNAME_LENGTH = 3
    MIN_PASSWORD_LENGTH = 6

    def __init__(self, db_path: str, iterations: int = 210_000, token_expire_minutes: int = 5):
        self.db_path = db_path
        self.iterations = iterations
        self.token_expire_minutes = token_expire_minutes
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT (datetime('now', 'localtime'))
                );
                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_id ON auth_tokens(user_id);
            ''')
            user_columns = {
                row['name'] for row in conn.execute('PRAGMA table_info(users)').fetchall()
            }
            if 'last_login_at' not in user_columns:
                conn.execute('ALTER TABLE users ADD COLUMN last_login_at TIMESTAMP')
            if 'is_admin' not in user_columns:
                conn.execute('ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0')
            if 'must_change_password' not in user_columns:
                conn.execute('ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0')
            if 'role_type' not in user_columns:
                conn.execute('ALTER TABLE users ADD COLUMN role_type TEXT')
            if 'emotion_prompt_type' not in user_columns:
                conn.execute('ALTER TABLE users ADD COLUMN emotion_prompt_type TEXT')
            if 'voice_prompt_id' not in user_columns:
                conn.execute('ALTER TABLE users ADD COLUMN voice_prompt_id INTEGER')
            if not conn.execute('SELECT 1 FROM users WHERE is_admin = 1 LIMIT 1').fetchone():
                first_user = conn.execute('SELECT id FROM users ORDER BY id LIMIT 1').fetchone()
                if first_user:
                    conn.execute('UPDATE users SET is_admin = 1 WHERE id = ?', (first_user['id'],))

    def _hash_password(self, password: str, salt: bytes) -> str:
        value = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, self.iterations)
        return value.hex()

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode('utf-8')).hexdigest()

    def register(self, username: str, password: str) -> dict:
        username = username.strip()
        salt = os.urandom(16)
        password_hash = self._hash_password(password, salt)
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    'INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)',
                    (username, password_hash, salt.hex()),
                )
                return {'id': cursor.lastrowid, 'username': username}
        except sqlite3.IntegrityError as exc:
            raise ValueError('用户名已存在') from exc

    def ensure_system_admin(self) -> dict:
        """确保系统有且仅需数据库标记的管理员；全新数据库生成一次性初始密码。"""
        with self._connect() as conn:
            row = conn.execute('SELECT id, username FROM users WHERE is_admin = 1 ORDER BY id LIMIT 1').fetchone()
            if row:
                return {'id': row['id'], 'username': row['username'], 'created': False}
            username = 'admin'
            password = secrets.token_urlsafe(12)
            salt = os.urandom(16)
            cursor = conn.execute(
                '''INSERT INTO users (username, password_hash, salt, is_admin, must_change_password)
                   VALUES (?, ?, ?, 1, 1)''',
                (username, self._hash_password(password, salt), salt.hex()),
            )
            return {
                'id': cursor.lastrowid, 'username': username,
                'initial_password': password, 'created': True,
            }

    def list_users(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute('''
                SELECT id, username, created_at, last_login_at, role_type, emotion_prompt_type, voice_prompt_id
                FROM users
                WHERE is_admin = 0
                ORDER BY created_at DESC, id DESC
            ''').fetchall()
        return [dict(row) for row in rows]

    def get_user(self, user_id: int) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT id, username, created_at, last_login_at, role_type, emotion_prompt_type, voice_prompt_id, is_admin, must_change_password FROM users WHERE id = ?', (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_user(self, user_id: int, username: Optional[str] = None, password: Optional[str] = None) -> Optional[dict]:
        with self._connect() as conn:
            user = conn.execute(
                'SELECT id, username, created_at, last_login_at, role_type, emotion_prompt_type, voice_prompt_id, is_admin, must_change_password FROM users WHERE id = ?', (user_id,)
            ).fetchone()
            if not user:
                return None
            try:
                if username is not None:
                    conn.execute('UPDATE users SET username = ? WHERE id = ?', (username.strip(), user_id))
                if password is not None:
                    salt = os.urandom(16)
                    conn.execute(
                        'UPDATE users SET password_hash = ?, salt = ? WHERE id = ?',
                        (self._hash_password(password, salt), salt.hex(), user_id),
                    )
                    conn.execute('DELETE FROM auth_tokens WHERE user_id = ?', (user_id,))
                    conn.execute('UPDATE users SET must_change_password = 0 WHERE id = ?', (user_id,))
            except sqlite3.IntegrityError as exc:
                raise ValueError('用户名已存在') from exc
            updated = conn.execute(
                'SELECT id, username, created_at, last_login_at, role_type, emotion_prompt_type, voice_prompt_id, is_admin, must_change_password FROM users WHERE id = ?', (user_id,)
            ).fetchone()
            return dict(updated)

    def delete_user(self, user_id: int) -> bool:
        with self._connect() as conn:
            conn.execute('DELETE FROM auth_tokens WHERE user_id = ?', (user_id,))
            cursor = conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
            return cursor.rowcount > 0

    def login(self, username: str, password: str) -> Optional[dict]:
        with self._connect() as conn:
            user = conn.execute(
                'SELECT id, username, password_hash, salt, role_type, is_admin, must_change_password FROM users WHERE username = ?',
                (username.strip(),),
            ).fetchone()
            if not user:
                return None
            candidate = self._hash_password(password, bytes.fromhex(user['salt']))
            if not hmac.compare_digest(candidate, user['password_hash']):
                return None

            conn.execute(
                "UPDATE users SET last_login_at = datetime('now', 'localtime') WHERE id = ?",
                (user['id'],),
            )
            token = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
            conn.execute('DELETE FROM auth_tokens WHERE expires_at <= ?', (datetime.utcnow().isoformat(),))
            conn.execute(
                'INSERT INTO auth_tokens (token_hash, user_id, expires_at) VALUES (?, ?, ?)',
                (self._hash_token(token), user['id'], expires_at.isoformat()),
            )
            return {
                'access_token': token,
                'token_type': 'bearer',
                'expires_in': self.token_expire_minutes * 60,
                'user': {
                    'id': user['id'], 'username': user['username'], 'role_type': user['role_type'],
                    'is_admin': bool(user['is_admin']),
                    'must_change_password': bool(user['must_change_password']),
                },
            }

    def authenticate(self, token: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute('''
                SELECT u.id, u.username, u.role_type, u.is_admin, u.must_change_password, t.expires_at
                FROM auth_tokens t
                JOIN users u ON u.id = t.user_id
                WHERE t.token_hash = ?
            ''', (self._hash_token(token),)).fetchone()
            if not row or datetime.fromisoformat(row['expires_at']) <= datetime.utcnow():
                return None
            # 滑动有效期：用户持续操作并产生请求时，从当前时间重新计算过期时间。
            new_expires_at = datetime.utcnow() + timedelta(minutes=self.token_expire_minutes)
            conn.execute(
                'UPDATE auth_tokens SET expires_at = ? WHERE token_hash = ?',
                (new_expires_at.isoformat(), self._hash_token(token)),
            )
            return {
                'id': row['id'], 'username': row['username'], 'role_type': row['role_type'],
                'is_admin': bool(row['is_admin']),
                'must_change_password': bool(row['must_change_password']),
            }

    def logout(self, token: str):
        with self._connect() as conn:
            conn.execute('DELETE FROM auth_tokens WHERE token_hash = ?', (self._hash_token(token),))
