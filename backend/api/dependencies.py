from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


class AuthDependencies:
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
        bearer_scheme = HTTPBearer(auto_error=False)

        def get_token(
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
        ) -> str:
            if not credentials or credentials.scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=401,
                    detail="请先登录",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return credentials.credentials

        def get_current_user(token: str = Depends(get_token)) -> dict:
            user = auth_manager.authenticate(token)
            if not user:
                raise HTTPException(
                    status_code=401,
                    detail="登录已过期，请重新登录",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return user

        def require_admin(user: dict = Depends(get_current_user)) -> dict:
            if not user["is_admin"]:
                raise HTTPException(status_code=403, detail="需要管理员权限")
            return user

        def require_chat_user(user: dict = Depends(get_current_user)) -> dict:
            if user["is_admin"]:
                raise HTTPException(status_code=403, detail="管理员账号仅用于用户管理")
            return user

        self.get_token = get_token
        self.get_current_user = get_current_user
        self.require_admin = require_admin
        self.require_chat_user = require_chat_user
