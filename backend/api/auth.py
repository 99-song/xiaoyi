from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field


class AuthRequest(BaseModel):
    username: str = Field(..., max_length=50)
    password: str = Field(..., max_length=128)


def create_auth_router(auth_manager, auth_dependencies) -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["认证"])

    @router.post("/login")
    async def login(payload: AuthRequest, request: Request):
        result = auth_manager.login(payload.username, payload.password)
        if not result:
            logger.warning(
                f"[AUTH] 登录失败 | username={payload.username.strip()} | "
                f"ip={request.client.host if request.client else 'unknown'} | reason=用户名或密码错误"
            )
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        logger.info(
            f"[AUTH] 登录成功 | username={result['user']['username']} | "
            f"user_id={result['user']['id']} | "
            f"role={'admin' if result['user']['is_admin'] else 'user'} | "
            f"ip={request.client.host if request.client else 'unknown'}"
        )
        return result

    @router.post("/logout")
    async def logout(
        request: Request,
        token: str = Depends(auth_dependencies.get_token),
        user: dict = Depends(auth_dependencies.get_current_user),
    ):
        auth_manager.logout(token)
        logger.info(
            f"[AUTH] 退出登录 | username={user['username']} | user_id={user['id']} | "
            f"ip={request.client.host if request.client else 'unknown'}"
        )
        return {"message": "已退出登录"}

    @router.get("/me")
    async def auth_me(user: dict = Depends(auth_dependencies.get_current_user)):
        return {"user": user, "expires_in": auth_manager.token_expire_minutes * 60}

    return router
