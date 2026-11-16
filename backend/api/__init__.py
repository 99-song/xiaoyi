"""业务 API 路由和语音对话处理模块。"""

from fastapi import APIRouter

from .auth import create_auth_router
from .chat import create_chat_router
from .context import ApiContext
from .dependencies import AuthDependencies
from .session import create_session_router


def create_api_router(context: ApiContext, auth_dependencies: AuthDependencies) -> APIRouter:
    router = APIRouter()
    router.include_router(create_auth_router(context.auth_manager, auth_dependencies))
    router.include_router(create_chat_router(context, auth_dependencies))
    router.include_router(create_session_router(context, auth_dependencies))
    return router


__all__ = ["ApiContext", "AuthDependencies", "create_api_router"]
