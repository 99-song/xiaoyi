"""管理员后台功能。"""

from .auth import AuthManager
from .router import create_admin_router

__all__ = ["AuthManager", "create_admin_router"]
