import os
from typing import List

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse


def create_session_router(context, auth_dependencies) -> APIRouter:
    router = APIRouter(tags=["会话"])
    require_chat_user = auth_dependencies.require_chat_user

    @router.post("/get_session_history")
    async def get_session_history(
        session_id: str = Form(..., description="会话 ID"),
        user: dict = Depends(require_chat_user),
    ):
        if not session_id or not session_id.strip() or session_id == "default":
            return JSONResponse(content={"error": "无效的会话 ID"}, status_code=400)
        if not context.session_manager.user_owns_session(session_id, user["id"]):
            raise HTTPException(status_code=404, detail="会话不存在")

        history = []
        for message in context.session_manager.get_messages(session_id, is_show=True):
            audio_url = None
            if message.get("wav_path") and os.path.exists(message["wav_path"]):
                filename = os.path.basename(message["wav_path"])
                audio_url = f"/api/audio/{session_id}/{filename}"
            history.append({
                "role": message["role"], "content": message["content"],
                "emotion": message["emotion"], "audio_url": audio_url,
                "created_at": message["created_at"],
            })
        return JSONResponse(content={"session_id": session_id, "history": history})

    @router.get("/api/audio/{session_id}/{filename}")
    async def serve_audio(
        session_id: str, filename: str,
        user: dict = Depends(require_chat_user),
    ):
        if ".." in filename or "/" in filename or "\\" in filename or ".." in session_id:
            return JSONResponse(status_code=400, content={"error": "非法的文件路径"})
        if not context.session_manager.user_owns_session(session_id, user["id"]):
            raise HTTPException(status_code=404, detail="音频文件未找到")
        file_path = os.path.join(context.audio_dir, session_id, filename)
        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={"error": "音频文件未找到"})
        return FileResponse(path=file_path, media_type="audio/wav")

    @router.post("/get_all_sessions")
    async def get_all_sessions(user: dict = Depends(require_chat_user)):
        sessions = context.session_manager.get_all_sessions(user["id"])
        return JSONResponse(content={"sessions": sessions})

    @router.post("/delete_sessions")
    async def delete_sessions(
        session_ids: List[str], user: dict = Depends(require_chat_user)
    ):
        context.session_manager.delete_sessions(session_ids, user["id"])
        return JSONResponse(content={"message": "会话已删除"})

    return router
