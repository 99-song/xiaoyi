import os
import urllib.parse
import uuid
from typing import Optional

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from loguru import logger

from .asr import transcribe_user_audio
from .emotion import analyze_user_emotion, extract_assistant_emotion
from .llm import generate_reply, generate_reply_emotion
from .tts import synthesize_reply


def create_chat_router(context, auth_dependencies) -> APIRouter:
    router = APIRouter(tags=["对话"])

    @router.post("/chat")
    async def chat(
        audio_file: UploadFile = File(..., description="语音输入"),
        session_id: Optional[str] = Form(None, description="会话 ID"),
        user: dict = Depends(auth_dependencies.require_chat_user),
    ):
        logger.info(f"[USER] 聊天请求 | username={user['username']} | user_id={user['id']}")
        if not session_id or not session_id.strip() or session_id == "default":
            session_id = uuid.uuid4().hex
            context.session_manager.create_session(session_id, user["id"])
            logger.info(f"New Session Id: {session_id}")
        elif not context.session_manager.user_owns_session(session_id, user["id"]):
            raise HTTPException(status_code=404, detail="会话不存在")
        else:
            logger.info(f"History Session Id: {session_id}")

        message_id = uuid.uuid4().hex
        session_dir = os.path.join(context.audio_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        user_wav_path = os.path.join(session_dir, f"user_{message_id}.wav")
        assistant_wav_path = os.path.join(session_dir, f"assistant_{message_id}.wav")

        user_audio = await audio_file.read()
        async with aiofiles.open(user_wav_path, "wb") as file:
            await file.write(user_audio)

        user_text = await transcribe_user_audio(context.asr_service, user_audio, session_id)
        if not user_text:
            logger.info(f"[{session_id}] No valid text recognized")
            return JSONResponse(
                content={"error": "无法识别语音内容，请重试。", "session_id": session_id},
                status_code=400,
            )
        user_emotion = await analyze_user_emotion(
            context.emotion_service, user_wav_path, session_id
        )
        reply_result, reply_text = await generate_reply(
            context.llm_service, user_text, user_emotion, session_id,
            user_wav_path, assistant_wav_path, user["id"],
        )
        instruct_text = await generate_reply_emotion(
            context.llm_service, user_text, user_emotion, user["id"], session_id
        )
        tts_audio = await synthesize_reply(
            context.tts_service, context.voice_prompt_manager, reply_text,
            instruct_text, user["id"], session_id,
        )
        async with aiofiles.open(assistant_wav_path, "wb") as file:
            await file.write(tts_audio)

        assistant_emotion = extract_assistant_emotion(instruct_text, session_id)
        if reply_result.message_id is not None:
            context.session_manager.update_session_emotion(
                reply_result.message_id, assistant_emotion
            )

        encoded_emotion = urllib.parse.quote(assistant_emotion) if assistant_emotion else ""
        return Response(
            content=tts_audio, media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=response_{session_id}.wav",
                "X-Session-ID": session_id,
                "X-Assistant-Emotion": encoded_emotion,
                "Access-Control-Expose-Headers": "X-Session-ID, X-Assistant-Emotion",
            },
        )

    return router
