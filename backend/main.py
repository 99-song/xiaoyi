import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from admin import AuthManager, create_admin_router
from admin.prompt_manager import PromptManager
from admin.voice_prompt_manager import VoicePromptManager
from api import ApiContext, AuthDependencies, create_api_router
from asr.asr import ASRService
from emotion.emotion import EmotionService
from llm.llm import LLMService
from llm.session import SessionManager
from tools.common import AppConfig, init_logger
from tts.tts import TTSService


AUDIO_DIR = "./db_data/wavs"
DB_PATH = "./db_data/sessions.db"


def create_app() -> FastAPI:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    init_logger()

    config = AppConfig()
    session_manager = SessionManager(DB_PATH, AUDIO_DIR)
    auth_manager = AuthManager(DB_PATH)
    system_admin = auth_manager.ensure_system_admin()
    if system_admin["created"]:
        logger.warning(
            "[AUTH] 已初始化系统管理员，请登录后立即修改账号信息 | "
            f"username={system_admin['username']} | password={system_admin['initial_password']}"
        )

    prompt_manager = PromptManager(DB_PATH, "./llm/llm_prompt.jsonl")
    voice_prompt_manager = VoicePromptManager(DB_PATH, "./db_data/voice_prompts")
    asr_service = ASRService(config=config)
    tts_service = TTSService(config=config)
    emotion_service = EmotionService(config=config)
    llm_service = LLMService(
        session_manager, config=config, prompt_manager=prompt_manager
    )

    context = ApiContext(
        audio_dir=AUDIO_DIR,
        auth_manager=auth_manager,
        session_manager=session_manager,
        voice_prompt_manager=voice_prompt_manager,
        asr_service=asr_service,
        tts_service=tts_service,
        emotion_service=emotion_service,
        llm_service=llm_service,
    )
    auth_dependencies = AuthDependencies(auth_manager)

    app = FastAPI(title="AI Assistant", description="AI 心灵引航员")
    app.state.config = config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(create_api_router(context, auth_dependencies))
    app.include_router(
        create_admin_router(
            auth_manager,
            session_manager,
            prompt_manager,
            voice_prompt_manager,
            asr_service,
            config,
            auth_dependencies.require_admin,
        )
    )
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=app.state.config.backend_port)
