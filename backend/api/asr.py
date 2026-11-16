from loguru import logger


async def transcribe_user_audio(asr_service, audio_bytes: bytes, session_id: str) -> str:
    logger.info(f"[{session_id}] ASR Processing...")
    user_text = (await asr_service.transcribe_audio(audio_bytes)).strip()
    logger.info(f"[{session_id}] User Input: {user_text}")
    return user_text
