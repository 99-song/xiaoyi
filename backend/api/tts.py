from loguru import logger


async def synthesize_reply(
    tts_service,
    voice_prompt_manager,
    reply_text: str,
    instruct_text: str,
    user_id: int,
    session_id: str,
) -> bytes:
    voice_prompt = voice_prompt_manager.get_user_prompt(user_id)
    logger.info(f"[{session_id}] 使用音色模板: {voice_prompt['id']} | user_id={user_id}")
    audio = await tts_service.synthesize_speech_instruct(
        reply_text, instruct_text, voice_prompt["wav_path"]
    )
    logger.info(f"[{session_id}] TTS Processing complete")
    return audio
