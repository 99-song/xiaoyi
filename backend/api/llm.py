from loguru import logger

from tools.text import clean_text_for_tts


async def generate_reply(
    llm_service,
    user_text: str,
    user_emotion: str,
    session_id: str,
    user_wav_path: str,
    assistant_wav_path: str,
    user_id: int,
):
    logger.info(f"[{session_id}] LLM Processing...")
    result = await llm_service.reply(
        user_text,
        user_emotion,
        session_id,
        user_wav_path,
        assistant_wav_path,
        user_id,
    )
    reply_text = clean_text_for_tts(result.text)
    if not reply_text or len(reply_text) < 2:
        logger.error(f"[{session_id}] LLM returned empty or too short text, using fallback")
        reply_text = "我刚才走神了，能再说一遍吗？"
    logger.info(f"[{session_id}] LLM Reply: {reply_text}")
    return result, reply_text


async def generate_reply_emotion(
    llm_service, user_text: str, user_emotion: str, user_id: int, session_id: str
) -> str:
    instruct_text = await llm_service.reply_emotion(user_text, user_emotion, user_id)
    instruct_text = clean_text_for_tts(instruct_text)
    logger.info(f"[{session_id}] LLM Reply Emotion: {instruct_text}")
    return instruct_text
