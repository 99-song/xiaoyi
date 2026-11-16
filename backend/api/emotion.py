from loguru import logger

from tools.common import extract_emotion


async def analyze_user_emotion(emotion_service, audio_path: str, session_id: str) -> str:
    try:
        result = await emotion_service.emotion_analyze(audio_path)
        emotion = result.get("emotion", "未知")
        logger.info(f"[{session_id}] User Emotion: {emotion}")
        return emotion
    except Exception as exc:
        logger.error(f"[{session_id}] 用户情感检测失败: {exc}")
        return "未知"


def extract_assistant_emotion(instruct_text: str, session_id: str) -> str:
    try:
        emotion = extract_emotion(instruct_text)
        logger.info(f"[{session_id}] Assistant Emotion: {emotion}")
        return emotion
    except Exception as exc:
        logger.error(f"[{session_id}] 助手情感检测失败: {exc}")
        return "未知"
