from dataclasses import dataclass


@dataclass(frozen=True)
class ApiContext:
    audio_dir: str
    auth_manager: object
    session_manager: object
    voice_prompt_manager: object
    asr_service: object
    tts_service: object
    emotion_service: object
    llm_service: object
