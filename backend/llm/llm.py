import httpx
from loguru import logger
from dataclasses import dataclass

@dataclass
class ReplyResult:
    text: str
    message_id: int | None = None

class LLMService:
    def __init__(self, session_manager, config, prompt_manager, timeout: float = 600.0):
        self.config = config
        self.thinking = False
        self.timeout = timeout
        self.session_manager = session_manager
        self.prompt_manager = prompt_manager
        
    async def reply(self, user_text: str, user_emotion: str, session_id: str, user_wav_path: str, assistant_wav_path: str, user_id: int) -> ReplyResult:
        self.session_manager.add_user_message(session_id, user_text, user_emotion, user_wav_path)
        role_type, role_prompt = self.prompt_manager.get_user_prompt(user_id)
        logger.info(f"[{session_id}] 使用角色提示词: {role_type} | user_id={user_id}")
        return await self._handle_local_llm_common(
            user_text, user_emotion, session_id, assistant_wav_path, role_prompt
        )
        
    async def reply_emotion(self, user_text: str, user_emotion: str, user_id: int):
        prompt_type, prompt = self.prompt_manager.get_user_system_prompt(user_id)
        logger.info(f"使用情绪判断提示词: {prompt_type} | user_id={user_id}")
        return await self._handle_local_llm_emotion(user_text, user_emotion, prompt)

    async def _handle_local_llm_common(self, user_text: str, user_emotion: str, session_id: str, assistant_wav_path: str, role_prompt: str) -> ReplyResult:
        messages = self.session_manager.get_messages(session_id)
        history = messages[:-1]
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_prompt": role_prompt,
            "user_input": user_text,
            "user_emotion": user_emotion,
            "history": history,
            "thinking": self.thinking,
        }

        try:
            logger.info(f"[{session_id}] sending request to local LLM ({self.config.llm_url})...")
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(self.config.llm_url, headers=headers, json=payload)
                r.raise_for_status()
                llm_text = r.json().get("content", "") 
                msg_id = self.session_manager.add_assistant_message(session_id, llm_text, assistant_wav_path)
                return ReplyResult(text=llm_text, message_id=msg_id)
        except Exception as e:
            logger.error(f"[{session_id}] Local LLM error: {e}")
            self.session_manager.rollback_last_user_message(session_id)
            return ReplyResult(text="抱歉，我刚刚有点走神了，你可以再和我说一次。",message_id=None)


    async def _handle_local_llm_emotion(self, user_text: str, user_emotion: str, prompt: str):
        params = {"is_one": True}
        headers = {"Content-Type": "application/json"}
        payload = {"prompt": prompt, "user_input": user_text, "user_emotion": user_emotion, "thinking": self.thinking}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(self.config.llm_url, headers=headers, json=payload, params=params)
                r.raise_for_status()
                llm_text = r.json().get("content", "") 
                return llm_text
        except Exception as e:
            logger.info(f"情绪检测错误：{e}")
            return "请以正常的语气说这句话"
        
