
import os
import sys
import httpx
import traceback
from loguru import logger

class TTSService:
    def __init__(self, config, timeout: float = 120.0):
        self.config = config
        self.timeout = timeout

    async def synthesize_speech_zero_short(self, text: str, prompt_text: str, prompt_wav: str) -> bytes:
        if not os.path.exists(prompt_wav):
            logger.error(f"prompt_wav 文件不存在: {prompt_wav}")
            raise FileNotFoundError(f"Prompt wav not found: {prompt_wav}")
        
        data = {"tts_text": text, "prompt_text": prompt_text}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                with open(prompt_wav, "rb") as f:
                    files = {"prompt_wav": (os.path.basename(prompt_wav), f, "audio/wav")}
                    response = await client.post(self.config.tts_url_zero_shot, data=data, files=files)
                    response.raise_for_status()
                    return response.content
            except httpx.HTTPStatusError as e:
                logger.error(f"TTS HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:
                error_msg = f"TTS 请求异常: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                logger.error(error_msg)
                print(error_msg, file=sys.stderr)
                
        raise RuntimeError("TTS 合成失败")
    
    async def synthesize_speech_instruct(self, text: str, instruct_text: str, prompt_wav: str) -> bytes:
        if not os.path.exists(prompt_wav):
            logger.error(f"prompt_wav 文件不存在: {prompt_wav}")
            raise FileNotFoundError(f"Prompt wav not found: {prompt_wav}")
        
        data = {"tts_text": text, "instruct_text": instruct_text}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                with open(prompt_wav, "rb") as f:
                    files = {"prompt_wav": (os.path.basename(prompt_wav), f, "audio/wav")}
                    response = await client.post(self.config.tts_url_instruct, data=data, files=files)
                    response.raise_for_status()
                    return response.content
            except httpx.HTTPStatusError as e:
                logger.error(f"TTS HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:
                error_msg = f"TTS 请求异常: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                logger.error(error_msg)
                print(error_msg, file=sys.stderr)
                
        raise RuntimeError("TTS 合成失败")
