import io
import httpx
from pydub import AudioSegment
from loguru import logger

class ASRService:
    def __init__(self, config, timeout: float = 120.0):
        self.config = config
        self.timeout = timeout
        
    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        if len(audio_bytes) < 500:
            logger.error(f"录音数据太短 ({len(audio_bytes)} 字节)，可能是前端采集失败")
            return ""
            
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            processed_audio = wav_io.getvalue()
        except Exception as e:
            logger.error(f"音频转换失败: {e}")
            return ""
            
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            files = {'file': ('user_audio.wav', processed_audio, 'audio/wav')}
            data = {}
            
            if self.config.asr_language:
                data['language'] = str(self.config.asr_language)
                
            logger.info(f"ASR: {self.config.asr_url}  {data.get('language', 'default')}")
            
            try:
                response = await client.post(self.config.asr_url, data=data, files=files)
                response.raise_for_status() 
                
                res_data = response.json()
                if res_data.get("status") == "success":
                    return res_data["result"]["text"]
                else:
                    logger.error(f"ASR Error Response: {res_data}")
                    return ""
                    
            except Exception as e:
                logger.error(f"ASR Request Error: {str(e)}")
                return ""
