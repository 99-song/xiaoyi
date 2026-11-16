import os
import sys
import httpx
import traceback
from loguru import logger

class EmotionService:
    def __init__(self, config, timeout: float = 120.0):
        self.config = config  
        self.timeout = timeout

    async def emotion_analyze(self, wav_file: str) -> dict:
        """
        请求情感分析接口
        :param wav_file: 音频文件的绝对路径
        :return: 解析后的情感数据字典，例如 {'emotion': '难过', 'raw_label': '难过/sad', 'confidence': 0.994}
        """
        if not os.path.exists(wav_file):
            logger.error(f"Emotion analyze wav_path 文件不存在: {wav_file}")
            raise FileNotFoundError(f"Emotion analyze wav not found: {wav_file}")
        
        # 构造 JSON 请求体
        payload = {"wav_file": wav_file}
        headers = {"accept": "application/json","Content-Type": "application/json"}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # 发送 POST 请求，使用 json 参数传递 payload
                response = await client.post(
                    self.config.emotion_url, 
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
                # 解析 JSON 响应
                result = response.json()
                
                # 校验返回状态并提取数据
                if result.get("status") == "success":
                    data = result.get("data", {})
                    logger.info(f"情感检测成功: {data}")
                    return data
                else:
                    error_msg = f"情感检测接口返回业务错误: {result}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            except httpx.HTTPStatusError as e:
                logger.error(f"Emotion HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:
                error_msg = f"Emotion 请求异常: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                logger.error(error_msg)
                print(error_msg, file=sys.stderr)
                
        # 如果走到这里，说明 try 块内发生了异常并未成功 return
        raise RuntimeError("Emotion 检测失败")