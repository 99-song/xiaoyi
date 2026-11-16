import re
import sys
import yaml
from loguru import logger

def init_logger():
    logger.remove()
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>"
    file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {function} - {message}"

    # 1. 控制台输出
    logger.add(
        sys.stderr, 
        format=console_format, 
        level="INFO"
    )
    
    # 2. 文件输出
    logger.add(
        "logs/{time:YYYY-MM-DD}.log", 
        format=file_format,      
        level="INFO",             
        rotation="10 MB",       
        retention="5 days",     
        compression="zip",       
        encoding="utf-8",        
        enqueue=True             
    )
    
    return logger

class AppConfig:
    def __init__(self, config_yaml: str = "./config.yaml"):
        self._load_yaml(config_yaml)

    def _load_yaml(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) 
        self.backend_port = int(config['backend']['port'])
        api_host = config['api']['host']
        self.asr_url = self._api_url(api_host, config['asr']['port'], config['asr']['path'])
        self.asr_language = config['asr'].get('language', '')
        self.tts_url_zero_shot = self._api_url(api_host, config['tts']['port'], config['tts']['zero_shot_path'])
        self.tts_url_instruct = self._api_url(api_host, config['tts']['port'], config['tts']['instruct_path'])
        self.emotion_url = self._api_url(api_host, config['emotion']['port'], config['emotion']['path'])
        self.llm_url = self._api_url(api_host, config['llm']['port'], config['llm']['path'])

    @staticmethod
    def _api_url(host: str, port: int, path: str) -> str:
        return f"http://{host}:{int(port)}{path if path.startswith('/') else '/' + path}"

def extract_emotion(llm_output: str) -> str:
    if not llm_output:
        return ""
    
    text = llm_output.strip()
    
    match = re.search(r'请用\s*\[(.+?)\]\s*的语气说这一句话', text)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'用\s*\[(.+?)\]', text)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'用\s*\((.+?)\)', text)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'用\s*[“”‘\'"](.+?)[”’\'"]', text)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'用\s*[《【<](.+?)[》】>]', text)
    if match:
        return match.group(1).strip()
    
    match = re.search(r'用\s*([^用，。,\.\s]+?)\s*的\s*语气\s*说这一句话', text)
    if match:
        return match.group(1).strip()
    
    # 7. 更宽松：“用...说”或“用...语气”，提取“用”和“说/语气”之间的部分
    match = re.search(r'用\s*(.+?)\s*(?:的\s*(?:情绪|语气|口吻|方式|语调|回复|回答))?\s*[说回]', text)
    if match:
        candidate = match.group(1).strip()
        # 排除过长或含标点的提取
        if candidate and len(candidate) <= 10 and re.fullmatch(r'[\u4e00-\u9fff\w]+', candidate):
            return candidate
    
    # 8. 最简单情况：句子以“用”开头，后面跟着情绪词
    match = re.match(r'用\s*([\u4e00-\u9fff]+)', text)
    if match:
        return match.group(1)
    
    return "平和"
