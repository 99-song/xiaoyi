import sys
import os
import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from funasr import AutoModel
from loguru import logger
from contextlib import asynccontextmanager # 新增导入


def init_logger():
    logger.remove()
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>"
    file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {function} - {message}"

    logger.add(sys.stderr, format=console_format, level="INFO")
    os.makedirs("logs", exist_ok=True)
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


class EmotionRequest(BaseModel):
    wav_file: str
    


class EmotionRecognizer:
    def __init__(self, model_path: str):
        self.model = AutoModel(
            model=model_path,
            hub="ms",
        )
            
    def analyze(self, wav_file: str) -> dict:
        rec_result = self.model.generate(
            wav_file,
            output_dir="./outputs",
            granularity="utterance",
            extract_embedding=False,
        )

        emotion_dict = {}
        if rec_result and isinstance(rec_result, list):
            item = rec_result[0]                     
            scores = item["scores"]
            labels = item["labels"]
            max_index = scores.index(max(scores))    
            best_label = labels[max_index]           
            chinese_label = best_label.split("/")[0] 
            
            emotion_dict = {
                "emotion": chinese_label,
                "raw_label": best_label,
                "confidence": scores[max_index]      
            }

        return emotion_dict


# 全局变量
recognizer = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognizer
    # model_dir = "/home/songhongyi/data/xiaoyi/model/emotion2vec_base_finetuned"emotion2vec_plus_large
    # model_dir = "./model/emotion2vec_base_finetuned"
    model_dir = "./model/emotion2vec_plus_large"
    
    
    if not os.path.exists(model_dir):
        logger.error(f"找不到模型路径: {model_dir}")
        raise FileNotFoundError(f"找不到模型路径: {model_dir}")

    logger.info(f"正在加载情感识别模型: {model_dir} ...")
    recognizer = EmotionRecognizer(model_path=model_dir)
    logger.info("模型加载完成，服务已就绪！")
    yield 
    logger.info("正在关闭服务，清理模型资源...")
    recognizer = None

app = FastAPI(title="Emotion Recognition API", description="基于 emotion2vec_base_finetuned 的情感识别接口",lifespan=lifespan)

@app.post("/analyze")
async def analyze_emotion(request: EmotionRequest):
    logger.info(f"收到分析请求，音频文件路径: {request.wav_file}")
    
    if not os.path.exists(request.wav_file):
        logger.error(f"找不到指定的音频文件: {request.wav_file}")
        return {
            "status": "error", 
            "message": f"找不到指定的音频文件: {request.wav_file}"
        }
    
    try:
        emotion_result = recognizer.analyze(wav_file=request.wav_file)
        logger.info(f"分析完成: {emotion_result}")
        
        return {
            "status": "success",
            "data": emotion_result
        }
        
    except Exception as e:
        logger.error(f"推理过程中发生错误: {str(e)}")
        return {"status": "error", "message": str(e)}


if __name__ == '__main__':
    init_logger()
    logger.info("启动 FastAPI 服务...")
    config_path = os.environ.get("XIAOYI_CONFIG_PATH", "/app/config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        api_port = int(yaml.safe_load(file)["emotion"]["port"])
    uvicorn.run(app, host="0.0.0.0", port=api_port)
