import io
import os
import torch
import numpy as np
import tempfile
import uvicorn
import yaml
import soundfile as sf
from typing import List, Optional, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from contextlib import asynccontextmanager  
from qwen_asr import Qwen3ASRModel

import sys
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


CUSTOM_TEMP_DIR = "/app/my_custom_temp"
ASR_MODEL_PATH = "/app/models/Qwen3-ASR"
FORCED_ALIGNER_PATH = "/app/models/Qwen3-ForcedAligner-0.6B"
os.makedirs(CUSTOM_TEMP_DIR, exist_ok=True)
tempfile.tempdir = CUSTOM_TEMP_DIR
asr_model = None

def _read_wav_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model
    logger.info(f"Loading Qwen3-ASR model from {ASR_MODEL_PATH}...")
    asr_model = Qwen3ASRModel.from_pretrained(
        ASR_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        forced_aligner=FORCED_ALIGNER_PATH,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
        ),
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    logger.info("✅ Model loaded successfully.")
    
    yield
    
    logger.info("Shutting down, releasing VRAM...")
    asr_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="ASR API (Single File)", lifespan=lifespan)

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="上传单个音频文件"),
    context: str = Form(default="", description="可选：上下文提示词"),
    language: str = Form(default="", description="可选：指定语言(如 Chinese, English)"),
    return_time_stamps: bool = Form(default=False, description="是否返回时间戳")
):
    try:
        logger.info(f"ASR_wav_path: {file}")
        logger.info(f"Language: {language}")
        audio_bytes = await file.read()
        wav_data, sr = _read_wav_from_bytes(audio_bytes)
        lang_param = language if language else None
        ctx_param = context if context else ""

        # 3. 执行推理 (传入元组即可)
        results = asr_model.transcribe(
            audio=(wav_data, sr),
            context=ctx_param,
            language=lang_param,
            return_time_stamps=return_time_stamps,
        )
        
        # 4. 提取结果 (单文件推理时，如果返回的是列表则取第一项)
        r = results[0] if isinstance(results, list) else results

        # 5. 序列化结果
        res_dict = {
            "filename": file.filename,
            "language": r.language,
            "text": r.text,
            "time_stamps": []
        }
        
        if r.time_stamps:
            res_dict["time_stamps"] = [
                {"text": ts.text, "start": ts.start_time, "end": ts.end_time}
                for ts in r.time_stamps
            ]
            
        return {"status": "success", "result": res_dict}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": asr_model is not None}

if __name__ == "__main__":
    init_logger()
    config_path = os.environ.get("XIAOYI_CONFIG_PATH", "/app/config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        api_port = int(yaml.safe_load(file)["asr"]["port"])
    uvicorn.run(app, host="0.0.0.0", port=api_port)
