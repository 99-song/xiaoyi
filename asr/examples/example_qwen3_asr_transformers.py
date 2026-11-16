import io
import torch
import numpy as np
import soundfile as sf
from typing import List, Optional, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from contextlib import asynccontextmanager  
from qwen_asr import Qwen3ASRModel

# 使用测试脚本中的本地路径
ASR_MODEL_PATH = "/home/songhongyi/data/xiaoyi/model/Qwen3-ASR"
FORCED_ALIGNER_PATH = "/home/songhongyi/data/xiaoyi/model/Qwen3-ForcedAligner-0.6B"

asr_model = None

def _read_wav_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """将字节流读取为音频 numpy 数组和采样率"""
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model
    print(f"Loading Qwen3-ASR model from {ASR_MODEL_PATH}...")
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
    print("✅ Model loaded successfully.")
    
    yield  # 这一行代表初始化完成，应用开始正式接收和处理外部请求
    
    # 程序结束、服务关闭时的清理工作
    print("Shutting down, releasing VRAM...")
    asr_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="Qwen3-ASR API", lifespan=lifespan)

@app.post("/transcribe")
async def transcribe(
    files: List[UploadFile] = File(..., description="上传的音频文件，支持同时上传多个文件"),
    context: List[str] = Form(default=[], description="可选：上下文提示词。如果是多个文件，可传多个"),
    language: List[str] = Form(default=[], description="可选：指定语言(如 Chinese, English)。如果是多个文件，可传多个"),
    return_time_stamps: bool = Form(default=False, description="是否返回时间戳")
):
    """
    语音识别接口，支持直接上传文件 (multipart/form-data)。
    支持多文件批量处理。
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    try:
        audio_inputs = []
        # 将所有上传的文件在内存中转为 (wav_array, sample_rate) 格式
        for file in files:
            audio_bytes = await file.read()
            wav_data, sr = _read_wav_from_bytes(audio_bytes)
            audio_inputs.append((wav_data, sr))
            
        # 补齐 context 和 language 的长度以匹配文件数量
        # 如果只传了一个 context/language，但传了多个文件，则应用到所有文件
        ctx_list = None
        if context:
            ctx_list = context * len(files) if len(context) == 1 else context
            
        lang_list = None
        if language:
            lang_list = language * len(files) if len(language) == 1 else language

        # 执行推理
        results = asr_model.transcribe(
            audio=audio_inputs,
            context=ctx_list,
            language=lang_list,
            return_time_stamps=return_time_stamps,
        )
        
        # 序列化结果
        output = []
        # asr_model.transcribe 无论是传单个还是列表，当传入列表时会返回列表
        # 这里统一处理
        if not isinstance(results, list):
            results = [results]
            
        for r, f in zip(results, files):
            res_dict = {
                "filename": f.filename,
                "language": r.language,
                "text": r.text,
                "time_stamps": []
            }
            if r.time_stamps:
                res_dict["time_stamps"] = [
                    {"text": ts.text, "start": ts.start_time, "end": ts.end_time}
                    for ts in r.time_stamps
                ]
            output.append(res_dict)
            
        return {"status": "success", "results": output}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": asr_model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)