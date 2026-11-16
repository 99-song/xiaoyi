import sys
import os
import io
import tempfile
import torch
import uvicorn
import yaml
import torchaudio
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import Response


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

sys.path.append('third_party/Matcha-TTS')
CUSTOM_TEMP_DIR = "/app/my_custom_temp"
os.makedirs(CUSTOM_TEMP_DIR, exist_ok=True)
tempfile.tempdir = CUSTOM_TEMP_DIR
from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI(title="TTS API", description="支持 Zero-Shot、跨语种/细粒度控制、指令控制的 TTS")
cosyvoice = None

@app.on_event("startup")
async def load_model():
    global cosyvoice
    model_dir = '/workspace/models/CosyVoice3'
    logger.info(f"正在加载模型: {model_dir} ...")
    cosyvoice = AutoModel(model_dir=model_dir)
    logger.info("模型加载完成！")

def _generate_audio_response(generator):
    """提取的公共方法：将推理生成器转为音频流返回"""
    output_audio = None
    for i, j in enumerate(generator):
        if output_audio is None:
            output_audio = j['tts_speech']
        else:
            output_audio = torch.cat([output_audio, j['tts_speech']], dim=1)

    buffer = io.BytesIO()
    torchaudio.save(buffer, output_audio, cosyvoice.sample_rate, format="wav")
    buffer.seek(0)

    return Response(
        content=buffer.read(), 
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=result.wav"
        }
    )

@app.post("/zero_shot")
async def zero_shot_inference(
    tts_text: str = Form(..., description="你需要模型说出的目标文本 (支持热修复拼音如: 高管也对报道[j][ǐ]予好评)"),
    prompt_text: str = Form(..., description="参考音频里的原话"),
    prompt_wav: UploadFile = File(..., description="参考音频文件 (.wav)")
):
    """1. Zero-Shot 声音克隆接口 (包含拼音 Hotfix)"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav.write(await prompt_wav.read())
        tmp_wav_path = tmp_wav.name

    try:
        formatted_prompt = f'You are a helpful assistant.<|endofprompt|>{prompt_text}'
        generator = cosyvoice.inference_zero_shot(tts_text, formatted_prompt, tmp_wav_path, stream=False)
        return _generate_audio_response(generator)
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

@app.post("/cross_lingual")
async def cross_lingual_inference(
    tts_text: str = Form(..., description="目标文本。可包含控制符如 [breath]，如果是日语请转为片假名。"),
    prompt_wav: UploadFile = File(..., description="参考音频文件 (.wav)")
):
    """2. Cross-Lingual 接口：用于跨语种(如日语片假名) 或 细粒度语音控制(如 [breath])"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav.write(await prompt_wav.read())
        tmp_wav_path = tmp_wav.name

    try:
        # Cross-lingual 接口的特殊格式：prompt 融合在 tts_text 前面
        formatted_text = f'You are a helpful assistant.<|endofprompt|>{tts_text}'
        generator = cosyvoice.inference_cross_lingual(formatted_text, tmp_wav_path, stream=False)
        return _generate_audio_response(generator)
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

@app.post("/instruct")
async def instruct_inference(
    tts_text: str = Form(..., description="需要合成的目标文本"),
    instruct_text: str = Form(..., description="指令内容（例如：请用广东话表达。/ 请用尽可能快地语速说一句话。）"),
    prompt_wav: UploadFile = File(..., description="参考音频文件 (.wav)")
):
    """3. Instruct 指令接口：用于控制方言、语速、情感等"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        tmp_wav.write(await prompt_wav.read())
        tmp_wav_path = tmp_wav.name
    try:
        formatted_instruct = f'You are a helpful assistant. {instruct_text}<|endofprompt|>'
        generator = cosyvoice.inference_instruct2(tts_text, formatted_instruct, tmp_wav_path, stream=False)
        return _generate_audio_response(generator)
    finally:
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

if __name__ == '__main__':
    init_logger()
    config_path = os.environ.get("XIAOYI_CONFIG_PATH", "/app/config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        api_port = int(yaml.safe_load(file)["tts"]["port"])
    uvicorn.run(app, host="0.0.0.0", port=api_port)
