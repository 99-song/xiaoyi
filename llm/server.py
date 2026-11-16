import sys
import os
import yaml
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger


def init_logger():
    logger.remove()
    console_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>"
    file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {function} - {message}"

    # 控制台输出
    logger.add(sys.stderr, format=console_format, level="INFO")
    
    # 文件输出
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


class ChatRequest(BaseModel):
    system_prompt: str = ""
    prompt: str = ""
    user_input: str
    user_emotion: str
    history: Optional[List[Dict[str, str]]] = []
    thinking: Optional[bool] = False
    

class QwenChatbot:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype="auto",device_map="auto")
        
        with open('./config.yaml', "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) 
            
    def generate_response(self, messages, thinking=False):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config['max_new_tokens'], temperature=self.config['temperature'], top_p=self.config['top_p'], top_k=20, do_sample=True)
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return thinking_content, content


app = FastAPI(title="Qwen API", description="Qwen3-1.7B")
chatbot = None


@app.on_event("startup")
async def load_model():
    global chatbot
    model_dir = "/app/model/Qwen3-1.7B"
    # model_dir = "/home/songhongyi/data/xiaoyi/model/Qwen3-1.7B"
    
    if not os.path.exists(model_dir):
        logger.error(f"找不到模型路径: {model_dir}")
        raise FileNotFoundError(f"找不到模型路径: {model_dir}")

    logger.info(f"正在加载模型: {model_dir} ...")
    chatbot = QwenChatbot(model_path=model_dir)
    logger.info("模型加载完成！")


@app.post("/chat")
async def chat_inference(request: ChatRequest, is_one: bool = Query(False)):
    messages = []
    if not is_one:
        system_parts = []
        if request.system_prompt:
            system_parts.append(request.system_prompt)

        # 兼容旧客户端：普通对话中的 prompt 同样视为系统提示词。
        if request.prompt:
            system_parts.append(request.prompt)

        if request.user_emotion:
            system_parts.append(
                f"当前用户情绪为：{request.user_emotion}，请根据情绪调整回复语气。"
            )

        if system_parts:
            messages.append({"role": "system", "content": "\n".join(system_parts)})

        if request.history:
            for msg in request.history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role not in ["user", "assistant"]:
                    logger.warning(f"Invalid role: {role} user")
                    role = "user"
                messages.append({"role": role,"content": content})

        messages.append({"role": "user", "content": request.user_input})
        logger.info(f"Final messages: {messages}")

    else:
        combined_content = ""

        if request.prompt:
            combined_content += f"{request.prompt}\n"

        if request.user_emotion:
            combined_content += f"【背景信息】当前老人的情绪被检测为：{request.user_emotion}。\n"
        combined_content += f"【老人说的话】：{request.user_input}"
        messages.append({"role": "user", "content": combined_content})
        
        logger.info(f"Final messages: {messages}")
    
    try:
        thinking, content = chatbot.generate_response(messages=messages,thinking=request.thinking)
        messages.append({"role": "assistant", "content": content})
        
        logger.info(f"Thinking content: {thinking}")
        logger.info(f"Generated content: {content}")
        
        return {
            "status": "success",
            "thinking_content": thinking,
            "content": content,
            "history": messages
        }
        
    except Exception as e:
        logger.error(f"推理过程中发生错误: {str(e)}")
        return {"status": "error", "message": str(e)}
    


if __name__ == '__main__':
    init_logger()
    logger.info("启动 FastAPI 服务...")
    config_path = os.environ.get("XIAOYI_CONFIG_PATH", "/app/config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        api_port = int(yaml.safe_load(file)["llm"]["port"])
    uvicorn.run(app, host="0.0.0.0", port=api_port)
