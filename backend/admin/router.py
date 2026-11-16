import os
import tempfile
from typing import Callable, Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from loguru import logger
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pydantic import BaseModel, Field


class UserCreateRequest(BaseModel):
    username: str = Field(..., max_length=50)
    password: str = Field(..., max_length=128)
    role_type: Optional[str] = Field(None, max_length=50)
    emotion_prompt_type: Optional[str] = Field(None, max_length=50)
    voice_prompt_id: Optional[int] = None


class UserUpdateRequest(BaseModel):
    username: Optional[str] = Field(None, max_length=50)
    password: Optional[str] = Field(None, max_length=128)


class PromptCreateRequest(BaseModel):
    type: str = Field(..., max_length=50)
    content: str = Field(..., max_length=10000)
    category: str = Field("role", pattern="^(role|system)$")


class PromptUpdateRequest(BaseModel):
    content: str = Field(..., max_length=10000)
    new_type: Optional[str] = Field(None, max_length=50)
    category: Optional[str] = Field(None, pattern="^(role|system)$")


class RoleAssignmentRequest(BaseModel):
    role_type: str = Field(..., max_length=50)


class EmotionPromptAssignmentRequest(BaseModel):
    emotion_prompt_type: str = Field(..., max_length=50)


class VoicePromptAssignmentRequest(BaseModel):
    voice_prompt_id: int


class PromptGenerateRequest(BaseModel):
    description: str = Field(..., min_length=2, max_length=2000)
    category: str = Field(..., pattern="^(role|system)$")


def create_admin_router(auth_manager, session_manager, prompt_manager, voice_prompt_manager, asr_service, config, require_admin: Callable) -> APIRouter:
    """创建仅数据库系统管理员可访问的管理路由。"""
    router = APIRouter(prefix="/admin", tags=["用户管理"])

    async def save_audio_as_wav(audio: UploadFile) -> str:
        data = await audio.read()
        if not data:
            raise HTTPException(status_code=400, detail="音频文件不能为空")
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="音频文件不能超过 20MB")
        source_suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
        source_file = tempfile.NamedTemporaryFile(suffix=source_suffix, delete=False)
        wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        source_path, wav_path = source_file.name, wav_file.name
        try:
            source_file.write(data)
            source_file.close()
            wav_file.close()
            converted = AudioSegment.from_file(source_path).set_frame_rate(16000).set_channels(1)
            if len(converted) < 300:
                raise ValueError("音频时间太短")
            silence_threshold = min(converted.dBFS - 14, -35) if converted.dBFS != float("-inf") else -35
            ranges = detect_nonsilent(converted, min_silence_len=120, silence_thresh=silence_threshold)
            if not ranges:
                raise ValueError("音频中没有检测到有效声音")
            start = max(0, ranges[0][0] - 80)
            end = min(len(converted), ranges[-1][1] + 80)
            converted[start:end].export(wav_path, format="wav")
            return wav_path
        except Exception as exc:
            try:
                os.remove(wav_path)
            except FileNotFoundError:
                pass
            raise HTTPException(status_code=400, detail="无法解析音频，请上传常见音频格式") from exc
        finally:
            try:
                os.remove(source_path)
            except FileNotFoundError:
                pass

    def validate_username(username: str) -> str:
        username = username.strip()
        if len(username) < auth_manager.MIN_USERNAME_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"用户名至少需要 {auth_manager.MIN_USERNAME_LENGTH} 个字符",
            )
        return username

    def validate_password(password: str) -> str:
        if len(password) < auth_manager.MIN_PASSWORD_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"密码至少需要 {auth_manager.MIN_PASSWORD_LENGTH} 个字符",
            )
        return password

    @router.get("/users")
    async def list_users(admin: dict = Depends(require_admin)):
        logger.info(f"[AUTH] 查询用户列表 | operator={admin['username']}")
        return {"users": auth_manager.list_users()}

    @router.get("/account")
    async def get_admin_account(admin: dict = Depends(require_admin)):
        return {"user": auth_manager.get_user(admin["id"])}

    @router.patch("/account")
    async def update_admin_account(
        payload: UserUpdateRequest,
        request: Request,
        admin: dict = Depends(require_admin),
    ):
        if payload.username is None and payload.password is None:
            raise HTTPException(status_code=400, detail="至少提供一个需要修改的字段")
        username = validate_username(payload.username) if payload.username is not None else None
        password = validate_password(payload.password) if payload.password is not None else None
        try:
            updated = auth_manager.update_user(admin["id"], username, password)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info(
            f"[AUTH] 修改系统管理员账号 | operator={admin['username']} | "
            f"new_username={updated['username']} | password_changed={password is not None} | "
            f"ip={request.client.host if request.client else 'unknown'}"
        )
        return {
            "message": "管理员账号已更新",
            "user": updated,
            "reauthenticate": password is not None,
        }

    @router.get("/users/{user_id}")
    async def get_user(user_id: int, admin: dict = Depends(require_admin)):
        target = auth_manager.get_user(user_id)
        if not target or target["is_admin"]:
            raise HTTPException(status_code=404, detail="用户不存在")
        logger.info(
            f"[AUTH] 查询用户详情 | operator={admin['username']} | "
            f"target={target['username']} | target_id={user_id}"
        )
        return {"user": target}

    @router.post("/users", status_code=201)
    async def create_user(
        payload: UserCreateRequest,
        request: Request,
        admin: dict = Depends(require_admin),
    ):
        username = validate_username(payload.username)
        password = validate_password(payload.password)
        role_type = payload.role_type or prompt_manager.default_role
        emotion_prompt_type = payload.emotion_prompt_type or prompt_manager.default_system_prompt
        assignable_roles = {
            item["type"] for item in prompt_manager.list_prompts() if item["assignable"]
        }
        if role_type not in assignable_roles:
            raise HTTPException(status_code=400, detail="角色不存在或不可分配")
        system_prompts = {
            item["type"] for item in prompt_manager.list_prompts() if item["category"] == "system"
        }
        if emotion_prompt_type not in system_prompts:
            raise HTTPException(status_code=400, detail="情绪判断提示词不存在或不可分配")
        voice_prompt_id = payload.voice_prompt_id or voice_prompt_manager.default_id
        if voice_prompt_id is None:
            raise HTTPException(status_code=400, detail="请先创建至少一个参考音色")
        if not voice_prompt_manager.get_prompt(voice_prompt_id):
            raise HTTPException(status_code=400, detail="音色模板不存在或不可分配")
        try:
            created = auth_manager.register(username, password)
            prompt_manager.assign_user_role(created["id"], role_type)
            prompt_manager.assign_user_system_prompt(created["id"], emotion_prompt_type)
            voice_prompt_manager.assign_user_prompt(
                created["id"], voice_prompt_id
            )
            created = auth_manager.get_user(created["id"])
        except ValueError as exc:
            logger.warning(
                f"[AUTH] 注册失败 | operator={admin['username']} | target={username} | "
                f"ip={request.client.host if request.client else 'unknown'} | reason={exc}"
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info(
            f"[AUTH] 注册成功 | operator={admin['username']} | target={created['username']} | "
            f"user_id={created['id']} | ip={request.client.host if request.client else 'unknown'}"
        )
        return {"message": "注册成功", "user": created}

    @router.put("/users/{user_id}/role")
    async def assign_user_role(
        user_id: int,
        payload: RoleAssignmentRequest,
        request: Request,
        admin: dict = Depends(require_admin),
    ):
        target = auth_manager.get_user(user_id)
        if not target or target["is_admin"]:
            raise HTTPException(status_code=404, detail="用户不存在")
        try:
            prompt_manager.assign_user_role(user_id, payload.role_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        updated = auth_manager.get_user(user_id)
        logger.info(
            f"[ROLE] 分配用户角色 | operator={admin['username']} | target={target['username']} | "
            f"role={payload.role_type} | ip={request.client.host if request.client else 'unknown'}"
        )
        return {"message": "角色已分配", "user": updated}

    @router.put("/users/{user_id}/emotion-prompt")
    async def assign_user_emotion_prompt(
        user_id: int,
        payload: EmotionPromptAssignmentRequest,
        request: Request,
        admin: dict = Depends(require_admin),
    ):
        target = auth_manager.get_user(user_id)
        if not target or target["is_admin"]:
            raise HTTPException(status_code=404, detail="用户不存在")
        try:
            prompt_manager.assign_user_system_prompt(user_id, payload.emotion_prompt_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        updated = auth_manager.get_user(user_id)
        logger.info(
            f"[ROLE] 分配情绪判断提示词 | operator={admin['username']} | target={target['username']} | "
            f"prompt={payload.emotion_prompt_type} | ip={request.client.host if request.client else 'unknown'}"
        )
        return {"message": "情绪判断提示词已分配", "user": updated}

    @router.put("/users/{user_id}/voice-prompt")
    async def assign_user_voice_prompt(
        user_id: int,
        payload: VoicePromptAssignmentRequest,
        request: Request,
        admin: dict = Depends(require_admin),
    ):
        target = auth_manager.get_user(user_id)
        if not target or target["is_admin"]:
            raise HTTPException(status_code=404, detail="用户不存在")
        try:
            voice_prompt_manager.assign_user_prompt(user_id, payload.voice_prompt_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info(
            f"[VOICE] 分配用户音色 | operator={admin['username']} | target={target['username']} | "
            f"voice_prompt_id={payload.voice_prompt_id} | ip={request.client.host if request.client else 'unknown'}"
        )
        return {"message": "音色模板已分配", "user": auth_manager.get_user(user_id)}

    @router.get("/voice-prompts")
    async def list_voice_prompts(admin: dict = Depends(require_admin)):
        return {"voice_prompts": voice_prompt_manager.list_prompts()}

    @router.post("/voice-prompts/transcribe")
    async def transcribe_voice_prompt(
        audio: UploadFile = File(...),
        admin: dict = Depends(require_admin),
    ):
        data = await audio.read()
        if not data:
            raise HTTPException(status_code=400, detail="音频文件不能为空")
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="音频文件不能超过 20MB")
        text = (await asr_service.transcribe_audio(data)).strip()
        if not text:
            raise HTTPException(status_code=422, detail="ASR 未识别出有效文本，请检查音频后重试")
        logger.info(f"[VOICE] ASR 识别参考音频 | operator={admin['username']} | filename={audio.filename}")
        return {"text": text}

    @router.get("/voice-prompts/{prompt_id}/audio")
    async def get_voice_prompt_audio(
        prompt_id: int,
        download: bool = Query(False),
        admin: dict = Depends(require_admin),
    ):
        prompt = voice_prompt_manager.get_prompt(prompt_id)
        if not prompt or not os.path.isfile(prompt["wav_path"]):
            raise HTTPException(status_code=404, detail="音色音频不存在")
        if download:
            return FileResponse(
                prompt["wav_path"], media_type="audio/wav", filename=f"{prompt['name']}.wav"
            )
        return FileResponse(
            prompt["wav_path"], media_type="audio/wav",
            headers={"Content-Disposition": "inline", "Cache-Control": "no-store"},
        )

    @router.post("/voice-prompts", status_code=201)
    async def create_voice_prompt(
        name: str = Form(..., max_length=100),
        text: str = Form(..., max_length=1000),
        audio: UploadFile = File(...),
        admin: dict = Depends(require_admin),
    ):
        wav_path = await save_audio_as_wav(audio)
        try:
            prompt = voice_prompt_manager.create_prompt(name, text, wav_path)
        except ValueError as exc:
            try:
                os.remove(wav_path)
            except FileNotFoundError:
                pass
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info(f"[VOICE] 新增音色模板 | operator={admin['username']} | id={prompt['id']}")
        return {"message": "音色模板已创建", "voice_prompt": prompt}

    @router.put("/voice-prompts/{prompt_id}")
    async def update_voice_prompt(
        prompt_id: int,
        name: str = Form(..., max_length=100),
        text: str = Form(..., max_length=1000),
        audio: Optional[UploadFile] = File(None),
        admin: dict = Depends(require_admin),
    ):
        wav_path = await save_audio_as_wav(audio) if audio and audio.filename else None
        try:
            prompt = voice_prompt_manager.update_prompt(prompt_id, name, text, wav_path)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info(f"[VOICE] 修改音色模板 | operator={admin['username']} | id={prompt_id}")
        return {"message": "音色模板已更新", "voice_prompt": prompt}

    @router.delete("/voice-prompts/{prompt_id}")
    async def delete_voice_prompt(prompt_id: int, admin: dict = Depends(require_admin)):
        try:
            voice_prompt_manager.delete_prompt(prompt_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info(f"[VOICE] 删除音色模板 | operator={admin['username']} | id={prompt_id}")
        return {"message": "音色模板已删除"}

    @router.get("/prompts")
    async def list_prompts(admin: dict = Depends(require_admin)):
        logger.info(f"[ROLE] 查询角色提示词 | operator={admin['username']}")
        return {"prompts": prompt_manager.list_prompts()}

    @router.post("/prompts/generate")
    async def generate_prompt(
        payload: PromptGenerateRequest,
        admin: dict = Depends(require_admin),
    ):
        if payload.category == "role":
            task = (
                "请根据管理员提供的角色信息，生成一份可直接作为大语言模型 system prompt 的完整角色提示词。"
                "需要清楚描述身份、服务对象、回答风格、行为边界和安全约束。"
            )
        else:
            task = (
                "请根据管理员提供的需求，生成一份用于判断回复情绪和语气的完整系统提示词。"
                "应明确输入信息、输出格式、允许的情绪范围和判断规则。"
            )
        request_body = {
            "system_prompt": task + "只输出最终提示词正文，不要解释，不要使用 Markdown 代码块。",
            "user_input": payload.description.strip(),
            "user_emotion": "",
            "history": [],
            "thinking": False,
        }
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(config.llm_url, json=request_body)
                response.raise_for_status()
                result = response.json()
        except Exception as exc:
            logger.error(f"[PROMPT] 本地 LLM 生成失败: {exc}")
            raise HTTPException(status_code=502, detail="本地 LLM 生成失败，请检查 LLM 服务") from exc
        content = str(result.get("content", "")).strip()
        if result.get("status") != "success" or not content:
            raise HTTPException(status_code=502, detail="本地 LLM 未返回有效提示词")
        logger.info(f"[PROMPT] 本地 LLM 生成提示词 | operator={admin['username']} | category={payload.category}")
        return {"content": content}

    @router.post("/prompts", status_code=201)
    async def create_prompt(
        payload: PromptCreateRequest,
        request: Request,
        admin: dict = Depends(require_admin),
    ):
        try:
            prompt = prompt_manager.create_prompt(payload.type, payload.content, payload.category)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info(
            f"[ROLE] 新增角色提示词 | operator={admin['username']} | role={prompt['type']} | "
            f"ip={request.client.host if request.client else 'unknown'}"
        )
        return {"message": "角色已创建", "prompt": prompt}

    @router.put("/prompts/{prompt_type}")
    async def update_prompt(
        prompt_type: str,
        payload: PromptUpdateRequest,
        request: Request,
        admin: dict = Depends(require_admin),
    ):
        try:
            prompt = prompt_manager.update_prompt(
                prompt_type, payload.content, payload.new_type, payload.category
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info(
            f"[ROLE] 修改角色提示词 | operator={admin['username']} | role={prompt_type} | "
            f"ip={request.client.host if request.client else 'unknown'}"
        )
        return {"message": "提示词已更新", "prompt": prompt}

    @router.delete("/prompts/{prompt_type}")
    async def delete_prompt(
        prompt_type: str,
        request: Request,
        category: Optional[str] = Query(None, pattern="^(role|system)$"),
        admin: dict = Depends(require_admin),
    ):
        try:
            prompt_manager.delete_prompt(prompt_type, category)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info(
            f"[ROLE] 删除角色提示词 | operator={admin['username']} | role={prompt_type} | "
            f"ip={request.client.host if request.client else 'unknown'}"
        )
        return {"message": "角色已删除"}

    @router.patch("/users/{user_id}")
    async def update_user(
        user_id: int,
        payload: UserUpdateRequest,
        request: Request,
        admin: dict = Depends(require_admin),
    ):
        target = auth_manager.get_user(user_id)
        if not target or target["is_admin"]:
            raise HTTPException(status_code=404, detail="用户不存在")
        if payload.username is None and payload.password is None:
            raise HTTPException(status_code=400, detail="至少提供一个需要修改的字段")
        username = validate_username(payload.username) if payload.username is not None else None
        password = validate_password(payload.password) if payload.password is not None else None
        try:
            updated = auth_manager.update_user(user_id, username, password)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        changed_fields = [
            name for name, value in (("username", payload.username), ("password", payload.password))
            if value is not None
        ]
        logger.info(
            f"[AUTH] 修改用户成功 | operator={admin['username']} | target={updated['username']} | "
            f"target_id={user_id} | fields={','.join(changed_fields)} | "
            f"ip={request.client.host if request.client else 'unknown'}"
        )
        return {"message": "用户已更新", "user": updated}

    @router.delete("/users/{user_id}")
    async def delete_user(
        user_id: int,
        request: Request,
        admin: dict = Depends(require_admin),
    ):
        target = auth_manager.get_user(user_id)
        if not target or target["is_admin"]:
            raise HTTPException(status_code=404, detail="用户不存在")
        session_manager.delete_user_sessions(user_id)
        auth_manager.delete_user(user_id)
        logger.info(
            f"[AUTH] 删除用户成功 | operator={admin['username']} | target={target['username']} | "
            f"target_id={user_id} | ip={request.client.host if request.client else 'unknown'}"
        )
        return {"message": "用户已删除"}

    return router
