"""
FastAPI HTTP 服务，提供流式 SSE 接口给 Web UI，支持多轮对话

接口：
  POST /query/manual       - 手写版 ReAct，流式返回每步
  POST /query/fc           - Function Calling 版，流式返回每步
  POST /session/new        - 新建会话
  DELETE /session/{id}     - 删除会话
  POST /session/{id}/clear - 清空会话历史
  GET  /health             - 健康检查

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── SessionManager ────────────────────────────────────────────────────────────
from session import SessionManager

session_manager = SessionManager()

# 每个 mode 对应的 system prompt（用于新会话初始化 + 清空时重建）
SYSTEM_PROMPTS = {}


def _init_prompts():
    from react_manual import SYSTEM_PROMPT
    from react_function_calling import FC_SYSTEM_PROMPT
    SYSTEM_PROMPTS["manual"] = SYSTEM_PROMPT
    SYSTEM_PROMPTS["fc"] = FC_SYSTEM_PROMPT


# ── 预加载 FAISS（启动时执行一次）────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引和 Embedding 模型...")
    from tools import _load_rag
    await asyncio.to_thread(_load_rag)
    _init_prompts()
    logger.info("预加载完成，服务就绪")
    yield


app = FastAPI(title="ReAct Financial Agent", lifespan=lifespan)


# ── 请求模型 ─────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:   str
    max_steps:  int = 10
    session_id: str | None = None   # None 则自动新建会话


# ── SSE ──────────────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(question: str, max_steps: int, mode: str, session_id: str | None):
    """流式 ReAct，支持多轮会话"""
    system_prompt = SYSTEM_PROMPTS[mode]

    # 获取或创建会话
    try:
        sid, session = session_manager.get_or_create(
            session_id, mode, system_prompt=system_prompt
        )
    except ValueError as e:
        yield _sse({"type": "error", "message": str(e)})
        return

    # 并发保护：同一会话同时只允许一个请求
    if not session_manager.acquire(sid):
        yield _sse({"type": "error", "message": "该会话正在处理中，请等待上一个请求完成"})
        return

    try:
        if mode == "manual":
            from react_manual import run as react_run
        else:
            from react_function_calling import run as react_run

        queue: asyncio.Queue = asyncio.Queue()
        _SENTINEL = object()

        def _worker():
            try:
                for step_data in react_run(question, max_steps=max_steps,
                                           messages=session.messages):
                    queue.put_nowait(step_data)
            except Exception as e:
                queue.put_nowait({"type": "error", "message": str(e)})
            finally:
                queue.put_nowait(_SENTINEL)

        yield _sse({"type": "start", "question": question, "mode": mode,
                     "session_id": sid})

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _worker)

        while True:
            step_data = await queue.get()
            if step_data is _SENTINEL:
                break
            yield _sse(step_data)

        yield _sse({"type": "done", "session_id": sid})

    finally:
        session_manager.release(sid)


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual", req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "fc", req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 会话管理端点 ─────────────────────────────────────────────────────────────
class NewSessionRequest(BaseModel):
    mode: str = "manual"   # "manual" | "fc"

@app.post("/session/new")
async def new_session(req: NewSessionRequest):
    sid = session_manager.create(mode=req.mode,
                                 system_prompt=SYSTEM_PROMPTS.get(req.mode, ""))
    return {"session_id": sid, "mode": req.mode}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    ok = session_manager.delete(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"deleted": session_id}


@app.post("/session/{session_id}/clear")
async def clear_session(session_id: str):
    s = session_manager.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="会话不存在")
    mode = s.mode
    session_manager.clear(session_id,
                          system_prompt=SYSTEM_PROMPTS.get(mode, ""))
    return {"cleared": session_id}


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")
