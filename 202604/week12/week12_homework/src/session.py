"""
会话管理器：内存存储多轮对话历史

教学重点：
  1. Session 就是 messages list 的容器——messages 本身就是记忆
  2. 会话绑定 mode（manual/fc），互换会出错
  3. 并发保护：同一会话同时只允许一个请求处理

使用方式：
  from session import SessionManager
  mgr = SessionManager()
  sid = mgr.create(mode="manual", system_prompt="你是金融分析助手...")
  session = mgr.get(sid)
  session.messages.append(...)
"""

import time
import uuid
import threading
from dataclasses import dataclass, field


@dataclass
class Session:
    session_id: str
    mode: str                            # "manual" | "fc"
    messages: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    processing: bool = False             # 并发保护：同一会话同时只处理一个请求


class SessionManager:
    """内存会话管理器，支持并发安全读写"""

    def __init__(self):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ── 创建 / 获取 ──────────────────────────────────────────────────────────

    def create(self, mode: str, system_prompt: str = "") -> str:
        """新建会话，返回 session_id。messages 初始化为 [system_prompt]（如果提供）"""
        session_id = uuid.uuid4().hex[:12]
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        with self._lock:
            self._sessions[session_id] = Session(
                session_id=session_id,
                mode=mode,
                messages=messages,
            )
        return session_id

    def get(self, session_id: str) -> Session | None:
        """获取会话，自动刷新 last_active"""
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.last_active = time.time()
            return s

    def get_or_create(self, session_id: str | None, mode: str,
                      system_prompt: str = "") -> tuple[str, Session]:
        """如果 session_id 为 None 或不存在则新建；存在则校验 mode 一致。返回 (session_id, session)"""
        if session_id:
            s = self.get(session_id)
            if s:
                if s.mode != mode:
                    raise ValueError(
                        f"会话 {session_id} 的 mode 为 '{s.mode}'，"
                        f"与请求的 '{mode}' 不匹配"
                    )
                return session_id, s
        # 新建
        sid = self.create(mode=mode, system_prompt=system_prompt)
        return sid, self.get(sid)

    # ── 更新 / 清空 / 删除 ───────────────────────────────────────────────────

    def clear(self, session_id: str, system_prompt: str = "") -> bool:
        """清空会话历史，重置为 [system_prompt]"""
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return False
            s.messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            s.processing = False
            s.last_active = time.time()
            return True

    def delete(self, session_id: str) -> bool:
        """删除会话"""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    # ── 并发保护 ────────────────────────────────────────────────────────────

    def acquire(self, session_id: str) -> bool:
        """尝试获取处理锁，成功返回 True，已被占用或不存在返回 False"""
        with self._lock:
            s = self._sessions.get(session_id)
            if not s or s.processing:
                return False
            s.processing = True
            return True

    def release(self, session_id: str):
        """释放处理锁"""
        with self._lock:
            s = self._sessions.get(session_id)
            if s:
                s.processing = False

    # ── 过期清理 ────────────────────────────────────────────────────────────

    def cleanup_expired(self, ttl: float = 3600):
        """清理超过 ttl 秒未活动的会话"""
        now = time.time()
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items()
                if now - s.last_active > ttl and not s.processing
            ]
            for sid in expired:
                del self._sessions[sid]
        return len(expired)
