import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend import database
from backend.app.auth.middleware import SESSION_COOKIE
from backend.app.auth.store import get_session, get_user
from backend.services.pipeline import register_ws_subscriber, unregister_ws_subscriber

logger = logging.getLogger(__name__)

router = APIRouter()


async def _authorize_ws(websocket: WebSocket, job_id: str) -> bool:
    """Authenticate a WebSocket connection + verify it owns ``job_id``.

    The FastAPI HTTP middleware doesn't run on WebSocket upgrades, so we
    resolve the session cookie manually here. Without this gate, any
    user who knows (or guesses) another user's job_id could subscribe
    to their job's real-time progress messages.
    """
    token = websocket.cookies.get(SESSION_COOKIE, "")
    if not token:
        return False
    session = await get_session(token)
    if session is None:
        return False
    user = await get_user(session.user_id)
    if user is None or not user.active:
        return False

    job = await database.load_job(job_id)
    if not job:
        return False
    owner = getattr(job, "owner_user_id", "") or ""
    owner_name = (getattr(job, "owner_username", "") or "").strip().lower()
    caller_name = (user.username or "").strip().lower()
    if owner and owner == user.id:
        return True
    if owner_name and caller_name and owner_name == caller_name:
        return True
    # Legacy unowned jobs: only the head admin can subscribe.
    if not owner and getattr(user, "head_admin", False):
        return True
    return False


@router.websocket("/ws/jobs/{job_id}")
async def websocket_job_progress(websocket: WebSocket, job_id: str):
    if not await _authorize_ws(websocket, job_id):
        # Close with policy-violation code so the browser sees a clean
        # rejection rather than a mid-session drop.
        await websocket.close(code=1008)
        return
    await websocket.accept()
    register_ws_subscriber(job_id, websocket)
    try:
        while True:
            # Keep connection alive; client can send pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        unregister_ws_subscriber(job_id, websocket)
