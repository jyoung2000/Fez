from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.pipeline import register_ws_subscriber, unregister_ws_subscriber

router = APIRouter()


@router.websocket("/ws/jobs/{job_id}")
async def websocket_job_progress(websocket: WebSocket, job_id: str):
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
