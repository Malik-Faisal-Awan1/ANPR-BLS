import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI()

# Allow CORS for local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
async def status():
    return {"status": "running (stub)", "watcher_directory": "C:/stub", "output_directory": "C:/stub", "failed_directory": "C:/stub"}

@app.get("/api/cameras")
async def cameras():
    return {"cameras": [{"id": "cam_01", "name": "Entry Barrier", "status": "online", "location": "Main Gate", "watchDirectory": "C:/stub"}]}

@app.get("/api/history")
async def history():
    return {"data": [], "total": 0, "page": 1, "limit": 50}

async def event_generator():
    while True:
        await asyncio.sleep(5)
        yield 'data: {"id": "test", "cameraId": "cam_01", "plateNumber": "STUB123", "success": true, "confidence": 0.99, "processingMs": 100, "imagePath": ""}\n\n'

@app.get("/api/events")
async def events():
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5001)
