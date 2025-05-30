from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
from typing import Optional
from config import generate_image
import uvicorn

app = FastAPI()

# Concurrency management
MAX_CONCURRENT_REQUESTS = 20
current_requests = 0
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class GenerateRequest(BaseModel):
    prompt: str

class UpdateConcurrencyRequest(BaseModel):
    max_requests: int = Field(gt=0, le=100, description="New maximum number of concurrent requests (1-100)")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.put("/config")
async def update_concurrency(request: UpdateConcurrencyRequest):
    global MAX_CONCURRENT_REQUESTS, request_semaphore
    
    # Update the semaphore with new value
    old_semaphore = request_semaphore
    request_semaphore = asyncio.Semaphore(request.max_requests)
    MAX_CONCURRENT_REQUESTS = request.max_requests
    
    return {
        "status": "success",
        "message": f"Concurrency limit updated from {old_semaphore._value} to {request.max_requests}",
        "new_limit": request.max_requests
    }

@app.post("/generate")
async def generate(request: GenerateRequest):
    global current_requests
    
    if current_requests >= MAX_CONCURRENT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Server is currently at maximum capacity. Please try again later."
        )
    
    try:
        current_requests += 1
        async with request_semaphore:
            image_bytes, image_base64 = generate_image(request.prompt)
            return {
                "status": "success",
                "image_base64": image_base64,
                "image_bytes": image_bytes
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating image: {str(e)}"
        )
    finally:
        current_requests -= 1

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
