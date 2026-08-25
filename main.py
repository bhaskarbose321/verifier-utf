from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
import os
from verifier import verify_bundle

app = FastAPI(title="Model Bundle Verifier")


@app.get("/health")
def health():
    """Health check endpoint for Render."""
    return {"status": "healthy"}


@app.post("/verify-bundle")
async def verify_bundle_endpoint(request: Request):
    """Verify a model bundle according to the specification."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"error": "INVALID_INPUT"}, status_code=400)
    
    if not isinstance(body, dict):
        return JSONResponse(content={"error": "INVALID_INPUT"}, status_code=400)
    
    policy = body.get("policy")
    files = body.get("files")
    
    if not isinstance(policy, dict) or not isinstance(files, dict):
        return JSONResponse(content={"error": "INVALID_INPUT"}, status_code=400)
    
    result = verify_bundle(policy, files)
    
    if "error" in result:
        return JSONResponse(content=result, status_code=400)
    
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
