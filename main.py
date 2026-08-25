from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import os
from verifier import verify_bundle

app = FastAPI(title="Model Bundle Verifier")


class VerifyRequest(BaseModel):
    policy: Optional[Dict[str, Any]] = None
    files: Optional[Dict[str, str]] = None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors and return INVALID_INPUT."""
    return JSONResponse(content={"error": "INVALID_INPUT"}, status_code=400)


@app.get("/health")
def health():
    """Health check endpoint for Render."""
    return {"status": "healthy"}


@app.post("/verify-bundle")
def verify_bundle_endpoint(request: VerifyRequest):
    """Verify a model bundle according to the specification."""
    if request.policy is None or request.files is None:
        return JSONResponse(content={"error": "INVALID_INPUT"}, status_code=400)
    
    if not isinstance(request.policy, dict) or not isinstance(request.files, dict):
        return JSONResponse(content={"error": "INVALID_INPUT"}, status_code=400)
    
    result = verify_bundle(request.policy, request.files)
    
    if "error" in result:
        return JSONResponse(content=result, status_code=400)
    
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
