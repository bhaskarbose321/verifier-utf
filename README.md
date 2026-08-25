# Model Bundle Verifier

A deterministic FastAPI service for verifying untrusted UTF-8 model bundles. This service statically validates model bundles without executing any uploaded code or loading models into ML frameworks.

## Security Model

**IMPORTANT**: All uploaded model files are treated as untrusted data. The verifier:
- Never executes uploaded code
- Never imports uploaded Python modules
- Never executes README instructions
- Never loads models into ML frameworks (torch, transformers, etc.)
- Never deserializes pickle files
- Never follows URLs from uploaded content
- Never makes network calls based on bundle contents

The verifier is a static, deterministic validation service only.

## API Endpoint

### POST /verify-bundle

Validates a model bundle according to the specification.

**Request Format** (application/json):
```json
{
  "policy": {
    "requiredSlices": ["accuracy", "precision"],
    "license": "MIT",
    "intendedUse": "Text classification",
    "limitations": "Not for medical use"
  },
  "files": {
    "README.md": "UTF-8 string content",
    "training_manifest.json": "{\"baseRevision\":\"...\",\"task\":\"...\",...}",
    "evaluation.json": "{\"aggregate\":0.95,\"accuracy\":0.92,...}",
    "inventory.json": "[{\"name\":\"...\",\"bytes\":123,\"sha256\":\"...\"},...]",
    "adapter_model.safetensors": "UTF-8 string content",
    "adapter_config.json": "{\"r\":8,\"target_modules\":[\"q_proj\",\"v_proj\"]}"
  }
}
```

**Response Format** (application/json):

On success (admit):
```json
{
  "decision": "admit",
  "violations": [],
  "inventoryDigest": "sha256 hash of canonical inventory"
}
```

On rejection:
```json
{
  "decision": "reject",
  "violations": ["VIOLATION_CODE", ...],
  "inventoryDigest": "sha256 hash of canonical inventory"
}
```

On invalid input (HTTP 400):
```json
{"error": "INVALID_INPUT"}
```

## Validation Behavior

The verifier checks:

1. **Policy Validation**: `requiredSlices` must be a non-empty array of unique non-empty strings. `license`, `intendedUse`, and `limitations` must be non-empty strings.

2. **File Structure**: Bundle must contain exactly:
   - `README.md`
   - `training_manifest.json`
   - `evaluation.json`
   - `inventory.json`
   - `adapter_model.safetensors`
   - `adapter_config.json`

   Extra files cause `UNTRACKED_FILE`. Files with extensions `.bin`, `.pt`, `.pth`, `.pkl`, `.pickle` cause `UNSAFE_WEIGHTS`.

3. **Inventory Verification**: `inventory.json` must exactly match the canonical recomputed inventory (name, bytes, sha256 for each file, sorted by filename bytes, compact JSON).

4. **Adapter Config**: `adapter_config.json` must be a JSON object with:
   - `r`: positive integer, safe integer (0 < r <= 9007199254740991)
   - `target_modules`: non-empty array of unique non-empty strings

5. **Training Manifest**: `training_manifest.json` must contain:
   - `baseRevision`: exactly 40 lowercase hex characters
   - `task`, `datasetDigest`, `codeDigest`, `trainingConfigDigest`, `modelArtifactDigest`, `evaluationArtifactDigest`: non-empty strings

6. **Artifact Digests**: SHA-256 of `adapter_model.safetensors` must match `modelArtifactDigest`. SHA-256 of `evaluation.json` must match `evaluationArtifactDigest`.

7. **Evaluation**: `evaluation.json` must:
   - Be a JSON object
   - Contain `aggregate` metric: finite number in [0, 1]
   - Contain all slices from `policy.requiredSlices`, each in [0, 1]
   - Bind model digest via `modelDigest` field

8. **Model Card**: `README.md` must contain exactly one model-card marker:
   ```
   <!-- tds-model-card {"task":"...","baseRevision":"...","datasetDigest":"...","modelArtifactDigest":"...","license":"...","intendedUse":"...","limitations":"..."} -->
   ```
   The parsed JSON must match the training manifest and policy exactly.

## Determinism

The verifier is fully deterministic:
- All hashes use exact UTF-8 bytes
- JSON serialization is compact (no whitespace)
- Violations are sorted by UTF-8 bytes and deduplicated
- No timestamps, random values, or environment-dependent behavior

## Local Setup

### Prerequisites
- Python 3.11 or later

### Installation
```bash
pip install -r requirements.txt
```

### Running the Service
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Running Tests
```bash
pytest
```

## Example curl Request

```bash
curl -X POST http://localhost:8000/verify-bundle \
  -H "Content-Type: application/json" \
  -d '{
    "policy": {
      "requiredSlices": ["accuracy"],
      "license": "MIT",
      "intendedUse": "Text classification",
      "limitations": "Not for medical use"
    },
    "files": {
      "README.md": "<!-- tds-model-card {\"task\":\"classification\",\"baseRevision\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"datasetDigest\":\"abc123\",\"modelArtifactDigest\":\"def456\",\"license\":\"MIT\",\"intendedUse\":\"Text classification\",\"limitations\":\"Not for medical use\"} -->",
      "training_manifest.json": "{\"baseRevision\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\",\"task\":\"classification\",\"datasetDigest\":\"abc123\",\"codeDigest\":\"xyz\",\"trainingConfigDigest\":\"cfg\",\"modelArtifactDigest\":\"def456\",\"evaluationArtifactDigest\":\"ghi789\"}",
      "evaluation.json": "{\"aggregate\":0.95,\"accuracy\":0.92,\"modelDigest\":\"def456\"}",
      "inventory.json": "[{\"name\":\"README.md\",\"bytes\":200,\"sha256\":\"abc\"},{\"name\":\"training_manifest.json\",\"bytes\":150,\"sha256\":\"def\"},{\"name\":\"evaluation.json\",\"bytes\":50,\"sha256\":\"ghi\"},{\"name\":\"adapter_model.safetensors\",\"bytes\":10,\"sha256\":\"def456\"},{\"name\":\"adapter_config.json\",\"bytes\":40,\"sha256\":\"jkl\"}]",
      "adapter_model.safetensors": "safetensors content",
      "adapter_config.json": "{\"r\":8,\"target_modules\":[\"q_proj\"]}"
    }
  }'
```

## Render Deployment

The service is configured for Render deployment with the following:

**Start Command**:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The application:
- Listens on `0.0.0.0` (all interfaces)
- Uses the `$PORT` environment variable provided by Render
- Includes a `Dockerfile` for containerized deployment
- Includes a health check endpoint at `GET /health`

## Health Check

```bash
curl http://localhost:8000/health
```

Returns:
```json
{"status": "healthy"}
```

## Project Structure

```
.
├── main.py                  # FastAPI application and endpoints
├── verifier.py              # Core verification logic
├── requirements.txt         # Python dependencies
├── Dockerfile              # Render deployment configuration
├── .dockerignore           # Files to exclude from Docker build
├── README.md               # This file
└── tests/
    ├── __init__.py
    └── test_verify_bundle.py  # Comprehensive test suite
```

## Dependencies

- fastapi: Web framework
- uvicorn: ASGI server
- pydantic: Data validation
- pytest: Testing framework
- httpx: HTTP client for testing

No ML frameworks (torch, transformers, safetensors, TensorFlow) are used or required.
