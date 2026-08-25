import hashlib
import json
from typing import Any, Dict, List, Set, Tuple


def sha256_digest(data: str) -> str:
    """Compute SHA-256 digest of UTF-8 encoded string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_policy(policy: Any) -> List[str]:
    """Validate policy object and return violations."""
    violations = []
    
    if not isinstance(policy, dict):
        return ["INVALID_POLICY"]
    
    # Validate requiredSlices
    required_slices = policy.get("requiredSlices")
    if not isinstance(required_slices, list) or len(required_slices) == 0:
        violations.append("INVALID_POLICY")
    else:
        seen_slices: Set[str] = set()
        for slice_name in required_slices:
            if not isinstance(slice_name, str) or len(slice_name) == 0:
                violations.append("INVALID_POLICY")
                break
            if slice_name in seen_slices:
                violations.append("INVALID_POLICY")
                break
            seen_slices.add(slice_name)
    
    # Validate required string fields
    for field in ["license", "intendedUse", "limitations"]:
        value = policy.get(field)
        if not isinstance(value, str) or len(value) == 0:
            violations.append("INVALID_POLICY")
    
    return violations


def validate_files_and_structure(files: Dict[str, str]) -> Tuple[List[str], Dict[str, Any]]:
    """Validate file structure and return violations and file info."""
    violations = []
    
    required_files = {
        "README.md",
        "training_manifest.json",
        "evaluation.json",
        "inventory.json",
        "adapter_model.safetensors",
        "adapter_config.json",
    }
    
    unsafe_extensions = {".bin", ".pt", ".pth", ".pkl", ".pickle"}
    
    actual_files = set(files.keys())
    
    # Check for missing required files
    for required in required_files:
        if required not in actual_files:
            violations.append(f"MISSING_FILE:{required}")
    
    # Check for extra files
    for filename in actual_files:
        if filename not in required_files:
            violations.append("UNTRACKED_FILE")
        
        # Check for unsafe extensions
        for ext in unsafe_extensions:
            if filename.endswith(ext):
                violations.append("UNSAFE_WEIGHTS")
                break
    
    return violations, {"required": required_files, "actual": actual_files}


def compute_inventory(files: Dict[str, str]) -> str:
    """Compute canonical inventory JSON string."""
    entries = []
    
    for filename in sorted(files.keys(), key=lambda x: x.encode("utf-8")):
        if filename == "inventory.json":
            continue
        
        content = files[filename]
        entry = {
            "name": filename,
            "bytes": len(content.encode("utf-8")),
            "sha256": sha256_digest(content),
        }
        entries.append(entry)
    
    return json.dumps(entries, ensure_ascii=False, separators=(",", ":"))


def validate_inventory(files: Dict[str, str]) -> Tuple[List[str], str]:
    """Validate inventory.json and return violations and recomputed digest."""
    violations = []
    
    if "inventory.json" not in files:
        violations.append("MISSING_FILE:inventory.json")
        return violations, sha256_digest(compute_inventory(files))
    
    try:
        supplied_inventory = json.loads(files["inventory.json"])
    except json.JSONDecodeError:
        violations.append("INVALID_JSON:inventory.json")
        return violations, sha256_digest(compute_inventory(files))
    
    if not isinstance(supplied_inventory, list):
        violations.append("INVALID_JSON:inventory.json")
        return violations, sha256_digest(compute_inventory(files))
    
    # Compute expected inventory
    expected_entries = []
    for filename in sorted(files.keys(), key=lambda x: x.encode("utf-8")):
        if filename == "inventory.json":
            continue
        
        content = files[filename]
        expected_entries.append({
            "name": filename,
            "bytes": len(content.encode("utf-8")),
            "sha256": sha256_digest(content),
        })
    
    expected_inventory = json.dumps(expected_entries, ensure_ascii=False, separators=(",", ":"))
    
    # Compare
    try:
        supplied_str = json.dumps(supplied_inventory, ensure_ascii=False, separators=(",", ":"))
        if supplied_str != expected_inventory:
            violations.append("INVENTORY_MISMATCH")
    except (TypeError, ValueError):
        violations.append("INVENTORY_MISMATCH")
    
    return violations, sha256_digest(expected_inventory)


def validate_adapter_config(files: Dict[str, str]) -> List[str]:
    """Validate adapter_config.json."""
    violations = []
    
    if "adapter_config.json" not in files:
        return violations  # Already reported as MISSING_FILE
    
    try:
        config = json.loads(files["adapter_config.json"])
    except json.JSONDecodeError:
        return ["INVALID_JSON:adapter_config.json"]
    
    if not isinstance(config, dict):
        return ["INVALID_ADAPTER_CONFIG"]
    
    # Validate r
    r = config.get("r")
    if not isinstance(r, int) or isinstance(r, bool):
        return ["INVALID_ADAPTER_CONFIG"]
    if r <= 0 or r > 9007199254740991:
        return ["INVALID_ADAPTER_CONFIG"]
    
    # Validate target_modules
    target_modules = config.get("target_modules")
    if not isinstance(target_modules, list) or len(target_modules) == 0:
        return ["INVALID_ADAPTER_CONFIG"]
    
    seen_modules: Set[str] = set()
    for module in target_modules:
        if not isinstance(module, str) or len(module) == 0:
            return ["INVALID_ADAPTER_CONFIG"]
        if module in seen_modules:
            return ["INVALID_ADAPTER_CONFIG"]
        seen_modules.add(module)
    
    return violations


def validate_training_manifest(files: Dict[str, str]) -> Tuple[List[str], Dict[str, Any]]:
    """Validate training_manifest.json and return violations and parsed manifest."""
    violations = []
    
    if "training_manifest.json" not in files:
        return violations, {}
    
    try:
        manifest = json.loads(files["training_manifest.json"])
    except json.JSONDecodeError:
        return ["INVALID_JSON:training_manifest.json"], {}
    
    if not isinstance(manifest, dict):
        return ["INVALID_TRAINING_MANIFEST"], {}
    
    required_fields = [
        "baseRevision",
        "task",
        "datasetDigest",
        "codeDigest",
        "trainingConfigDigest",
        "modelArtifactDigest",
        "evaluationArtifactDigest",
    ]
    
    for field in required_fields:
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) == 0:
            violations.append(f"MISSING_MANIFEST_FIELD:{field}")
    
    # Validate baseRevision format
    base_revision = manifest.get("baseRevision")
    if isinstance(base_revision, str) and len(base_revision) == 40:
        import re
        if not re.match(r"^[0-9a-f]{40}$", base_revision):
            violations.append("MUTABLE_BASE_REVISION")
    elif isinstance(base_revision, str):
        violations.append("MUTABLE_BASE_REVISION")
    
    return violations, manifest


def validate_artifact_digests(files: Dict[str, str], manifest: Dict[str, Any]) -> List[str]:
    """Validate model and evaluation artifact digests."""
    violations = []
    
    # Model artifact digest
    if "adapter_model.safetensors" in files:
        actual_model_digest = sha256_digest(files["adapter_model.safetensors"])
        expected_model_digest = manifest.get("modelArtifactDigest")
        if isinstance(expected_model_digest, str) and actual_model_digest != expected_model_digest:
            violations.append("MODEL_ARTIFACT_MISMATCH")
    
    # Evaluation artifact digest
    if "evaluation.json" in files:
        actual_eval_digest = sha256_digest(files["evaluation.json"])
        expected_eval_digest = manifest.get("evaluationArtifactDigest")
        if isinstance(expected_eval_digest, str) and actual_eval_digest != expected_eval_digest:
            violations.append("EVALUATION_ARTIFACT_MISMATCH")
    
    return violations


def validate_evaluation(files: Dict[str, str], manifest: Dict[str, Any], policy: Dict[str, Any]) -> List[str]:
    """Validate evaluation.json."""
    violations = []
    
    if "evaluation.json" not in files:
        return violations
    
    try:
        evaluation = json.loads(files["evaluation.json"])
    except json.JSONDecodeError:
        return ["INVALID_JSON:evaluation.json"]
    
    if not isinstance(evaluation, dict):
        return ["INVALID_EVALUATION"]
    
    # Validate model digest binding
    if "adapter_model.safetensors" in files:
        actual_model_digest = sha256_digest(files["adapter_model.safetensors"])
        model_digest_field = evaluation.get("modelDigest")
        if isinstance(model_digest_field, str) and model_digest_field != actual_model_digest:
            violations.append("MODEL_ARTIFACT_MISMATCH")
    
    # Validate aggregate metric
    aggregate = evaluation.get("aggregate")
    if aggregate is None:
        violations.append("INVALID_AGGREGATE")
    else:
        try:
            agg_float = float(aggregate)
            if not (0 <= agg_float <= 1):
                violations.append("INVALID_AGGREGATE")
        except (TypeError, ValueError):
            violations.append("INVALID_AGGREGATE")
    
    # Validate required slices
    required_slices = policy.get("requiredSlices", [])
    if isinstance(required_slices, list):
        for slice_name in required_slices:
            if isinstance(slice_name, str) and slice_name not in evaluation:
                violations.append(f"MISSING_SLICE:{slice_name}")
            elif isinstance(slice_name, str):
                slice_value = evaluation.get(slice_name)
                try:
                    slice_float = float(slice_value)
                    if not (0 <= slice_float <= 1):
                        violations.append(f"SLICE_RANGE:{slice_name}")
                except (TypeError, ValueError):
                    violations.append(f"SLICE_RANGE:{slice_name}")
    
    return violations


def parse_model_card(readme_content: str) -> Tuple[List[str], Dict[str, Any]]:
    """Parse model card from README and return violations and parsed payload."""
    violations = []
    
    marker_prefix = "<!-- tds-model-card "
    marker_suffix = "-->"
    
    # Find all markers
    markers = []
    start_idx = 0
    while True:
        prefix_pos = readme_content.find(marker_prefix, start_idx)
        if prefix_pos == -1:
            break
        
        suffix_pos = readme_content.find(marker_suffix, prefix_pos + len(marker_prefix))
        if suffix_pos == -1:
            break
        
        # Extract payload between prefix and suffix
        payload_start = prefix_pos + len(marker_prefix)
        payload = readme_content[payload_start:suffix_pos].strip()
        markers.append(payload)
        start_idx = suffix_pos + len(marker_suffix)
    
    if len(markers) == 0:
        return ["MODEL_CARD_COUNT", "MISSING_MODEL_CARD"], {}
    
    if len(markers) > 1:
        return ["MODEL_CARD_COUNT"], {}
    
    # Parse the single marker
    try:
        payload = json.loads(markers[0])
    except json.JSONDecodeError:
        return ["INVALID_MODEL_CARD"], {}
    
    if not isinstance(payload, dict):
        return ["INVALID_MODEL_CARD"], {}
    
    return violations, payload


def validate_model_card(
    payload: Dict[str, Any],
    manifest: Dict[str, Any],
    policy: Dict[str, Any]
) -> List[str]:
    """Validate model card payload against manifest and policy."""
    violations = []
    
    required_fields = [
        "task",
        "baseRevision",
        "datasetDigest",
        "modelArtifactDigest",
        "license",
        "intendedUse",
        "limitations",
    ]
    
    for field in required_fields:
        card_value = payload.get(field)
        
        if field in ["license", "intendedUse", "limitations"]:
            policy_value = policy.get(field)
            if card_value != policy_value:
                violations.append("MODEL_CARD_MISMATCH")
                break
        elif field in ["task", "datasetDigest", "modelArtifactDigest"]:
            manifest_value = manifest.get(field)
            if card_value != manifest_value:
                violations.append("MODEL_CARD_MISMATCH")
                break
        elif field == "baseRevision":
            manifest_value = manifest.get(field)
            if card_value != manifest_value:
                violations.append("MODEL_CARD_MISMATCH")
                break
    
    return violations


def verify_bundle(policy: Any, files: Any) -> Dict[str, Any]:
    """Main verification function."""
    # Validate top-level input
    if not isinstance(policy, dict) or not isinstance(files, dict):
        return {"error": "INVALID_INPUT"}
    
    violations = []
    
    # Validate policy
    violations.extend(validate_policy(policy))
    
    # Validate file structure
    file_violations, _ = validate_files_and_structure(files)
    violations.extend(file_violations)
    
    # Validate inventory
    inventory_violations, inventory_digest = validate_inventory(files)
    violations.extend(inventory_violations)
    
    # Validate adapter config
    violations.extend(validate_adapter_config(files))
    
    # Validate training manifest
    manifest_violations, manifest = validate_training_manifest(files)
    violations.extend(manifest_violations)
    
    # Validate artifact digests
    violations.extend(validate_artifact_digests(files, manifest))
    
    # Validate evaluation
    violations.extend(validate_evaluation(files, manifest, policy))
    
    # Validate model card
    if "README.md" in files:
        card_violations, card_payload = parse_model_card(files["README.md"])
        violations.extend(card_violations)
        
        if len(card_violations) == 0 and isinstance(card_payload, dict):
            violations.extend(validate_model_card(card_payload, manifest, policy))
    
    # Sort and deduplicate violations
    violations = sorted(set(violations), key=lambda x: x.encode("utf-8"))
    
    decision = "admit" if len(violations) == 0 else "reject"
    
    return {
        "decision": decision,
        "violations": violations,
        "inventoryDigest": inventory_digest,
    }
