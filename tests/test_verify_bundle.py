import json
import pytest
from fastapi.testclient import TestClient
from main import app
from verifier import verify_bundle, sha256_digest

client = TestClient(app)


def create_valid_bundle():
    """Create a valid bundle for testing."""
    policy = {
        "requiredSlices": ["accuracy", "precision"],
        "license": "MIT",
        "intendedUse": "Text classification",
        "limitations": "Not for medical use"
    }
    
    base_revision = "a" * 40  # Valid 40-char hex
    
    # Compute digests
    adapter_content = "fake safetensors content"
    eval_content = '{"aggregate":0.95,"accuracy":0.92,"precision":0.88,"modelDigest":"placeholder"}'
    
    adapter_digest = sha256_digest(adapter_content)
    eval_digest = sha256_digest(eval_content)
    
    # Update eval content with actual model digest
    eval_content = f'{{"aggregate":0.95,"accuracy":0.92,"precision":0.88,"modelDigest":"{adapter_digest}"}}'
    eval_digest = sha256_digest(eval_content)
    
    files = {
        "README.md": f'Some prose.\n\n<!-- tds-model-card {{"task":"classification","baseRevision":"{base_revision}","datasetDigest":"abc123","modelArtifactDigest":"{adapter_digest}","license":"MIT","intendedUse":"Text classification","limitations":"Not for medical use"}} -->\n\nMore prose.',
        "training_manifest.json": json.dumps({
            "baseRevision": base_revision,
            "task": "classification",
            "datasetDigest": "abc123",
            "codeDigest": "def123",
            "trainingConfigDigest": "cfg123",
            "modelArtifactDigest": adapter_digest,
            "evaluationArtifactDigest": eval_digest
        }),
        "evaluation.json": eval_content,
        "adapter_model.safetensors": adapter_content,
        "adapter_config.json": json.dumps({
            "r": 8,
            "target_modules": ["q_proj", "v_proj"]
        })
    }
    
    # Compute inventory
    inventory_entries = []
    for filename in sorted(files.keys(), key=lambda x: x.encode("utf-8")):
        content = files[filename]
        inventory_entries.append({
            "name": filename,
            "bytes": len(content.encode("utf-8")),
            "sha256": sha256_digest(content)
        })
    
    files["inventory.json"] = json.dumps(inventory_entries, ensure_ascii=False, separators=(",", ":"))
    
    return policy, files


def test_completely_valid_bundle():
    """Test 1: Completely valid bundle → admit"""
    policy, files = create_valid_bundle()
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "admit"
    assert result["violations"] == []
    assert "inventoryDigest" in result
    assert len(result["inventoryDigest"]) == 64


def test_missing_policy():
    """Test 2: Missing policy → HTTP 400 exactly {"error":"INVALID_INPUT"}"""
    response = client.post("/verify-bundle", json={"files": {}})
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_files_not_object():
    """Test 3: files not object → HTTP 400 exactly {"error":"INVALID_INPUT"}"""
    response = client.post("/verify-bundle", json={"policy": {}, "files": "not an object"})
    assert response.status_code == 400
    assert response.json() == {"error": "INVALID_INPUT"}


def test_missing_required_file():
    """Test 4: Missing required file"""
    policy, files = create_valid_bundle()
    del files["README.md"]
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "MISSING_FILE:README.md" in result["violations"]


def test_extra_file():
    """Test 5: Extra file"""
    policy, files = create_valid_bundle()
    files["extra.txt"] = "extra content"
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "UNTRACKED_FILE" in result["violations"]


def test_unsafe_bin():
    """Test 6: Unsafe .bin"""
    policy, files = create_valid_bundle()
    files["model.bin"] = "binary content"
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "UNSAFE_WEIGHTS" in result["violations"]


def test_unsafe_pt():
    """Test 7: Unsafe .pt"""
    policy, files = create_valid_bundle()
    files["model.pt"] = "pytorch content"
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "UNSAFE_WEIGHTS" in result["violations"]


def test_unsafe_pth():
    """Test 8: Unsafe .pth"""
    policy, files = create_valid_bundle()
    files["model.pth"] = "pytorch content"
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "UNSAFE_WEIGHTS" in result["violations"]


def test_unsafe_pkl():
    """Test 9: Unsafe .pkl"""
    policy, files = create_valid_bundle()
    files["model.pkl"] = "pickle content"
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "UNSAFE_WEIGHTS" in result["violations"]


def test_unsafe_pickle():
    """Test 10: Unsafe .pickle"""
    policy, files = create_valid_bundle()
    files["model.pickle"] = "pickle content"
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "UNSAFE_WEIGHTS" in result["violations"]


def test_inventory_hash_mismatch():
    """Test 11: Inventory hash mismatch"""
    policy, files = create_valid_bundle()
    files["inventory.json"] = json.dumps([{"name": "wrong", "bytes": 0, "sha256": "0"}])
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVENTORY_MISMATCH" in result["violations"]


def test_inventory_ordering_mismatch():
    """Test 12: Inventory ordering mismatch"""
    policy, files = create_valid_bundle()
    # Reorder inventory entries
    entries = json.loads(files["inventory.json"])
    files["inventory.json"] = json.dumps(list(reversed(entries)), ensure_ascii=False, separators=(",", ":"))
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVENTORY_MISMATCH" in result["violations"]


def test_inventory_key_order_mismatch():
    """Test 13: Inventory key-order mismatch"""
    policy, files = create_valid_bundle()
    # Change key order
    entries = json.loads(files["inventory.json"])
    reordered = [{"sha256": e["sha256"], "bytes": e["bytes"], "name": e["name"]} for e in entries]
    files["inventory.json"] = json.dumps(reordered, ensure_ascii=False, separators=(",", ":"))
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVENTORY_MISMATCH" in result["violations"]


def test_inventory_extra_entry():
    """Test 14: Inventory extra entry"""
    policy, files = create_valid_bundle()
    entries = json.loads(files["inventory.json"])
    entries.append({"name": "extra.txt", "bytes": 100, "sha256": "abc123"})
    files["inventory.json"] = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVENTORY_MISMATCH" in result["violations"]


def test_inventory_missing_entry():
    """Test 15: Inventory missing entry"""
    policy, files = create_valid_bundle()
    entries = json.loads(files["inventory.json"])
    entries.pop(0)
    files["inventory.json"] = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVENTORY_MISMATCH" in result["violations"]


def test_invalid_inventory_json():
    """Test 16: Invalid inventory JSON"""
    policy, files = create_valid_bundle()
    files["inventory.json"] = "not valid json"
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_JSON:inventory.json" in result["violations"]


def test_invalid_adapter_config():
    """Test 17: Invalid adapter config"""
    policy, files = create_valid_bundle()
    files["adapter_config.json"] = "not valid json"
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_JSON:adapter_config.json" in result["violations"]


def test_invalid_r():
    """Test 18: Invalid r"""
    policy, files = create_valid_bundle()
    files["adapter_config.json"] = json.dumps({"r": -1, "target_modules": ["q_proj"]})
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_ADAPTER_CONFIG" in result["violations"]


def test_r_boolean():
    """Test 19: r boolean"""
    policy, files = create_valid_bundle()
    files["adapter_config.json"] = json.dumps({"r": True, "target_modules": ["q_proj"]})
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_ADAPTER_CONFIG" in result["violations"]


def test_empty_target_modules():
    """Test 20: Empty target_modules"""
    policy, files = create_valid_bundle()
    files["adapter_config.json"] = json.dumps({"r": 8, "target_modules": []})
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_ADAPTER_CONFIG" in result["violations"]


def test_duplicate_target_modules():
    """Test 21: Duplicate target_modules"""
    policy, files = create_valid_bundle()
    files["adapter_config.json"] = json.dumps({"r": 8, "target_modules": ["q_proj", "q_proj"]})
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_ADAPTER_CONFIG" in result["violations"]


def test_invalid_training_manifest():
    """Test 22: Invalid training manifest"""
    policy, files = create_valid_bundle()
    files["training_manifest.json"] = "not valid json"
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_JSON:training_manifest.json" in result["violations"]


def test_invalid_base_revision():
    """Test 23: Invalid base revision"""
    policy, files = create_valid_bundle()
    manifest = json.loads(files["training_manifest.json"])
    manifest["baseRevision"] = "not40chars"
    files["training_manifest.json"] = json.dumps(manifest)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "MUTABLE_BASE_REVISION" in result["violations"]


def test_missing_manifest_field():
    """Test 24: Missing manifest field"""
    policy, files = create_valid_bundle()
    manifest = json.loads(files["training_manifest.json"])
    del manifest["task"]
    files["training_manifest.json"] = json.dumps(manifest)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "MISSING_MANIFEST_FIELD:task" in result["violations"]


def test_model_artifact_mismatch():
    """Test 25: Model artifact mismatch"""
    policy, files = create_valid_bundle()
    manifest = json.loads(files["training_manifest.json"])
    manifest["modelArtifactDigest"] = "wrongdigest"
    files["training_manifest.json"] = json.dumps(manifest)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "MODEL_ARTIFACT_MISMATCH" in result["violations"]


def test_evaluation_artifact_mismatch():
    """Test 26: Evaluation artifact mismatch"""
    policy, files = create_valid_bundle()
    manifest = json.loads(files["training_manifest.json"])
    manifest["evaluationArtifactDigest"] = "wrongdigest"
    files["training_manifest.json"] = json.dumps(manifest)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "EVALUATION_ARTIFACT_MISMATCH" in result["violations"]


def test_invalid_evaluation_json():
    """Test 27: Invalid evaluation JSON"""
    policy, files = create_valid_bundle()
    files["evaluation.json"] = "not valid json"
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_JSON:evaluation.json" in result["violations"]


def test_invalid_aggregate():
    """Test 28: Invalid aggregate"""
    policy, files = create_valid_bundle()
    eval_obj = json.loads(files["evaluation.json"])
    eval_obj["aggregate"] = "not a number"
    files["evaluation.json"] = json.dumps(eval_obj)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_AGGREGATE" in result["violations"]


def test_aggregate_nan():
    """Test 29: Aggregate NaN"""
    policy, files = create_valid_bundle()
    eval_obj = json.loads(files["evaluation.json"])
    eval_obj["aggregate"] = float("nan")
    files["evaluation.json"] = json.dumps(eval_obj)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_AGGREGATE" in result["violations"]


def test_aggregate_infinity():
    """Test 30: Aggregate Infinity"""
    policy, files = create_valid_bundle()
    eval_obj = json.loads(files["evaluation.json"])
    eval_obj["aggregate"] = float("inf")
    files["evaluation.json"] = json.dumps(eval_obj)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_AGGREGATE" in result["violations"]


def test_aggregate_below_zero():
    """Test 31: Aggregate below zero"""
    policy, files = create_valid_bundle()
    eval_obj = json.loads(files["evaluation.json"])
    eval_obj["aggregate"] = -0.1
    files["evaluation.json"] = json.dumps(eval_obj)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_AGGREGATE" in result["violations"]


def test_aggregate_above_one():
    """Test 32: Aggregate above one"""
    policy, files = create_valid_bundle()
    eval_obj = json.loads(files["evaluation.json"])
    eval_obj["aggregate"] = 1.1
    files["evaluation.json"] = json.dumps(eval_obj)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_AGGREGATE" in result["violations"]


def test_missing_required_slice():
    """Test 33: Missing required slice"""
    policy, files = create_valid_bundle()
    eval_obj = json.loads(files["evaluation.json"])
    del eval_obj["accuracy"]
    files["evaluation.json"] = json.dumps(eval_obj)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "MISSING_SLICE:accuracy" in result["violations"]


def test_required_slice_below_zero():
    """Test 34: Required slice below zero"""
    policy, files = create_valid_bundle()
    eval_obj = json.loads(files["evaluation.json"])
    eval_obj["accuracy"] = -0.1
    files["evaluation.json"] = json.dumps(eval_obj)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "SLICE_RANGE:accuracy" in result["violations"]


def test_required_slice_above_one():
    """Test 35: Required slice above one"""
    policy, files = create_valid_bundle()
    eval_obj = json.loads(files["evaluation.json"])
    eval_obj["accuracy"] = 1.1
    files["evaluation.json"] = json.dumps(eval_obj)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "SLICE_RANGE:accuracy" in result["violations"]


def test_required_slice_nan():
    """Test 36: Required slice NaN"""
    policy, files = create_valid_bundle()
    eval_obj = json.loads(files["evaluation.json"])
    eval_obj["accuracy"] = float("nan")
    files["evaluation.json"] = json.dumps(eval_obj)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "SLICE_RANGE:accuracy" in result["violations"]


def test_required_slice_infinity():
    """Test 37: Required slice Infinity"""
    policy, files = create_valid_bundle()
    eval_obj = json.loads(files["evaluation.json"])
    eval_obj["accuracy"] = float("inf")
    files["evaluation.json"] = json.dumps(eval_obj)
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "SLICE_RANGE:accuracy" in result["violations"]


def test_zero_model_card_markers():
    """Test 38: Zero model-card markers"""
    policy, files = create_valid_bundle()
    files["README.md"] = "No markers here"
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "MODEL_CARD_COUNT" in result["violations"]
    assert "MISSING_MODEL_CARD" in result["violations"]


def test_two_model_card_markers():
    """Test 39: Two model-card markers → only MODEL_CARD_COUNT"""
    policy, files = create_valid_bundle()
    files["README.md"] = '<!-- tds-model-card {"task":"a"} -->\n<!-- tds-model-card {"task":"b"} -->'
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "MODEL_CARD_COUNT" in result["violations"]
    # Should NOT have INVALID_MODEL_CARD or MODEL_CARD_MISMATCH
    assert "INVALID_MODEL_CARD" not in result["violations"]
    assert "MODEL_CARD_MISMATCH" not in result["violations"]


def test_malformed_model_card_json():
    """Test 40: Malformed model-card JSON"""
    policy, files = create_valid_bundle()
    files["README.md"] = '<!-- tds-model-card {not valid json} -->'
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_MODEL_CARD" in result["violations"]


def test_model_card_json_with_braces():
    """Test 41: Model-card JSON string containing {still text}"""
    policy, files = create_valid_bundle()
    base_revision = "a" * 40
    adapter_digest = sha256_digest(files["adapter_model.safetensors"])
    files["README.md"] = f'Some prose.\n\n<!-- tds-model-card {{"task":"classification","limitations":"Text containing {{still text}} here","baseRevision":"{base_revision}","datasetDigest":"abc123","modelArtifactDigest":"{adapter_digest}","license":"MIT","intendedUse":"Text classification"}} -->\n\nMore prose.'
    
    # Update manifest to match
    manifest = json.loads(files["training_manifest.json"])
    manifest["baseRevision"] = base_revision
    manifest["datasetDigest"] = "abc123"
    manifest["modelArtifactDigest"] = adapter_digest
    files["training_manifest.json"] = json.dumps(manifest)
    
    result = verify_bundle(policy, files)
    
    # Should parse correctly, but will have MODEL_CARD_MISMATCH due to limitations field
    assert result["decision"] == "reject"
    assert "INVALID_MODEL_CARD" not in result["violations"]


def test_model_card_non_object_payload():
    """Test 42: Model-card non-object payload"""
    policy, files = create_valid_bundle()
    files["README.md"] = '<!-- tds-model-card ["array", "not", "object"] -->'
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_MODEL_CARD" in result["violations"]


def test_model_card_field_mismatch():
    """Test 43: Model-card field mismatch"""
    policy, files = create_valid_bundle()
    # Change task in model card
    files["README.md"] = files["README.md"].replace('"task":"classification"', '"task":"different"')
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "MODEL_CARD_MISMATCH" in result["violations"]


def test_extra_model_card_properties_allowed():
    """Test 44: Extra model-card properties allowed"""
    policy, files = create_valid_bundle()
    # Add extra property to model card
    files["README.md"] = files["README.md"].replace('}', ',"extraProperty":"allowed"}')
    # Recompute inventory since README changed
    inventory_entries = []
    for filename in sorted(files.keys(), key=lambda x: x.encode("utf-8")):
        if filename == "inventory.json":
            continue
        content = files[filename]
        inventory_entries.append({
            "name": filename,
            "bytes": len(content.encode("utf-8")),
            "sha256": sha256_digest(content)
        })
    files["inventory.json"] = json.dumps(inventory_entries, ensure_ascii=False, separators=(",", ":"))
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "admit"
    assert result["violations"] == []


def test_extra_evaluation_properties_allowed():
    """Test 45: Extra evaluation properties allowed"""
    policy, files = create_valid_bundle()
    eval_obj = json.loads(files["evaluation.json"])
    eval_obj["extraMetric"] = 0.5
    files["evaluation.json"] = json.dumps(eval_obj)
    # Update manifest with new evaluation digest
    manifest = json.loads(files["training_manifest.json"])
    manifest["evaluationArtifactDigest"] = sha256_digest(files["evaluation.json"])
    files["training_manifest.json"] = json.dumps(manifest)
    # Recompute inventory since files changed
    inventory_entries = []
    for filename in sorted(files.keys(), key=lambda x: x.encode("utf-8")):
        if filename == "inventory.json":
            continue
        content = files[filename]
        inventory_entries.append({
            "name": filename,
            "bytes": len(content.encode("utf-8")),
            "sha256": sha256_digest(content)
        })
    files["inventory.json"] = json.dumps(inventory_entries, ensure_ascii=False, separators=(",", ":"))
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "admit"
    assert result["violations"] == []


def test_non_required_evaluation_slices_allowed():
    """Test 46: Non-required evaluation slices allowed"""
    policy, files = create_valid_bundle()
    eval_obj = json.loads(files["evaluation.json"])
    eval_obj["recall"] = 0.85
    files["evaluation.json"] = json.dumps(eval_obj)
    # Update manifest with new evaluation digest
    manifest = json.loads(files["training_manifest.json"])
    manifest["evaluationArtifactDigest"] = sha256_digest(files["evaluation.json"])
    files["training_manifest.json"] = json.dumps(manifest)
    # Recompute inventory since files changed
    inventory_entries = []
    for filename in sorted(files.keys(), key=lambda x: x.encode("utf-8")):
        if filename == "inventory.json":
            continue
        content = files[filename]
        inventory_entries.append({
            "name": filename,
            "bytes": len(content.encode("utf-8")),
            "sha256": sha256_digest(content)
        })
    files["inventory.json"] = json.dumps(inventory_entries, ensure_ascii=False, separators=(",", ":"))
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "admit"
    assert result["violations"] == []


def test_unicode_filenames():
    """Test 47: Unicode filenames"""
    policy, files = create_valid_bundle()
    # This should still fail due to UNTRACKED_FILE, but handle unicode correctly
    files["файл.txt"] = "unicode content"
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "UNTRACKED_FILE" in result["violations"]


def test_unicode_readme_content():
    """Test 48: Unicode README content"""
    policy, files = create_valid_bundle()
    base_revision = "a" * 40
    adapter_digest = sha256_digest(files["adapter_model.safetensors"])
    files["README.md"] = f'Unicode: 你好世界\n\n<!-- tds-model-card {{"task":"classification","baseRevision":"{base_revision}","datasetDigest":"abc123","modelArtifactDigest":"{adapter_digest}","license":"MIT","intendedUse":"Text classification","limitations":"Not for medical use"}} -->'
    
    manifest = json.loads(files["training_manifest.json"])
    manifest["baseRevision"] = base_revision
    manifest["datasetDigest"] = "abc123"
    manifest["modelArtifactDigest"] = adapter_digest
    files["training_manifest.json"] = json.dumps(manifest)
    
    # Recompute inventory since files changed
    inventory_entries = []
    for filename in sorted(files.keys(), key=lambda x: x.encode("utf-8")):
        if filename == "inventory.json":
            continue
        content = files[filename]
        inventory_entries.append({
            "name": filename,
            "bytes": len(content.encode("utf-8")),
            "sha256": sha256_digest(content)
        })
    files["inventory.json"] = json.dumps(inventory_entries, ensure_ascii=False, separators=(",", ":"))
    
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "admit"
    assert result["violations"] == []


def test_exact_utf8_byte_hashing():
    """Test 49: Exact UTF-8 byte hashing"""
    # Test that hashing uses exact UTF-8 bytes
    test_str = "Hello"
    digest1 = sha256_digest(test_str)
    digest2 = sha256_digest(test_str)
    assert digest1 == digest2
    
    # Different strings should have different hashes
    assert digest1 != sha256_digest("Hello ")


def test_deterministic_violation_ordering():
    """Test 50: Deterministic violation ordering"""
    policy, files = create_valid_bundle()
    # Create multiple violations
    del files["README.md"]
    files["extra.txt"] = "extra"
    files["model.bin"] = "binary"
    
    result = verify_bundle(policy, files)
    
    # Check that violations are sorted
    violations = result["violations"]
    assert violations == sorted(violations, key=lambda x: x.encode("utf-8"))


def test_deterministic_violation_deduplication():
    """Test 51: Deterministic violation deduplication"""
    policy, files = create_valid_bundle()
    # Create conditions that might produce duplicate violations
    del files["README.md"]
    del files["training_manifest.json"]
    
    result = verify_bundle(policy, files)
    
    # Check no duplicates
    violations = result["violations"]
    assert len(violations) == len(set(violations))


def test_health_endpoint():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_extra_adapter_config_properties_allowed():
    """Test that extra properties in adapter_config.json are allowed"""
    policy, files = create_valid_bundle()
    config = json.loads(files["adapter_config.json"])
    config["extraProperty"] = "allowed"
    files["adapter_config.json"] = json.dumps(config)
    # Recompute inventory since adapter_config.json changed
    inventory_entries = []
    for filename in sorted(files.keys(), key=lambda x: x.encode("utf-8")):
        if filename == "inventory.json":
            continue
        content = files[filename]
        inventory_entries.append({
            "name": filename,
            "bytes": len(content.encode("utf-8")),
            "sha256": sha256_digest(content)
        })
    files["inventory.json"] = json.dumps(inventory_entries, ensure_ascii=False, separators=(",", ":"))
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "admit"
    assert result["violations"] == []


def test_policy_empty_required_slices():
    """Test policy with empty requiredSlices"""
    policy = {
        "requiredSlices": [],
        "license": "MIT",
        "intendedUse": "Test",
        "limitations": "Test"
    }
    files = {}
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_POLICY" in result["violations"]


def test_policy_non_string_in_required_slices():
    """Test policy with non-string in requiredSlices"""
    policy = {
        "requiredSlices": [123],
        "license": "MIT",
        "intendedUse": "Test",
        "limitations": "Test"
    }
    files = {}
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_POLICY" in result["violations"]


def test_policy_duplicate_required_slices():
    """Test policy with duplicate requiredSlices"""
    policy = {
        "requiredSlices": ["accuracy", "accuracy"],
        "license": "MIT",
        "intendedUse": "Test",
        "limitations": "Test"
    }
    files = {}
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_POLICY" in result["violations"]


def test_policy_empty_string_field():
    """Test policy with empty string field"""
    policy = {
        "requiredSlices": ["accuracy"],
        "license": "",
        "intendedUse": "Test",
        "limitations": "Test"
    }
    files = {}
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_POLICY" in result["violations"]


def test_r_max_safe_integer():
    """Test r at max safe integer boundary"""
    policy, files = create_valid_bundle()
    files["adapter_config.json"] = json.dumps({"r": 9007199254740991, "target_modules": ["q_proj"]})
    # Recompute inventory since adapter_config.json changed
    inventory_entries = []
    for filename in sorted(files.keys(), key=lambda x: x.encode("utf-8")):
        if filename == "inventory.json":
            continue
        content = files[filename]
        inventory_entries.append({
            "name": filename,
            "bytes": len(content.encode("utf-8")),
            "sha256": sha256_digest(content)
        })
    files["inventory.json"] = json.dumps(inventory_entries, ensure_ascii=False, separators=(",", ":"))
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "admit"
    assert result["violations"] == []


def test_r_exceeds_max_safe_integer():
    """Test r exceeding max safe integer"""
    policy, files = create_valid_bundle()
    files["adapter_config.json"] = json.dumps({"r": 9007199254740992, "target_modules": ["q_proj"]})
    # Recompute inventory since adapter_config.json changed
    inventory_entries = []
    for filename in sorted(files.keys(), key=lambda x: x.encode("utf-8")):
        if filename == "inventory.json":
            continue
        content = files[filename]
        inventory_entries.append({
            "name": filename,
            "bytes": len(content.encode("utf-8")),
            "sha256": sha256_digest(content)
        })
    files["inventory.json"] = json.dumps(inventory_entries, ensure_ascii=False, separators=(",", ":"))
    result = verify_bundle(policy, files)
    
    assert result["decision"] == "reject"
    assert "INVALID_ADAPTER_CONFIG" in result["violations"]


def test_inventory_digest_computed_from_actual_files():
    """Test that inventoryDigest is computed from actual files even when inventory.json is invalid"""
    policy, files = create_valid_bundle()
    files["inventory.json"] = "invalid json"
    result = verify_bundle(policy, files)
    
    # Should still return an inventoryDigest
    assert "inventoryDigest" in result
    assert len(result["inventoryDigest"]) == 64
