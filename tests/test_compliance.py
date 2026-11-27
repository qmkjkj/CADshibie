from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from compliance.params import ModelComplianceError, audit_models

ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def test_manifest_with_no_models_is_valid(tmp_path):
    _write_manifest(tmp_path, {
        "limit": 5_000_000_000,
        "total_parameters": 0,
        "external_network_required": False,
        "external_apis": [],
        "models": [],
    })
    report = audit_models(tmp_path)
    assert report.ok
    assert report.total_parameters == 0


def test_manifest_rejects_external_network(tmp_path):
    _write_manifest(tmp_path, {
        "limit": 5_000_000_000,
        "total_parameters": 0,
        "external_network_required": True,
        "models": [],
    })
    with pytest.raises(ModelComplianceError):
        audit_models(tmp_path)


def test_manifest_rejects_external_api(tmp_path):
    _write_manifest(tmp_path, {
        "limit": 5_000_000_000,
        "total_parameters": 0,
        "external_network_required": False,
        "external_apis": ["https://api.example.com"],
        "models": [],
    })
    with pytest.raises(ModelComplianceError):
        audit_models(tmp_path)


def test_manifest_rejects_parameters_over_limit(tmp_path):
    _write_manifest(tmp_path, {
        "limit": 5_000_000_000,
        "total_parameters": 6_000_000_000,
        "external_network_required": False,
        "external_apis": [],
        "models": [{
            "name": "too-big", "kind": "test", "path": "missing.bin",
            "parameters": 6_000_000_000, "optional": True,
        }],
    })
    with pytest.raises(ModelComplianceError):
        audit_models(tmp_path)


def test_manifest_requires_missing_non_optional_file(tmp_path):
    _write_manifest(tmp_path, {
        "limit": 5_000_000_000,
        "total_parameters": 0,
        "external_network_required": False,
        "external_apis": [],
        "models": [{
            "name": "missing", "kind": "test", "path": "missing.bin",
            "parameters": 0,
        }],
    })
    with pytest.raises(ModelComplianceError):
        audit_models(tmp_path)


def test_manifest_counts_sklearn_tree_nodes(tmp_path):
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=0)
    model.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    joblib.dump(model, tmp_path / "primitive_classifier.joblib")
    nodes = int(sum(tree.tree_.node_count for tree in model.estimators_))
    _write_manifest(tmp_path, {
        "limit": 5_000_000_000,
        "total_parameters": nodes,
        "external_network_required": False,
        "external_apis": [],
        "models": [{
            "name": "primitive-classifier", "kind": "sklearn-classifier",
            "path": "primitive_classifier.joblib", "parameters": nodes,
        }],
    })
    report = audit_models(tmp_path)
    assert report.models[0].parameter_source == "sklearn-tree-nodes"
    assert report.total_parameters == nodes


def test_offline_guard_blocks_network():
    script = (
        "from compliance.net_guard import enable_offline_guard;"
        "enable_offline_guard();"
        "import socket;"
        "socket.create_connection(('example.com', 80), timeout=0.1)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode != 0
    assert "OfflineViolation" in completed.stderr


def test_source_scan_excludes_environment_but_checks_project_code(tmp_path, monkeypatch):
    from tools import check_compliance

    for directory in (".venv/Lib/site-packages/vendor", "_work/experiment", "cadvec"):
        path = tmp_path / directory / "transport.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("import requests\n", encoding="utf-8")
    monkeypatch.setattr(check_compliance, "ROOT", tmp_path)
    problems = check_compliance.scan_imports()
    assert len(problems) == 1
    assert "cadvec" in problems[0] and "requests" in problems[0]
