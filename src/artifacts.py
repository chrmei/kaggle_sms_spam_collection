import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pipeline(output_dir: str, model: Any) -> Path:
    out = ensure_dir(output_dir) / "model.joblib"
    joblib.dump(model, out)
    return out


def save_json(output_dir: str, filename: str, payload: Dict[str, Any]) -> Path:
    out = ensure_dir(output_dir) / filename
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def save_dataframe(output_dir: str, filename: str, df: pd.DataFrame) -> Path:
    out = ensure_dir(output_dir) / filename
    df.to_csv(out, index=True)
    return out


def save_metrics_history(output_dir: str, metrics: Dict[str, Any]) -> Path:
    out = ensure_dir(output_dir) / "metrics.json"
    model_name = str(metrics.get("model_name", "unknown_model"))
    config = metrics.get("config", {})
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:12]

    entry = dict(metrics)
    entry["config_hash"] = config_hash
    entry["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()

    def _normalize_entry(payload: Dict[str, Any]) -> Dict[str, Any]:
        payload_config = payload.get("config", {})
        payload_hash = payload.get("config_hash")
        if not payload_hash:
            payload_json = json.dumps(payload_config, sort_keys=True, separators=(",", ":"))
            payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:12]
        normalized = dict(payload)
        normalized["config_hash"] = payload_hash
        normalized.setdefault("recorded_at_utc", datetime.now(timezone.utc).isoformat())
        return normalized

    history: Dict[str, Any] = {}
    if out.exists():
        try:
            loaded = json.loads(out.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                # Backward compatibility: migrate old flat one-run metrics format.
                if "model_name" in loaded and "config" in loaded:
                    migrated = _normalize_entry(loaded)
                    migrated_model = str(migrated.get("model_name", "unknown_model"))
                    migrated_hash = str(migrated["config_hash"])
                    history = {migrated_model: {migrated_hash: migrated}}
                else:
                    history = loaded
        except json.JSONDecodeError:
            history = {}
    model_history = history.get(model_name, {})
    if not isinstance(model_history, dict):
        model_history = {}
    model_history[config_hash] = entry
    history[model_name] = model_history

    out.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return out
