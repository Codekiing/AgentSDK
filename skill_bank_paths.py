"""Shared path resolution for skill-bank and trajectory tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
SKILL_BANK_ROOT = REPO_ROOT / "skill-bank"
CLAUDE_SKILLS_ROOT = REPO_ROOT / ".claude" / "skills"
TRAJ_OUTPUT_ROOT = REPO_ROOT / "traj_opt" / "output"
RLLM_OUTPUT_ROOT = REPO_ROOT / "rllm_train" / "output"
PACKAGES_ROOT = SKILL_BANK_ROOT / "packages"


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts).resolve()


def skill_bank_path(*parts: str) -> Path:
    return SKILL_BANK_ROOT.joinpath(*parts).resolve()


def compiled_root(compiled_dir: str = "compiled") -> Path:
    return skill_bank_path(compiled_dir)


def claude_skill_output(output_rel: str) -> Path:
    return repo_path(output_rel)


def ensure_within(path: Path, root: Path, label: str) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes {resolved_root}: {resolved_path}") from exc
    return resolved_path


def validate_compile_output(path: Path) -> Path:
    return ensure_within(path, CLAUDE_SKILLS_ROOT, "compiled skill output")


def validate_snapshot_path(path: Path, compiled_dir: str = "compiled") -> Path:
    return ensure_within(path, compiled_root(compiled_dir), "snapshot")


def validate_package_path(path: Path) -> Path:
    return ensure_within(path, PACKAGES_ROOT, "package")


def validate_patch_path(path: Path, group: str, skill_name: str) -> Path:
    expected_root = skill_bank_path(group, skill_name, "patches")
    return ensure_within(path, expected_root, "patch")


def load_package_registry() -> dict[str, Any]:
    path = skill_bank_path("registry.json")
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_active_skill_package(domain: str | None = None) -> str:
    registry = load_package_registry()
    if domain:
        vertical = registry.get("vertical", {}).get(domain, {})
        if vertical.get("current"):
            return vertical["current"]
    return registry.get("stable", {}).get("current", "")


def package_manifest_path(package_id: str) -> Path | None:
    registry = load_package_registry()
    entries: list[dict[str, Any]] = []
    entries.extend(registry.get("stable", {}).get("packages", []))
    for section in ("experimental", "vertical"):
        for domain_entry in registry.get(section, {}).values():
            entries.extend(domain_entry.get("packages", []))
    entries.extend(registry.get("task_packages", []))
    entries.extend(registry.get("lineage_archive", []))

    for entry in entries:
        if entry.get("id") == package_id and entry.get("path"):
            return repo_path(entry["path"])
    return None
