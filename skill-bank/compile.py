#!/usr/bin/env python3
"""
Skill Bank Compiler

Compiles base.md + active patches into final SKILL.md files.

Usage:
    python skill-bank/compile.py rllm-config              # single skill
    python skill-bank/compile.py rllm-config -p small-model # with profile
    python skill-bank/compile.py --group rllm              # entire group
    python skill-bank/compile.py --all                     # everything
    python skill-bank/compile.py --dry-run rllm-config     # preview only
    python skill-bank/compile.py --diff rllm-config        # show diff
    python skill-bank/compile.py --status                  # patch summary
    python skill-bank/compile.py --squash rllm-config      # merge patches into base
"""

import argparse
import copy
import difflib
import json
import os
import re
import shutil
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import yaml


BANK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BANK_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skill_bank_paths import (  # noqa: E402
    PACKAGES_ROOT,
    RLLM_OUTPUT_ROOT,
    SKILL_BANK_ROOT,
    TRAJ_OUTPUT_ROOT,
    compiled_root,
    ensure_within,
    repo_path,
    skill_bank_path,
    validate_compile_output,
    validate_package_path,
    validate_snapshot_path,
)
SECTION_OPEN = re.compile(r"<!--\s*section:([a-z0-9-]+)\s*-->")
SECTION_CLOSE = re.compile(r"<!--\s*/section:([a-z0-9-]+)\s*-->")
FRONTMATTER_DELIM = re.compile(r"^---\s*$")


def load_bank_yaml():
    path = BANK_DIR / "bank.yaml"
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def find_skill(bank, skill_name):
    for group_name, group in bank.get("groups", {}).items():
        skills = group.get("skills", {})
        if skill_name in skills:
            return group_name, skills[skill_name]
    return None, None


def get_skills_in_group(bank, group_name):
    group = bank.get("groups", {}).get(group_name)
    if not group:
        return {}
    return group.get("skills", {})


def get_all_skills(bank):
    result = {}
    for group_name, group in bank.get("groups", {}).items():
        for skill_name, skill_cfg in group.get("skills", {}).items():
            result[skill_name] = (group_name, skill_cfg)
    return result


def resolve_output_path(bank, output_rel):
    output_base = bank.get("settings", {}).get("output_base", "../")
    output_path = (BANK_DIR / output_base / output_rel).resolve()
    return validate_compile_output(output_path)


def resolve_skill_dir(group_name, skill_name):
    return skill_bank_path(group_name, skill_name)


def resolve_manifest_path(group_name, skill_name):
    return resolve_skill_dir(group_name, skill_name) / "manifest.yaml"


def parse_frontmatter(text):
    lines = text.split("\n")
    if not lines or not FRONTMATTER_DELIM.match(lines[0]):
        return {}, text
    end = None
    for i in range(1, len(lines)):
        if FRONTMATTER_DELIM.match(lines[i]):
            end = i
            break
    if end is None:
        return {}, text
    fm_text = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1 :])
    try:
        fm = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as e:
        print(f"Warning: YAML parse error in frontmatter: {e}", file=sys.stderr)
        fm = {}
    return fm, body


def parse_sections(body):
    sections = OrderedDict()
    gaps = []
    current_section = None
    current_lines = []
    gap_lines = []
    section_order = []

    for line in body.split("\n"):
        open_m = SECTION_OPEN.match(line.strip())
        close_m = SECTION_CLOSE.match(line.strip())

        if open_m:
            if current_section is not None:
                print(f"Error: nested section '{open_m.group(1)}' inside '{current_section}'", file=sys.stderr)
                sys.exit(1)
            if gap_lines:
                gaps.append(("\n".join(gap_lines), len(section_order)))
                gap_lines = []
            current_section = open_m.group(1)
            current_lines = []
            section_order.append(current_section)
        elif close_m:
            name = close_m.group(1)
            if current_section != name:
                print(f"Error: closing '{name}' but current section is '{current_section}'", file=sys.stderr)
                sys.exit(1)
            content = "\n".join(current_lines)
            if content.startswith("\n"):
                content = content[1:]
            if content.endswith("\n"):
                content = content[:-1]
            sections[current_section] = content
            current_section = None
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)
        else:
            gap_lines.append(line)

    if current_section is not None:
        print(f"Error: unclosed section '{current_section}'", file=sys.stderr)
        sys.exit(1)

    if gap_lines:
        gaps.append(("\n".join(gap_lines), len(section_order)))

    return sections, gaps, section_order


def load_patch(patch_path):
    text = patch_path.read_text()
    fm, body = parse_frontmatter(text)
    fm["_body"] = body.strip()
    fm["_path"] = str(patch_path)
    return fm


def topo_sort(patches):
    graph = {p["id"]: set() for p in patches}
    patch_map = {p["id"]: p for p in patches}
    local_ids = set(graph.keys())

    for p in patches:
        for dep in p.get("depends_on", []):
            if ":" not in dep and dep in local_ids:
                graph[p["id"]].add(dep)

    visited = set()
    order = []
    visiting = set()

    def dfs(node):
        if node in visiting:
            print(f"Error: circular dependency involving '{node}'", file=sys.stderr)
            sys.exit(1)
        if node in visited:
            return
        visiting.add(node)
        for dep in graph.get(node, set()):
            dfs(dep)
        visiting.discard(node)
        visited.add(node)
        order.append(node)

    for node in graph:
        dfs(node)

    return [patch_map[pid] for pid in order]


def check_conflicts(patches):
    active_ids = {p["id"] for p in patches}
    for p in patches:
        for conflict in p.get("conflicts_with", []):
            if conflict in active_ids:
                print(
                    f"Error: patch '{p['id']}' conflicts with active patch '{conflict}'",
                    file=sys.stderr,
                )
                sys.exit(1)


def check_cross_skill_deps(patches, bank):
    for p in patches:
        for dep in p.get("depends_on", []):
            if ":" not in dep:
                continue
            skill_ref, patch_ref = dep.split(":", 1)
            group_name, _ = find_skill(bank, skill_ref)
            if group_name is None:
                print(f"Error: cross-skill dep '{dep}' — skill '{skill_ref}' not found", file=sys.stderr)
                sys.exit(1)
            manifest_path = resolve_manifest_path(group_name, skill_ref)
            if not manifest_path.exists():
                print(f"Error: cross-skill dep '{dep}' — manifest not found at {manifest_path}", file=sys.stderr)
                sys.exit(1)
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f) or {}
            if patch_ref not in manifest.get("active", []):
                print(
                    f"Error: cross-skill dep '{dep}' — patch '{patch_ref}' is not active in '{skill_ref}'",
                    file=sys.stderr,
                )
                sys.exit(1)


def apply_patches(sections, section_order, patches):
    sections = OrderedDict(sections)
    section_order = list(section_order)

    for p in patches:
        target = p.get("target_section", "")
        action = p.get("action", "replace")
        body = p["_body"]

        if action == "insert_after":
            if target and target not in sections:
                print(f"Error: patch '{p['id']}' targets non-existent section '{target}'", file=sys.stderr)
                sys.exit(1)
            new_name = p["id"]
            sections[new_name] = body
            if target:
                idx = section_order.index(target)
                section_order.insert(idx + 1, new_name)
            else:
                section_order.append(new_name)
            continue

        if target not in sections:
            print(f"Error: patch '{p['id']}' targets non-existent section '{target}'", file=sys.stderr)
            sys.exit(1)

        if action == "replace":
            sections[target] = body
        elif action == "append":
            sections[target] = sections[target] + "\n\n" + body
        elif action == "prepend":
            sections[target] = body + "\n\n" + sections[target]
        else:
            print(f"Error: unknown action '{action}' in patch '{p['id']}'", file=sys.stderr)
            sys.exit(1)

    return sections, section_order


def reassemble(frontmatter, sections, section_order, gaps):
    parts = []

    if frontmatter:
        fm_text = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()
        parts.append(f"---\n{fm_text}\n---\n")

    gap_map = {}
    for text, pos in gaps:
        gap_map.setdefault(pos, []).append(text)

    for pos in sorted(gap_map):
        if pos == 0:
            for g in gap_map[pos]:
                stripped = g.strip("\n")
                if stripped:
                    parts.append(stripped)

    for i, sec_name in enumerate(section_order):
        content = sections.get(sec_name, "")
        parts.append(content)

        after_pos = i + 1
        if after_pos in gap_map:
            for g in gap_map[after_pos]:
                stripped = g.strip("\n")
                if stripped:
                    parts.append(stripped)

    remaining_pos = len(section_order)
    for pos in sorted(gap_map):
        if pos > remaining_pos:
            for g in gap_map[pos]:
                stripped = g.strip("\n")
                if stripped:
                    parts.append(stripped)

    return "\n\n".join(parts) + "\n"


def get_next_version(compiled_dir):
    if not compiled_dir.exists():
        return 1
    existing = [d.name for d in compiled_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not existing:
        return 1
    nums = []
    for name in existing:
        try:
            nums.append(int(name[1:]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def compile_skill(skill_name, bank, profile=None, dry_run=False, diff_mode=False):
    group_name, skill_cfg = find_skill(bank, skill_name)
    if group_name is None:
        print(f"Error: skill '{skill_name}' not found in bank.yaml", file=sys.stderr)
        return False

    skill_dir = resolve_skill_dir(group_name, skill_name)
    if not skill_dir.exists():
        print(f"Error: skill directory not found: {skill_dir}", file=sys.stderr)
        return False

    manifest_path = resolve_manifest_path(group_name, skill_name)
    if not manifest_path.exists():
        print(f"Error: manifest.yaml not found in {skill_dir}", file=sys.stderr)
        return False

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}

    base_file = skill_dir / manifest.get("base", "base.md")
    if not base_file.exists():
        print(f"Error: base file not found: {base_file}", file=sys.stderr)
        return False

    base_text = base_file.read_text()
    frontmatter, body = parse_frontmatter(base_text)
    sections, gaps, section_order = parse_sections(body)

    if profile:
        profiles = manifest.get("profiles", {})
        if profile not in profiles:
            print(f"Error: profile '{profile}' not found in {manifest_path}", file=sys.stderr)
            return False
        active_ids = profiles[profile].get("active", [])
    else:
        active_ids = manifest.get("active", [])

    if not active_ids:
        compiled = reassemble(frontmatter, sections, section_order, gaps)
    else:
        patches_dir = skill_dir / "patches"
        patches = []
        warnings = []

        for pid in active_ids:
            patch_file = patches_dir / f"{pid}.md"
            if not patch_file.exists():
                print(f"Error: patch file not found: {patch_file}", file=sys.stderr)
                return False
            patch = load_patch(patch_file)
            if "id" not in patch:
                patch["id"] = pid

            status = patch.get("status", "active")
            if status == "archived":
                print(f"Error: patch '{pid}' is archived and cannot be activated", file=sys.stderr)
                return False
            if status == "deprecated":
                superseded = patch.get("superseded_by", "")
                warnings.append(f"Warning: patch '{pid}' is deprecated (superseded by '{superseded}')")

            patches.append(patch)

        for w in warnings:
            print(w, file=sys.stderr)

        check_conflicts(patches)
        check_cross_skill_deps(patches, bank)
        sorted_patches = topo_sort(patches)
        sections, section_order = apply_patches(sections, section_order, sorted_patches)
        compiled = reassemble(frontmatter, sections, section_order, gaps)

    output_path = resolve_output_path(bank, skill_cfg["output"])

    if diff_mode:
        if output_path.exists():
            current = output_path.read_text().splitlines(keepends=True)
            new = compiled.splitlines(keepends=True)
            diff = difflib.unified_diff(current, new, fromfile=str(output_path), tofile="compiled")
            diff_text = "".join(diff)
            if diff_text:
                print(diff_text)
            else:
                print(f"  {skill_name}: no changes")
        else:
            print(f"  {skill_name}: output file does not exist yet (new)")
        return True

    if dry_run:
        print(f"--- {skill_name} (dry-run) ---")
        print(compiled[:500])
        if len(compiled) > 500:
            print(f"... ({len(compiled)} chars total)")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(compiled)
    print(f"  {skill_name} -> {output_path}")
    return True


def save_snapshot(bank):
    compiled_name = bank.get("settings", {}).get("compiled_dir", "compiled")
    compiled_dir = compiled_root(compiled_name)
    version = get_next_version(compiled_dir)
    snapshot_dir = validate_snapshot_path(compiled_dir / f"v{version:03d}", compiled_name)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    all_skills = get_all_skills(bank)
    for skill_name, (group_name, skill_cfg) in all_skills.items():
        output_path = resolve_output_path(bank, skill_cfg["output"])
        if output_path.exists():
            dest = snapshot_dir / group_name / f"{skill_name}.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, dest)

    print(f"  Snapshot saved: {snapshot_dir}")
    return snapshot_dir


def load_registry():
    path = skill_bank_path("registry.json")
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_registry(bank):
    registry = load_registry()
    errors = []
    warnings = []

    if SKILL_BANK_ROOT != BANK_DIR:
        errors.append(f"skill-bank root mismatch: {SKILL_BANK_ROOT} != {BANK_DIR}")

    if registry is None:
        warnings.append("registry.json not found; package layer is not initialized")
    else:
        paths = registry.get("paths", {})
        expected_paths = {
            "skill_bank_root": "skill-bank",
            "runtime_output_root": ".claude/skills",
            "compiled_snapshot_root": "skill-bank/compiled",
            "packages_root": "skill-bank/packages",
            "traj_output_root": "traj_opt/output",
            "rllm_output_root": "rllm_train/output",
        }
        for key, expected in expected_paths.items():
            actual = paths.get(key)
            if actual != expected:
                errors.append(f"registry paths.{key} expected {expected!r}, got {actual!r}")

        package_entries = []
        package_entries.extend(registry.get("stable", {}).get("packages", []))
        for section in ("experimental", "vertical"):
            for domain_entry in registry.get(section, {}).values():
                package_entries.extend(domain_entry.get("packages", []))
        package_entries.extend(registry.get("task_packages", []))
        package_entries.extend(registry.get("lineage_archive", []))

        seen_ids = set()
        for package in package_entries:
            package_id = package.get("id")
            if not package_id:
                errors.append(f"package entry missing id: {package}")
            elif package_id in seen_ids:
                errors.append(f"duplicate package id: {package_id}")
            else:
                seen_ids.add(package_id)

            manifest_path = repo_path(package.get("path", ""))
            try:
                validate_package_path(manifest_path)
            except ValueError as exc:
                errors.append(str(exc))
            if not manifest_path.exists():
                warnings.append(f"package manifest missing: {manifest_path}")

    compiled_name = bank.get("settings", {}).get("compiled_dir", "compiled")
    compiled_dir = compiled_root(compiled_name)
    try:
        validate_snapshot_path(compiled_dir, compiled_name)
    except ValueError as exc:
        errors.append(str(exc))

    for skill_name, (group_name, skill_cfg) in sorted(get_all_skills(bank).items(), key=lambda x: (x[1][0], x[0])):
        skill_dir = resolve_skill_dir(group_name, skill_name)
        manifest_path = resolve_manifest_path(group_name, skill_name)
        output_path = resolve_output_path(bank, skill_cfg["output"])

        if not output_path.match("*/.claude/skills/*/SKILL.md"):
            errors.append(f"unexpected output path for {skill_name}: {output_path}")

        compiled_output = output_path.exists()
        if not skill_dir.exists():
            if compiled_output:
                warnings.append(f"compiled-only skill source directory missing: {skill_name} ({skill_dir})")
            else:
                errors.append(f"source directory missing for {skill_name}: {skill_dir}")
            continue
        if not manifest_path.exists():
            if compiled_output:
                warnings.append(f"compiled-only skill source missing manifest: {skill_name} ({manifest_path})")
            else:
                errors.append(f"manifest missing for {skill_name}: {manifest_path}")
            continue

        with open(manifest_path, encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
        base_file = skill_dir / manifest.get("base", "base.md")
        if not base_file.exists():
            errors.append(f"base file missing for {skill_name}: {base_file}")

    for warning in warnings:
        print(f"WARN: {warning}")
    for error in errors:
        print(f"ERROR: {error}")

    if errors:
        print(f"Validation failed: {len(errors)} error(s), {len(warnings)} warning(s)")
        return False
    print(f"Validation passed: {len(warnings)} warning(s)")
    return True


def latest_compiled_version(bank):
    compiled_name = bank.get("settings", {}).get("compiled_dir", "compiled")
    root = compiled_root(compiled_name)
    if not root.exists():
        return None, None
    versions = []
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("v"):
            continue
        try:
            versions.append((int(path.name[1:]), path))
        except ValueError:
            continue
    if not versions:
        return None, None
    _, path = max(versions, key=lambda item: item[0])
    return path.name, path


def write_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp_path, path)


def list_packages():
    registry = load_registry()
    if registry is None:
        print("No package registry found at skill-bank/registry.json")
        return False

    stable = registry.get("stable", {})
    print("[stable]")
    print(f"  current: {stable.get('current', '')}")
    for package in stable.get("packages", []):
        print(f"  - {package.get('id')} ({package.get('status')}) -> {package.get('path')}")

    for section in ("experimental", "vertical"):
        print(f"\n[{section}]")
        entries = registry.get(section, {})
        if not entries:
            print("  <empty>")
            continue
        for key, value in entries.items():
            print(f"  {key}: {value}")

    for section in ("task_packages", "lineage_archive"):
        print(f"\n[{section}]")
        entries = registry.get(section, [])
        if not entries:
            print("  <empty>")
            continue
        for entry in entries:
            print(f"  - {entry.get('id')} ({entry.get('status')}) -> {entry.get('path')}")
    return True


def freeze_stable_package(bank, name, dry_run=False):
    if not name:
        print("Error: --package stable requires --name", file=sys.stderr)
        return False

    version_name, version_path = latest_compiled_version(bank)
    if version_path is None:
        print("Error: no compiled snapshots found under skill-bank/compiled", file=sys.stderr)
        return False

    package_dir = validate_package_path(PACKAGES_ROOT / "stable" / name)
    manifest_path = validate_package_path(package_dir / "manifest.json")
    compiled_dest = validate_package_path(package_dir / "compiled")
    package_id = f"stable:{name}"

    all_skills = get_all_skills(bank)
    skills = []
    for skill_name, (group_name, skill_cfg) in sorted(all_skills.items(), key=lambda x: (x[1][0], x[0])):
        skills.append({
            "name": skill_name,
            "group": group_name,
            "source_path": f"skill-bank/{group_name}/{skill_name}",
            "output": skill_cfg["output"],
        })

    manifest = {
        "id": package_id,
        "type": "stable",
        "status": "candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_compiled_version": version_name,
        "source_snapshot": f"skill-bank/compiled/{version_name}",
        "compatibility_mode": True,
        "source_layout": "skill-bank/<group>/<skill>",
        "runtime_output_root": ".claude/skills",
        "snapshot_root": "skill-bank/compiled",
        "skill_count": len(skills),
        "skills": skills,
    }

    if dry_run:
        print(f"Would freeze {package_id}")
        print(f"  source: {version_path}")
        print(f"  manifest: {manifest_path}")
        print(f"  compiled: {compiled_dest}")
        return True

    package_dir.mkdir(parents=True, exist_ok=True)
    if compiled_dest.exists():
        shutil.rmtree(compiled_dest)
    shutil.copytree(version_path, compiled_dest)
    write_json_atomic(manifest_path, manifest)

    registry = load_registry() or {}
    stable = registry.setdefault("stable", {})
    stable["current"] = package_id
    packages = stable.setdefault("packages", [])
    packages = [p for p in packages if p.get("id") != package_id]
    packages.append({
        "id": package_id,
        "type": "stable",
        "status": "candidate",
        "source_compiled_version": version_name,
        "path": f"skill-bank/packages/stable/{name}/manifest.json",
        "source_snapshot": f"skill-bank/compiled/{version_name}",
    })
    stable["packages"] = packages
    write_json_atomic(skill_bank_path("registry.json"), registry)

    print(f"Frozen {package_id} from {version_name} -> {package_dir}")
    return True


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_package_entry(registry, package_id):
    stable = registry.get("stable", {})
    for package in stable.get("packages", []):
        if package.get("id") == package_id:
            return package

    for section in ("experimental", "vertical"):
        for domain_entry in registry.get(section, {}).values():
            for package in domain_entry.get("packages", []):
                if package.get("id") == package_id:
                    return package
    return None


def package_ids_for_smoke(registry, package_id=None):
    if package_id:
        return [package_id]
    ids = []
    stable_current = registry.get("stable", {}).get("current")
    if stable_current:
        ids.append(stable_current)
    for domain_entry in registry.get("vertical", {}).values():
        current = domain_entry.get("current")
        if current:
            ids.append(current)
    return ids


def smoke_test_package(bank, package_id):
    registry = load_registry() or {}
    entry = find_package_entry(registry, package_id)
    errors = []
    if entry is None:
        errors.append(f"package not found in registry: {package_id}")
        return errors

    manifest_path = repo_path(entry.get("path", ""))
    try:
        validate_package_path(manifest_path)
    except ValueError as exc:
        errors.append(str(exc))
    if not manifest_path.exists():
        errors.append(f"manifest missing: {manifest_path}")
        return errors

    manifest = read_json(manifest_path)
    required_skills = {"rllm-train", "rllm-config", "rllm-run", "rllm-monitor", "rllm-analyze"}
    skill_names = {skill.get("name") for skill in manifest.get("skills", [])}
    missing_skills = sorted(required_skills - skill_names)
    if missing_skills:
        errors.append(f"{package_id} missing required skills: {', '.join(missing_skills)}")

    for skill in manifest.get("skills", []):
        source_path = repo_path(skill.get("source_path", ""))
        output_path = repo_path(skill.get("output", ""))
        if not source_path.exists():
            errors.append(f"source path missing for {skill.get('name')}: {source_path}")
        try:
            validate_compile_output(output_path)
        except ValueError as exc:
            errors.append(str(exc))

    source_snapshot = manifest.get("source_snapshot") or entry.get("source_snapshot")
    if source_snapshot and not repo_path(source_snapshot).exists():
        errors.append(f"source snapshot missing: {source_snapshot}")

    for skill_name in sorted(required_skills):
        if skill_name in get_all_skills(bank) and not compile_skill(skill_name, bank, dry_run=True):
            errors.append(f"compile dry-run failed for {skill_name}")
    return errors


def smoke_test_packages(bank, package_id=None):
    registry = load_registry() or {}
    ids = package_ids_for_smoke(registry, package_id=package_id)
    if not ids:
        print("No stable.current or vertical.current packages to smoke test")
        return False

    all_errors = []
    for pid in ids:
        print(f"[smoke] {pid}")
        errors = smoke_test_package(bank, pid)
        if errors:
            all_errors.extend(f"{pid}: {error}" for error in errors)
        else:
            print("  ok")

    for error in all_errors:
        print(f"ERROR: {error}")
    if all_errors:
        print(f"Smoke test failed: {len(all_errors)} error(s)")
        return False
    print(f"Smoke test passed: {len(ids)} package(s)")
    return True


def create_experimental_package(name, domain, from_package=None, dry_run=False):
    if not name or not domain:
        print("Error: --package experimental requires --name and --domain", file=sys.stderr)
        return False

    registry = load_registry() or {}
    source_id = from_package or registry.get("stable", {}).get("current")
    if not source_id:
        print("Error: no source package specified and no stable.current in registry", file=sys.stderr)
        return False

    source_entry = find_package_entry(registry, source_id)
    if source_entry is None:
        print(f"Error: source package not found in registry: {source_id}", file=sys.stderr)
        return False

    source_manifest = repo_path(source_entry["path"])
    if not source_manifest.exists():
        print(f"Error: source package manifest not found: {source_manifest}", file=sys.stderr)
        return False
    source_dir = source_manifest.parent

    package_id = f"experimental:{domain}:{name}"
    package_dir = validate_package_path(PACKAGES_ROOT / "experimental" / domain / name)
    manifest_path = validate_package_path(package_dir / "manifest.json")

    if package_dir.exists() and not dry_run:
        print(f"Error: experimental package already exists: {package_dir}", file=sys.stderr)
        return False

    if dry_run:
        print(f"Would create {package_id}")
        print(f"  from: {source_id} ({source_dir})")
        print(f"  to:   {package_dir}")
        return True

    shutil.copytree(source_dir, package_dir)
    manifest = read_json(manifest_path)
    manifest.update({
        "id": package_id,
        "type": "experimental",
        "status": "experimental",
        "domain": domain,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_package": source_id,
    })
    write_json_atomic(manifest_path, manifest)

    experimental = registry.setdefault("experimental", {})
    domain_entry = experimental.setdefault(domain, {"current": None, "packages": []})
    domain_entry["current"] = package_id
    packages = [p for p in domain_entry.get("packages", []) if p.get("id") != package_id]
    packages.append({
        "id": package_id,
        "type": "experimental",
        "status": "experimental",
        "domain": domain,
        "base_package": source_id,
        "path": f"skill-bank/packages/experimental/{domain}/{name}/manifest.json",
    })
    domain_entry["packages"] = packages
    write_json_atomic(skill_bank_path("registry.json"), registry)

    print(f"Created {package_id} from {source_id} -> {package_dir}")
    return True


def build_promotion_review(package_id, domain, source_id, source_manifest):
    source = read_json(source_manifest)
    skills = source.get("skills", [])
    rllm_skills = [s for s in skills if s.get("group") == "rllm"]
    traj_skills = [s for s in skills if s.get("group") == "traj"]
    return {
        "target_package": package_id,
        "domain": domain,
        "source_package": source_id,
        "source_package_manifest": str(source_manifest.relative_to(REPO_ROOT)),
        "classification": {
            "stable_candidates": [
                {
                    "rule": "Only promote here after proving the change is domain-independent across multiple domains.",
                    "skills": [],
                }
            ],
            "vertical_candidates": [
                {
                    "rule": "Reusable domain behavior for the target domain; eligible for vertical package contents.",
                    "skills": [s.get("name") for s in rllm_skills],
                }
            ],
            "task_package_only": [
                "single-run configs",
                "model checkpoints",
                "training logs",
                "eval outputs",
                "dataset snapshots",
                "reward implementation snapshots",
            ],
            "lineage_archive_only": [
                "traj round status",
                "raw Claude Code hook events",
                "segmented trajectories",
                "trajectory analysis reports",
                "patch generation evidence",
            ],
            "excluded_from_vertical": [
                {
                    "rule": "traj skills optimize rllm skills and must not be promoted as rllm vertical runtime behavior.",
                    "skills": [s.get("name") for s in traj_skills],
                }
            ],
        },
        "required_before_promotion": [
            "At least one successful task-package for this domain exists.",
            "A lineage archive exists for the traj optimization rounds that produced the candidate behavior.",
            "Regression smoke tests pass for stable and the candidate vertical package.",
            "Task-private artifacts are kept in task-packages, not copied into vertical.",
        ],
    }


def print_promotion_review(review):
    print(json.dumps(review, indent=2, ensure_ascii=False))


def promote_vertical_package(name, domain, from_package=None, dry_run=False, review=False):
    if not name or not domain:
        print("Error: --package vertical requires --name and --domain", file=sys.stderr)
        return False

    registry = load_registry() or {}
    source_id = from_package or registry.get("experimental", {}).get(domain, {}).get("current")
    if not source_id:
        print("Error: no source experimental package specified and no experimental current for domain", file=sys.stderr)
        return False

    source_entry = find_package_entry(registry, source_id)
    if source_entry is None:
        print(f"Error: source package not found in registry: {source_id}", file=sys.stderr)
        return False

    source_manifest = repo_path(source_entry["path"])
    if not source_manifest.exists():
        print(f"Error: source package manifest not found: {source_manifest}", file=sys.stderr)
        return False
    source_dir = source_manifest.parent

    package_id = f"vertical:{domain}:{name}"
    package_dir = validate_package_path(PACKAGES_ROOT / "vertical" / domain / name)
    manifest_path = validate_package_path(package_dir / "manifest.json")
    promotion_review = build_promotion_review(package_id, domain, source_id, source_manifest)

    if review:
        print_promotion_review(promotion_review)
        return True

    if package_dir.exists() and not dry_run:
        print(f"Error: vertical package already exists: {package_dir}", file=sys.stderr)
        return False

    if dry_run:
        print(f"Would promote {package_id}")
        print(f"  from: {source_id} ({source_dir})")
        print(f"  to:   {package_dir}")
        return True

    shutil.copytree(source_dir, package_dir)
    manifest = read_json(manifest_path)
    manifest.update({
        "id": package_id,
        "type": "vertical",
        "status": "candidate",
        "domain": domain,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_experimental_package": source_id,
        "promotion_review": promotion_review,
    })
    write_json_atomic(manifest_path, manifest)

    vertical = registry.setdefault("vertical", {})
    domain_entry = vertical.setdefault(domain, {"current": None, "packages": []})
    domain_entry["current"] = package_id
    packages = [p for p in domain_entry.get("packages", []) if p.get("id") != package_id]
    packages.append({
        "id": package_id,
        "type": "vertical",
        "status": "candidate",
        "domain": domain,
        "source_experimental_package": source_id,
        "path": f"skill-bank/packages/vertical/{domain}/{name}/manifest.json",
    })
    domain_entry["packages"] = packages
    write_json_atomic(skill_bank_path("registry.json"), registry)

    print(f"Promoted {package_id} from {source_id} -> {package_dir}")
    return True


def choose_default_task_source(registry, domain):
    if domain:
        vertical_current = registry.get("vertical", {}).get(domain, {}).get("current")
        if vertical_current:
            return vertical_current
        experimental_current = registry.get("experimental", {}).get(domain, {}).get("current")
        if experimental_current:
            return experimental_current
    return registry.get("stable", {}).get("current")


def resolve_run_dir(run_id=None, run_dir=None):
    if run_dir:
        path = repo_path(run_dir)
    elif run_id:
        path = RLLM_OUTPUT_ROOT / "runs" / run_id
    else:
        return None
    return ensure_within(path, RLLM_OUTPUT_ROOT, "rllm run")


def read_run_config(run_dir):
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def copy_if_exists(src, dest, package_dir):
    if not src.exists():
        return None
    dest = validate_package_path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
    return str(dest.relative_to(package_dir))


def freeze_task_package(name, domain=None, from_package=None, run_id=None, run_dir=None, dry_run=False):
    if not name:
        print("Error: --package task-package requires --name", file=sys.stderr)
        return False

    registry = load_registry() or {}
    resolved_run_dir = resolve_run_dir(run_id=run_id, run_dir=run_dir)
    run_config = {}
    if resolved_run_dir is not None:
        if not resolved_run_dir.exists():
            print(f"Error: run directory not found: {resolved_run_dir}", file=sys.stderr)
            return False
        run_config = read_run_config(resolved_run_dir)
        domain = domain or run_config.get("task_type")

    source_id = from_package or run_config.get("skill_package_id") or choose_default_task_source(registry, domain)
    if not source_id:
        print("Error: no source package specified and no default stable/vertical/experimental source", file=sys.stderr)
        return False

    source_entry = find_package_entry(registry, source_id)
    if source_entry is None:
        print(f"Error: source package not found in registry: {source_id}", file=sys.stderr)
        return False

    source_manifest = repo_path(source_entry["path"])
    if not source_manifest.exists():
        print(f"Error: source package manifest not found: {source_manifest}", file=sys.stderr)
        return False
    source_dir = source_manifest.parent

    package_id = f"task:{name}"
    package_dir = validate_package_path(PACKAGES_ROOT / "task-packages" / name)
    manifest_path = validate_package_path(package_dir / "package_manifest.json")
    skills_snapshot = validate_package_path(package_dir / "skills" / "source_package")

    if package_dir.exists() and not dry_run:
        print(f"Error: task package already exists: {package_dir}", file=sys.stderr)
        return False

    if dry_run:
        print(f"Would create {package_id}")
        print(f"  source package: {source_id} ({source_dir})")
        if resolved_run_dir is not None:
            print(f"  run:            {resolved_run_dir}")
        print(f"  package:        {package_dir}")
        return True

    for subdir in (
        "agent",
        "skills",
        "code",
        "configs",
        "data",
        "rewards",
        "eval",
        "docs",
        "trajectories",
    ):
        validate_package_path(package_dir / subdir).mkdir(parents=True, exist_ok=True)

    shutil.copytree(source_dir, skills_snapshot)
    copied_artifacts = {}
    artifact_refs = {}
    if resolved_run_dir is not None:
        copied_artifacts["config"] = copy_if_exists(resolved_run_dir / "config.json", package_dir / "configs" / "config.json", package_dir)
        copied_artifacts["analysis"] = copy_if_exists(resolved_run_dir / "analysis.json", package_dir / "eval" / "analysis.json", package_dir)
        copied_artifacts["perf_stats"] = copy_if_exists(resolved_run_dir / "perf_stats.json", package_dir / "eval" / "perf_stats.json", package_dir)
        copied_artifacts["training_log"] = copy_if_exists(resolved_run_dir / "training_log.txt", package_dir / "docs" / "training_log.txt", package_dir)
        copied_artifacts["trajectories"] = copy_if_exists(resolved_run_dir / "trajectories", package_dir / "trajectories" / "run", package_dir)
        final_model = resolved_run_dir / "final_model"
        if final_model.exists():
            artifact_refs["final_model"] = str(final_model)

    now = datetime.now(timezone.utc).isoformat()
    status = "frozen" if resolved_run_dir is not None else "skeleton"
    manifest = {
        "id": package_id,
        "type": "task_package",
        "status": status,
        "created_at": now,
        "domain": domain,
        "task_id": run_config.get("task_id") or name,
        "run_id": run_config.get("run_id") or run_id,
        "source_skill_package": source_id,
        "paths": {
            "agent": "agent/",
            "skills": "skills/source_package/",
            "code": "code/",
            "configs": "configs/",
            "data": "data/",
            "rewards": "rewards/",
            "eval": "eval/",
            "docs": "docs/",
            "trajectories": "trajectories/",
            "provenance": "provenance.json",
        },
        "copied_artifacts": {k: v for k, v in copied_artifacts.items() if v},
        "artifact_refs": artifact_refs,
    }
    provenance = {
        "task_package": package_id,
        "source_skill_package": source_id,
        "source_package_manifest": source_entry["path"],
        "source_run_dir": str(resolved_run_dir) if resolved_run_dir is not None else None,
        "run_config": run_config,
        "created_at": now,
        "compatibility_paths_preserved": True,
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(validate_package_path(package_dir / "provenance.json"), provenance)

    task_packages = registry.setdefault("task_packages", [])
    task_packages = [p for p in task_packages if p.get("id") != package_id]
    task_packages.append({
        "id": package_id,
        "type": "task_package",
        "status": status,
        "domain": domain,
        "task_id": manifest["task_id"],
        "run_id": manifest["run_id"],
        "source_skill_package": source_id,
        "path": f"skill-bank/packages/task-packages/{name}/package_manifest.json",
    })
    registry["task_packages"] = task_packages
    write_json_atomic(skill_bank_path("registry.json"), registry)

    print(f"Created {package_id} from {source_id} -> {package_dir}")
    return True


def parse_rounds(round_value=None, round_range=None):
    rounds = set()
    if round_value:
        for part in str(round_value).split(","):
            part = part.strip()
            if part:
                rounds.add(int(part))
    if round_range:
        start, end = str(round_range).split("-", 1)
        rounds.update(range(int(start), int(end) + 1))
    return sorted(rounds)


def collect_round_evidence(round_nums):
    evidence = []
    for round_num in round_nums:
        round_dir = TRAJ_OUTPUT_ROOT / "rounds" / f"round_{round_num}"
        status_path = round_dir / "status.json"
        status = {}
        if status_path.exists():
            with open(status_path, encoding="utf-8") as f:
                status = json.load(f)
        training = status.get("training", {})
        optimization = status.get("optimization", {})
        session_id = training.get("session_id")
        report_path = optimization.get("report_path")
        evidence.append({
            "round": round_num,
            "round_dir": round_dir,
            "status_path": status_path,
            "status": status,
            "session_id": session_id,
            "raw_dir": TRAJ_OUTPUT_ROOT / "rllm" / "raw" / session_id if session_id and session_id != "unknown" else None,
            "trajectory_dir": TRAJ_OUTPUT_ROOT / "rllm" / "trajectories" / session_id if session_id and session_id != "unknown" else None,
            "report_path": repo_path(report_path) if report_path else None,
            "rllm_reports": sorted((TRAJ_OUTPUT_ROOT / "rllm" / "reports").glob(f"*round{round_num}*")),
        })
    return evidence


def archive_lineage_package(name, domain, from_package=None, round_value=None, round_range=None, dry_run=False):
    if not name or not domain:
        print("Error: --package lineage-archive requires --name and --domain", file=sys.stderr)
        return False

    registry = load_registry() or {}
    source_id = from_package or registry.get("experimental", {}).get(domain, {}).get("current")
    if not source_id:
        print("Error: no source experimental package specified and no experimental current for domain", file=sys.stderr)
        return False

    source_entry = find_package_entry(registry, source_id)
    if source_entry is None:
        print(f"Error: source package not found in registry: {source_id}", file=sys.stderr)
        return False

    source_manifest = repo_path(source_entry["path"])
    if not source_manifest.exists():
        print(f"Error: source package manifest not found: {source_manifest}", file=sys.stderr)
        return False
    source_dir = source_manifest.parent

    round_nums = parse_rounds(round_value=round_value, round_range=round_range)
    round_evidence = collect_round_evidence(round_nums)

    package_id = f"lineage:{domain}:{name}"
    package_dir = validate_package_path(PACKAGES_ROOT / "lineage-archive" / domain / name)
    manifest_path = validate_package_path(package_dir / "lineage_manifest.json")
    snapshot_dir = validate_package_path(package_dir / "package_snapshot")
    rounds_dir = validate_package_path(package_dir / "rounds")
    reports_dir = validate_package_path(package_dir / "reports")
    raw_dir = validate_package_path(package_dir / "raw")
    trajectories_dir = validate_package_path(package_dir / "trajectories")

    if package_dir.exists() and not dry_run:
        print(f"Error: lineage archive already exists: {package_dir}", file=sys.stderr)
        return False

    if dry_run:
        print(f"Would archive {package_id}")
        print(f"  source package: {source_id} ({source_dir})")
        if round_nums:
            print(f"  rounds:         {', '.join(str(n) for n in round_nums)}")
        print(f"  archive:        {package_dir}")
        return True

    package_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, snapshot_dir)
    copied_rounds = []
    missing_evidence = []
    for item in round_evidence:
        round_num = item["round"]
        round_entry = {"round": round_num, "status": item["status"]}
        if item["round_dir"].exists():
            copied = copy_if_exists(item["round_dir"], rounds_dir / f"round_{round_num}", package_dir)
            round_entry["round_dir"] = copied
        else:
            missing_evidence.append(f"round_{round_num}/")
        if item["report_path"] and item["report_path"].exists():
            round_entry["optimization_report"] = copy_if_exists(item["report_path"], reports_dir / f"round_{round_num}" / item["report_path"].name, package_dir)
        for report in item["rllm_reports"]:
            copy_if_exists(report, reports_dir / f"round_{round_num}" / report.name, package_dir)
        if item["raw_dir"] and item["raw_dir"].exists():
            round_entry["raw_events"] = copy_if_exists(item["raw_dir"], raw_dir / f"round_{round_num}", package_dir)
        if item["trajectory_dir"] and item["trajectory_dir"].exists():
            round_entry["segmented_trajectories"] = copy_if_exists(item["trajectory_dir"], trajectories_dir / f"round_{round_num}", package_dir)
        copied_rounds.append(round_entry)

    manifest = {
        "id": package_id,
        "type": "lineage_archive",
        "status": "archived",
        "domain": domain,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_package": source_id,
        "source_package_manifest": source_entry["path"],
        "snapshot": "package_snapshot/",
        "rounds": copied_rounds,
        "missing_evidence": missing_evidence,
        "notes": [
            "This archive preserves the package state used for trajectory-driven evolution.",
            "It is for traceability and reproduction, not the default entrypoint for new tasks.",
        ],
    }
    write_json_atomic(manifest_path, manifest)

    lineage_archive = registry.setdefault("lineage_archive", [])
    lineage_archive = [p for p in lineage_archive if p.get("id") != package_id]
    lineage_archive.append({
        "id": package_id,
        "type": "lineage_archive",
        "status": "archived",
        "domain": domain,
        "source_package": source_id,
        "path": f"skill-bank/packages/lineage-archive/{domain}/{name}/lineage_manifest.json",
    })
    registry["lineage_archive"] = lineage_archive
    write_json_atomic(skill_bank_path("registry.json"), registry)

    print(f"Archived {package_id} from {source_id} -> {package_dir}")
    return True


def show_status(bank, group_filter=None):
    all_skills = get_all_skills(bank)
    current_group = None

    for skill_name, (group_name, skill_cfg) in sorted(all_skills.items(), key=lambda x: (x[1][0], x[0])):
        if group_filter and group_name != group_filter:
            continue

        if group_name != current_group:
            current_group = group_name
            print(f"\n[{group_name}]")

        skill_dir = resolve_skill_dir(group_name, skill_name)
        manifest_path = resolve_manifest_path(group_name, skill_name)

        if not manifest_path.exists():
            print(f"  {skill_name}: no manifest")
            continue

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f) or {}

        active = manifest.get("active", [])
        disabled = manifest.get("disabled", [])
        profiles = list(manifest.get("profiles", {}).keys())

        patches_dir = skill_dir / "patches"
        total_patches = len(list(patches_dir.glob("*.md"))) if patches_dir.exists() else 0

        status_parts = [f"{len(active)} active"]
        if disabled:
            status_parts.append(f"{len(disabled)} disabled")
        if total_patches > len(active) + len(disabled):
            other = total_patches - len(active) - len(disabled)
            status_parts.append(f"{other} other")
        if profiles:
            status_parts.append(f"profiles: {', '.join(profiles)}")

        print(f"  {skill_name}: {total_patches} patches ({', '.join(status_parts)})")


def squash_skill(skill_name, bank):
    group_name, skill_cfg = find_skill(bank, skill_name)
    if group_name is None:
        print(f"Error: skill '{skill_name}' not found", file=sys.stderr)
        return False

    skill_dir = resolve_skill_dir(group_name, skill_name)
    manifest_path = resolve_manifest_path(group_name, skill_name)

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f) or {}

    active_ids = manifest.get("active", [])
    if not active_ids:
        print(f"  {skill_name}: no active patches to squash")
        return True

    save_snapshot(bank)

    if not compile_skill(skill_name, bank):
        return False

    output_path = resolve_output_path(bank, skill_cfg["output"])
    compiled_text = output_path.read_text()

    base_file = skill_dir / manifest.get("base", "base.md")
    base_text = base_file.read_text()
    old_fm, _ = parse_frontmatter(base_text)

    fm_text = yaml.dump(old_fm, default_flow_style=False, allow_unicode=True).strip() if old_fm else ""
    new_base_lines = []
    if fm_text:
        new_base_lines.append(f"---\n{fm_text}\n---\n")

    compiled_fm, compiled_body = parse_frontmatter(compiled_text)
    sections_from_compiled = compiled_body.split("\n\n")

    new_base_lines.append(compiled_body.strip())
    base_file.write_text("\n".join(new_base_lines) + "\n")

    patches_dir = skill_dir / "patches"
    for pid in active_ids:
        patch_file = patches_dir / f"{pid}.md"
        if patch_file.exists():
            text = patch_file.read_text()
            text = re.sub(r"(?m)^status:\s*active", "status: archived", text)
            patch_file.write_text(text)

    manifest["active"] = []
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, allow_unicode=True)

    print(f"  {skill_name}: squashed {len(active_ids)} patches into base.md")
    return True


def main():
    parser = argparse.ArgumentParser(description="Skill Bank Compiler")
    parser.add_argument("skill", nargs="?", help="Skill name to compile")
    parser.add_argument("-p", "--profile", help="Profile name")
    parser.add_argument("--group", help="Compile all skills in a group")
    parser.add_argument("--all", action="store_true", help="Compile all skills")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--diff", action="store_true", help="Show diff with current output")
    parser.add_argument("--status", action="store_true", help="Show patch status summary")
    parser.add_argument("--validate", action="store_true", help="Validate package registry and path invariants")
    parser.add_argument("--list-packages", action="store_true", help="List registered skill packages")
    parser.add_argument("--smoke-test", action="store_true", help="Run regression smoke checks for stable/vertical packages")
    parser.add_argument("--package-id", help="Specific package id for --smoke-test")
    parser.add_argument("--package", choices=["stable", "experimental", "vertical", "task-package", "lineage-archive"], help="Create or inspect a skill package")
    parser.add_argument("--name", help="Package name for --package operations")
    parser.add_argument("--domain", help="Domain name for experimental or vertical packages")
    parser.add_argument("--from-package", help="Source package id for package lifecycle operations")
    parser.add_argument("--run-id", help="rllm_train run_id to freeze into a task package")
    parser.add_argument("--run-dir", help="rllm_train run directory to freeze into a task package")
    parser.add_argument("--round", help="Round number(s) to archive, comma-separated")
    parser.add_argument("--round-range", help="Inclusive round range to archive, e.g. 1-3")
    parser.add_argument("--review", action="store_true", help="Review package promotion classification without writing")
    parser.add_argument("--squash", action="store_true", help="Squash active patches into base")

    args = parser.parse_args()
    bank = load_bank_yaml()

    if args.status:
        show_status(bank, group_filter=args.group)
        return

    if args.validate:
        ok = validate_registry(bank)
        sys.exit(0 if ok else 1)

    if args.list_packages:
        ok = list_packages()
        sys.exit(0 if ok else 1)

    if args.smoke_test:
        ok = smoke_test_packages(bank, package_id=args.package_id)
        sys.exit(0 if ok else 1)

    if args.package:
        if args.package == "stable":
            ok = freeze_stable_package(bank, args.name, dry_run=args.dry_run)
        elif args.package == "experimental":
            ok = create_experimental_package(args.name, args.domain, from_package=args.from_package, dry_run=args.dry_run)
        elif args.package == "vertical":
            ok = promote_vertical_package(
                args.name,
                args.domain,
                from_package=args.from_package,
                dry_run=args.dry_run,
                review=args.review,
            )
        elif args.package == "task-package":
            ok = freeze_task_package(
                args.name,
                domain=args.domain,
                from_package=args.from_package,
                run_id=args.run_id,
                run_dir=args.run_dir,
                dry_run=args.dry_run,
            )
        elif args.package == "lineage-archive":
            ok = archive_lineage_package(
                args.name,
                args.domain,
                from_package=args.from_package,
                round_value=args.round,
                round_range=args.round_range,
                dry_run=args.dry_run,
            )
        else:
            ok = False
        sys.exit(0 if ok else 1)

    if args.squash:
        if not args.skill:
            print("Error: --squash requires a skill name", file=sys.stderr)
            sys.exit(1)
        ok = squash_skill(args.skill, bank)
        sys.exit(0 if ok else 1)

    skills_to_compile = []

    if args.all:
        all_skills = get_all_skills(bank)
        skills_to_compile = list(all_skills.keys())
    elif args.group:
        group_skills = get_skills_in_group(bank, args.group)
        if not group_skills:
            print(f"Error: group '{args.group}' not found or empty", file=sys.stderr)
            sys.exit(1)
        skills_to_compile = list(group_skills.keys())
    elif args.skill:
        skills_to_compile = [args.skill]
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Compiling {len(skills_to_compile)} skill(s)...")
    success = True
    for skill_name in skills_to_compile:
        if not compile_skill(skill_name, bank, profile=args.profile, dry_run=args.dry_run, diff_mode=args.diff):
            success = False

    if not args.dry_run and not args.diff and success:
        save_snapshot(bank)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
