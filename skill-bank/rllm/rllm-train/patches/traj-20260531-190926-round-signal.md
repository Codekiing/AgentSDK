---
id: traj-20260531-190926-round-signal
target_section: round-signal
action: append
description: 修复 Phase 6.5 session_id 获取：从 .current_session 文件读取（由 post_tool hook 写入），替代不存在的 TRAJ_SESSION_ID 环境变量
status: proposed
source: trajectory-analysis
source_sessions: ["run_1780247306", "run_1780248991", "run_1780249675", "run_1780250271"]
---

### session_id 获取方式修正

**重要**: Claude Code 不提供 `TRAJ_SESSION_ID` 环境变量。Hooks 通过 stdin JSON 获取 session_id。
`traj_opt/hooks/post_tool.py` 已修改为每次工具调用时将 session_id 写入 `.current_session` 文件。

Phase 6.5 步骤 2 替换为以下三层 fallback:

```python
import os
from pathlib import Path

def get_session_id() -> str:
    """获取当前 Claude Code session_id，三层 fallback。"""
    
    # 方法 1 (首选): 从 .current_session 文件读取 (post_tool.py hook 写入)
    markers = [
        Path("traj_opt/output/rllm/.current_session"),
        Path("traj_opt/output/traj/.current_session"),
        Path("traj_opt/output/meta/.current_session"),
    ]
    for marker in markers:
        if marker.exists():
            sid = marker.read_text().strip()
            if sid and sid != 'unknown':
                return sid
    
    # 方法 2: 快照差分 (需要 Phase 0 记录了 existing_sessions)
    raw_dir = Path('traj_opt/output/rllm/raw')
    if raw_dir.exists() and existing_sessions:
        current = set(os.listdir(raw_dir))
        new_sessions = current - existing_sessions
        if new_sessions:
            return sorted(new_sessions)[-1]
    
    # 方法 3 (最后手段): 最近修改的 session 目录
    if raw_dir.exists():
        dirs = sorted(
            raw_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if dirs:
            return dirs[0].name
    
    return "unknown"
```

Phase 6.5 步骤 2 的 Bash 调用改为:

```bash
python3 -c "
import os
from pathlib import Path
# 方法 1: .current_session 文件
marker = Path('traj_opt/output/rllm/.current_session')
if marker.exists():
    sid = marker.read_text().strip()
    if sid:
        print(sid)
        exit(0)
# 方法 2: 最近修改的目录
raw = Path('traj_opt/output/rllm/raw')
if raw.exists():
    dirs = sorted(raw.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if dirs:
        print(dirs[0].name)
        exit(0)
print('unknown')
"
```

**注意**: `.current_session` 文件由 `traj_opt/hooks/post_tool.py` 在每次工具调用时自动写入。
只要 hooks 已配置且 post_tool.py 包含 `write_current_session()` 调用, 此方法 100% 可靠。

