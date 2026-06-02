================================================================================
  rllm-train 训练链路问题分析
  范围: /rllm-train 调用 → Phase 0~6 → 多轮调参循环
  依据: 本次 8 次 smoke run 的实际执行过程 + 编译后的 skill 内容
================================================================================

================================================================================
一、问题全景图
================================================================================

本次训练的预期链路:
  /rllm-train → Phase0(分级) → Phase1(clarify) → Phase2(config)
  → Phase3(run) → Phase4(monitor) → Phase5(analyze) → Phase2(tune) → ... → Phase6(report)

实际执行链路:
  /rllm-train → Phase1 ✓ → Phase2 ✓ → Phase3 ✓ → Phase4(C8熔断) → Phase2(tune) ✓
  → Phase3 ✓ → Phase4(C7/C9熔断)
  → [用户中断: "参考文档修复问题"]
  → [此后 8 次 smoke run 全部手动执行, 编排层不再参与]
  → 手动: 写config → 启动训练 → 启动monitor → 读日志汇总 → 分析 → 再写config → ...

链路在第一次调参熔断后即断裂，此后完全依赖手动操作。

================================================================================
二、问题拆解: 5 个层面
================================================================================

--------------------------------------------------------------------------------
问题 1: 编排层自动续跑不生效
--------------------------------------------------------------------------------

表现:
  - 首轮训练 C8 熔断后, rllm-train 按 fix_preset 调用了 rllm-config 生成 tuned config
  - 但 tuned config 生成后, 没有自动调用 rllm-run → rllm-monitor
  - 需要用户说"继续"才能推进
  - 004-auto-continue-after-config.md 已编译到 SKILL.md, 但问题仍存在

根因分析:

  a) "Skill 调用后必须立即停止" 与 "自动续跑" 的矛盾
     rllm-train 规则 5: "调用 Skill 后当轮响应必须立即结束"
     004 补丁: "编排者必须在下一次可执行回合自动进入后续 Phase"
     
     这两个规则之间的衔接依赖一个前提: 子 skill 执行完毕后,
     rllm-train 能再被调度到, 并且能识别当前 Phase 和下一步。
     
     但实际的执行模型是:
       用户说"继续" → rllm-train 被注入 → 执行一个 Phase → 调用子 Skill → 停止
       下一轮 → 系统注入新的 skill 内容(可能是子 skill 而非 rllm-train)
     
     问题: 子 skill 执行完毕后, 下一轮注入的是"无 skill"状态,
     rllm-train 没有被重新激活, 所以自动续跑规则没有执行者。

  b) 子 skill 完成信号的不可检测性
     rllm-train 的自动续跑规则要求它 "检测" 子 skill 的完成标志:
       - 检测 config.json 是否生成 → 调用 rllm-run
       - 检测 CIRCUIT_BREAK 信号 → 调用 rllm-config (tuned)
     
     但 rllm-train 调用子 Skill 后就停止了, 下一轮它可能不活跃。
     即使活跃, 它也需要 Read 文件来"检测", 但规则说"禁止内联执行",
     这形成了一个死循环: 必须先检测才能路由, 但检测本身被当作内联执行禁止。

  c) 会话压缩 (context compaction) 丢失状态
     训练周期长达 30+ 分钟, 上下文会触发压缩。
     压缩后 rllm-train 丢失了:
       - 当前 run_id
       - 当前 Phase
       - circuit_break_count
       - 历史 reward 趋势
     
     training_state.json 被设计来持久化这些状态, 但在本次执行中:
       - rllm-train 在 Phase 转换时没有可靠写入 training_state.json
       - 恢复时也没有从 training_state.json 重建状态

修复方向:
  - 将 "自动续跑" 从 "编排者识别并调用下一个 Skill" 改为 "编排者写入指令文件,
    下一轮由系统/用户触发执行"
  - 或者在每个子 skill 的 SKILL.md 末尾增加 "完成后调用 rllm-train 继续"
  - training_state.json 需要在每个 Phase 完成后原子写入, 并在每轮开始时读取

--------------------------------------------------------------------------------
问题 2: Monitor 不可靠
--------------------------------------------------------------------------------

表现:
  - toolfix smoke: monitor 运行但用户反馈"monitor不会每个step自动输出"
  - lr=0 smoke: monitor 被 kill (status=killed)
  - lr=1e-6 smoke: monitor 启动了但日志更新后 monitor 没有输出 step 1-6
    需要手动 Bash tail 才能看到已存在的 MONITOR_STEP 行
  - 多次出现: 训练已完成 (Training Report 已写入日志) 但 monitor 仍在等待

根因分析:

  a) Monitor 工具本身的限制
     当前实现: Monitor(persistent=false, timeout=3600000)
     问题:
       - Monitor 工具会被系统 kill (lr=0 案例)
       - 没有自动重启机制
       - timeout=3600000(1h) 看起来够, 但 7B 训练可能超过 1h

  b) Python 轮询脚本的脆弱性
     ```python
     pos = 0  # 或 os.path.getsize(path)
     while True:
         f.seek(pos)
         data = f.read()
         pos = f.tell()
     ```
     问题:
       - 训练进程在写训练日志时用的是 `>>` 重定向
       - 如果训练进程因为 OOM/崩溃等原因, 日志文件的写入缓冲区没有 flush,
         Python 脚本可能读到半行或读不到新数据
       - `pos = f.tell()` 在文件被外部追加时可能不可靠
       - lr=1e-6 案例中, 日志有 Step 1-6 但 monitor 没输出:
         可能原因是文件从开头重新写了(两次 trainer 初始化),
         pos 指针已经超过了新写入的位置, 导致不读取

  c) 去重逻辑在日志重置时失效
     ```python
     seen.add(key)  # key = "Step 6/?"
     ```
     如果日志因训练进程重启而被覆盖, pos > 新文件大小,
     f.seek(pos) 回到文件尾, 新写入的 step 被跳过。
     并且 seen 集合中已有 Step 1-6, 即使读到也会被去重跳过。

  d) Monitor 与编排层脱节
     rllm-monitor 的 SKILL.md 要求:
       "收到 Training Report 时, 停止 Monitor task"
     但 Monitor 被 kill 时, rllm-monitor 没有收到通知,
     编排层也不知道 monitor 已死。

修复方向:
  - Monitor 方案改为: 短周期 CronCreate 定时 tail + Read, 而非长周期 persistent Monitor
  - 或者: Monitor 被 kill 后, 编排层有兜底检测 (如每 2 分钟检查一次日志)
  - 日志文件的 inode 检测: 如果 inode 变化说明文件被重建, 重置 pos=0
  - 训练进程写入的 MONITOR_STEP 行应该包含一个自增序号,
    monitor 按序号去重而非按 "Step N/?" key 去重

--------------------------------------------------------------------------------
问题 3: 训练进程生命周期不可观测
--------------------------------------------------------------------------------

表现:
  - lr=0 训练: 进程在 step 10 的 "generating 4 trajectories..." 处静默退出
    - 没有 Traceback
    - 没有 OOM 错误
    - 没有 Training Report
    - 后台 task 的 .output 文件为空 (stdout/stderr 重定向到了 training_log.txt)
  - 所有训练: 后台 task 完成通知有时延迟很久
  - 训练启动确认: "sleep 10 && head -5" 不能确保训练正常运行

根因分析:

  a) 错误输出与正常输出混在一起
     训练命令:
       python -m rllm_train.run_training ... > training_log.txt 2>&1
     
     问题: stderr 合并到 stdout, 如果进程被系统 kill (SIGKILL),
     不会有任何输出。如果是 Python 异常, 异常信息混在训练日志中,
     只能靠 grep Traceback 来发现。

  b) 没有心跳机制
     训练进程可能卡在某个 step 的 rollout (7B 模型生成慢),
     也可能已经崩溃。编排层无法区分, 只能等。
     
     当前 MONITOR_STEP 是最接近心跳的东西, 但它只在 step 完成时写入。
     如果 step 中间崩溃, 没有任何中间输出。

  c) 后台 task 完成通知的不可靠性
     Bash(run_in_background=true) 在进程退出后会发通知,
     但如果训练被 SIGKILL, 通知中只有 exit_code,
     没有最后几行日志, 无法诊断。

修复方向:
  - 分离 stderr: `2> training_stderr.txt`, 专门用于异常检测
  - 训练进程增加 per-trajectory 心跳写入 (如 trajectory 1/4 done 的日志行)
  - rllm-run 启动后, 增加一个 watchdog: 如果 N 分钟无日志增长且无新 MONITOR_STEP, 判定为卡死/崩溃
  - 使用 `timeout` 命令包裹训练进程, 防止无限挂起

--------------------------------------------------------------------------------
问题 4: 数据传递与状态管理断裂
--------------------------------------------------------------------------------

表现:
  - run_id 在多轮调参中靠手动跟踪, 没有自动传递
  - seed 从 44 到 49, 每次手动递增, 没有自动管理
  - 熔断后的 fix_preset 传递到了 rllm-config, 但调参结果没有写回 training_state.json
  - config.json 中的 task_id/skill_package_id 每次手动填写

根因分析:

  a) training_state.json 的被忽视
     rllm-train 设计了这个文件, 但在本次执行中:
       - 没有在 Phase 转换时写入
       - 没有在恢复时读取
       - 路径: `rllm_train/output/training_state.json`
     
     检查: 这个文件甚至可能不存在。

  b) Phase 间数据传递靠 "对话上下文" 而非持久化文件
     Phase 2 → Phase 3: "config.json 文件路径"
     -> 实际: 编排者在对话中说 "run_id=xxx", 下一轮 rllm-run 从对话中提取
     -> 对话压缩后, 这个信息丢失

  c) 熔断信号的不可靠传输
     CIRCUIT_BREAK 信号的载体是 "monitor 的输出文本",
     编排者需要 parse 这个文本。
     
     问题:
       - 如果 monitor 被 kill, 没有 CIRCUIT_BREAK 输出
       - 编排者需要在对话中 parse monitor 输出, 但如果编排者不活跃, parse 不执行
       - analysis.json 被设计为兜底, 但熔断时 analysis.json 是由 monitor 写入的,
         如果 monitor 被 kill, analysis.json 也不存在

修复方向:
  - training_state.json 必须成为唯一真相来源, 对话上下文只是缓存
  - 每个 Phase 完成后原子写入 training_state.json:
      { "current_phase": "config", "run_id": "xxx", "round": N, ... }
  - 每个 Phase 开始时先读取 training_state.json 重建状态
  - 熔断信号改为写入 analysis.json + 设置 training_state.json 中的 flag,
    不再依赖对话文本解析

--------------------------------------------------------------------------------
问题 5: 工程模式完全绕过编排层
--------------------------------------------------------------------------------

表现:
  - 从 "参考文档修复问题" 开始, 所有的代码修改、smoke run 都是手动执行
  - 8 次 smoke run 的 config.json 手动 Write
  - 训练手动 Bash 启动
  - 监控手动 Monitor 工具启动
  - 结果手动 Python 汇总
  - 编排层 (rllm-train) 从第一次熔断后就再没被调用过

根因分析:

  这不是 bug, 而是设计上的空白:
    rllm-train 编排的是 "正常训练循环" (clarify→config→run→monitor→analyze→tune)
    但不覆盖 "工程修复循环" (diagnose→modify code→smoke→verify→iterate)
  
  当 C7/C9 熔断且 fix_preset=diagnose 时, 编排层的正确行为是:
    → 走完整 Phase 5 (rllm-analyze)
    → rllm-analyze 触发专家文档参考条件
    → 输出诊断结论和修复方向
    → 回到 Phase 2 调参
  
  但 rllm-analyze 给出的建议是 "参考 verl-reward-slow-troubleshooting-guide.md 做分层诊断",
  后面的代码修改、smoke 验证等步骤没有对应的 Phase/子 skill。

修复方向:
  - 新增 rllm-diagnose skill: 专门处理需要代码修改的工程诊断
  - 或者在 rllm-analyze 的输出中区分 "调参建议" 和 "工程修复建议",
    后者触发一个工程修复循环: fix→smoke→verify
  - 短期: 在 diagnose fix_preset 时, rllm-train 明确告知用户
    "当前问题需要工程修复, 编排层暂停, 请手动修复后说'继续'"

================================================================================
三、优先级与修复建议
================================================================================

优先级 1 (影响所有训练): Monitor 可靠性
  - 改长周期 Monitor 为 CronCreate 定时 tail, 避免被 kill 后无输出
  - 增加 inode 检测, 日志重建时重置 pos
  - 增加 Monitor 被 kill 后的兜底 tail 检查

优先级 2 (影响自动循环): 编排层状态持久化
  - training_state.json 在 Phase 转换时原子写入
  - rllm-train 每轮开始时从 training_state.json 恢复
  - 熔断信号走文件通道 (analysis.json) 而非对话文本

优先级 3 (影响调参循环): 自动续跑修复
  - 子 skill SKILL.md 末尾增加 "完成后调用 Skill(rllm-train, args=next_phase)"
  - 或改为 "指令文件" 模式: 编排者写 next_phase 到文件, 用户下一轮触发

优先级 4 (影响工程修复): 工程修复子流程
  - 新增 rllm-diagnose → rllm-fix → rllm-smoke 子 skill 链
  - 或在 diagnose preset 时明确终止编排, 等待用户手动修复后恢复

================================================================================
四、当前链路 vs 设计意图对比
================================================================================

组件              设计意图                        实际表现                      差距
--------------------------------------------------------------------------------
rllm-train        全自动闭环, 无需用户干预         需要用户说"继续"才能推进      自动续跑机制名存实亡
rllm-config       生成配置, 传递 run_id            配置生成后停在摘要            没有自动触发 Phase 3
rllm-run          启动训练, 返回 task_id           训练启动后停在确认            没有自动触发 Phase 4
rllm-monitor      持续监控, 异常熔断               频繁被 kill, 丢 step 数据    可靠性不足
rllm-analyze      分析结果, 写 analysis.json       实际很少被调用                diagnose 触发后跳过
training_state    跨 Phase 状态持久化              从未被写入/读取              完全未使用
CIRCUIT_BREAK     结构化熔断信号                   只在 monitor 文本中出现      不可靠的信号通道
Monitor 工具      持久流式监控                     被 kill, timeout             不适合长时间任务
Skill 工具        调用子 skill, 等待完成           调用后编排者停止             无法自动串联
