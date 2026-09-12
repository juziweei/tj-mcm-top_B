# CUMCM 2026 B 题工作区

本目录用于研究和实现“无线电干扰源的快速自动定位与清除”。当前阶段先建立可验证的几何定位内核，再接入模拟器、在线规划与学习策略。

## 当前模块

- `docs/problem_analysis.md`：题意、数学对象、四问拆解与实现顺序。
- `src/cumcm_b/`：几何定位、信念状态、规划、代理仿真、官方协议客户端与数据生成主线。
- `tests/`：几何、仿真、规划、协议和训练数据的回归测试。
- `artifacts/q12_paper/`：问题一、二的绘图数据和当前参考图。
- `docs/q12_teammate_response.md`：按往年范文固定骨架整理的问题一、二数据、思路、方法和代码对应，可直接用于论文协作沟通。
- `artifacts/benchmarks/`：问题四的参数消融与最终评估摘要。
- `artifacts/surrogate_v5_180deg/`：代理数据生成质量与盲测摘要。
- `artifacts/official_practice/`：官方练习模式的运行摘要与客户端轨迹。
- `artifacts/neural_belief_v1/`：4.84M token 数据清单、审计、431 KB 模型和独立测试报告。
- `docs/plotting_handoff.md`：面向绘图协作者的数据入口、字段说明和建议图表。

## 运行测试

在本目录执行：

```powershell
python -m unittest discover -s tests -v
```

## 实现原则

1. 物理规则和定位几何由解析模型负责。
2. 学习模型只处理部分可观测条件下的跨频道调度和长期决策。
3. 所有学习策略必须与独立的非学习基线在同一测试分布、同一虚拟时间预算下比较。

## 问题 4 运行档位

- `conservative`：停止后验 0.998；500+500 独立仿真中的推荐可靠档。
- `balanced`：停止后验 0.97；源级清除率 99.9586%，混合源配对耗时改善 10.45%。
- `aggressive`：停止后验 0.95；只用于速度上界研究，未进入最终 500+500 验证。
- `source-efficient`：双风险高可靠档；独立 300+300 局源级清除率约 99.7%。
- `source-98`：双风险速度档；独立 300+300 局源级清除率约 98.98%。

官方运行器默认 `conservative`，并且只允许进入最新日志明确标记为演练的会话：

```powershell
python run_official_q4.py --profile conservative --robot-id <参赛队号> --simulator-data-dir <JammersSimulatorData路径> --log <日志路径> --summary <摘要路径>
```

演练比较时可将 `--profile conservative` 显式改成 `--profile source-efficient`（约 99.7% 源级可靠率）或 `--profile source-98`（独立代理仿真约 99.0% 源级可靠率的速度档）。性能、风险边界和神经网络可行性结论见 `docs/q4_optimization_report.md` 与 `docs/neural_distillation_experiment_contract.md`。

## 神经信念模型复现

```powershell
python generate_belief_dataset.py --output data/belief_v1 --train-episodes 12000 --val-episodes 1500 --test-episodes 1500 --workers 16
python audit_belief_dataset.py --data data/belief_v1 --output data/belief_v1/audit.json
python train_belief_model.py --data data/belief_v1 --output runs/belief_v1_gru128 --epochs 20 --batch-size 64
python evaluate_belief_model.py --data data/belief_v1 --checkpoint runs/belief_v1_gru128/best.pt --output runs/belief_v1_gru128/test.json
```

网络输入只含官方接口可得到的动作—观测历史，隐藏源状态只构造监督标签。独立测试表明其计算很快，但在近乎 100% 完整清除约束下没有形成足够的虚拟时间收益，因此暂不接管正式停止决策。

## 仓库体积说明

约 2.25 GB 的 NPZ 训练分片和约 700 MB 的官方模拟器运行时不会提交到 GitHub。仓库保留数据结构、统计摘要、验证报告、评估结果和可复现实验脚本，足够用于画图与论文协作。
