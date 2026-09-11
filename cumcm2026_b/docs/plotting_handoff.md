# 绘图协作交接

这份仓库只保留画图和复现实验所需的轻量内容。所有路径都相对于 `cumcm2026_b/`。

## 最直接的数据入口

### 问题一、二

- `artifacts/q12_paper/metrics.json`：定位多边形、最小包围圆、候选观测点和交叉角灵敏度的完整数值。
- `artifacts/q12_paper/q1_localization.png`：问题一当前参考图。
- `artifacts/q12_paper/q2_candidate_region.png`：问题二候选区域当前参考图。
- `artifacts/q12_paper/q2_angle_sensitivity.png`：交叉角灵敏度当前参考图。
- `build_q12_assets.py`：上述数据与图片的生成脚本。

建议优先重画：定位扇区交集、候选观测点热力/散点图、交叉角与定位区域直径曲线。

### 问题三、四

- `artifacts/official_practice/combined_validation.json`：官方练习模式的汇总指标。
- `artifacts/official_practice/q3_run_001/client.jsonl`：问题三逐动作轨迹。
- `artifacts/official_practice/q4_run_001/client.jsonl`：问题四逐动作轨迹。
- `artifacts/official_practice/*/summary.json`：单次运行摘要。
- `artifacts/benchmarks/q4_optimized_mixed_500.json`：混合环境 500 回合主结果。
- `artifacts/benchmarks/q4_best_500.json`：当前最佳配置 500 回合摘要。
- `artifacts/benchmarks/` 其余 JSON：参数消融结果，文件名编码了所改参数。

建议优先重画：Q3/Q4 轨迹图、累计清除数—虚拟时间曲线、参数消融对比、完成率与平均用时/路程的权衡图。

### 代理数据质量

- `artifacts/surrogate_v5_180deg/generation_report.json`：数据规模、质量、盲测和官方练习结果总览。
- `artifacts/surrogate_v5_180deg/controller_blind_2000.json`：2000 回合控制器盲测。
- `artifacts/surrogate_v5_180deg/teacher_blind_*.json`：教师策略盲测。

`surrogate_v4_10m` 使用了错误的 90° 方向覆盖假设，已明确作废，不应进入论文图表。

## 关键字段

- `source_clear_rate`：干扰源清除率。
- `episode_success_rate` / `complete_case_rate`：整局成功率。
- `virtual_time_*_s` / `mean_episode_mean_time_s`：虚拟任务用时。
- `mean_travel_m`：平均航程。
- `mean_measure_actions`：平均测量动作数。
- `mean_competitive_ratio`：相对近似 oracle 的时间比。
- `failures` / `failure_seeds`：失败样本和随机种子，适合单独诊断。

## 复现

```powershell
cd cumcm2026_b
python -m unittest discover -s tests -v
python build_q12_assets.py
```

大体积 NPZ 和官方模拟器没有提交；画图请直接读取上述 JSON/JSONL。若需要重新跑官方练习，需由代码侧提供本地模拟器环境。
