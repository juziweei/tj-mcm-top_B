# 主张—证据表

| 主张 | 证据/来源 | 强度 | 可用章节 | 风险 | 状态 |
|---|---|---|---|---|---|
| 官方 Q4 旧演练清除 16/16，355.19 s/源，墙钟 4.02 s | `artifacts/official_practice/q4_run_002/summary.json` | 官方单场景 | 实验 | 单场景不能代表总体 | evidence-backed |
| 固定 16 源基线 500 局源级/整局全清率均为 100% | `q4_final_base_fixed16_500.json` | 独立仿真 500 局 | 实验 | 代理分布偏差 | evidence-backed |
| 混合源基线 500 局源级 99.9846%、整局 99.8% | `q4_final_base_mixed_500.json` | 独立仿真 500 局 | 实验 | 代理分布偏差 | evidence-backed |
| balanced 在混合源上平均节省 69.93 s/真实源 | `q4_final_paired_mixed_500.json` | 配对 bootstrap | 实验、结论 | 依赖 0.97 风险预算 | evidence-backed |
| balanced 混合源相对改善均值 10.45%，95% CI 9.87%--11.00% | 同上 | 配对 bootstrap 10000 次 | 实验、结论 | 不能外推到任意官方分布 | evidence-backed |
| balanced 合并 1000 局共漏 6/14478 个源 | 两份 final balanced 500 报告 | 独立仿真 | 实验、局限 | 整局全清率仅 99.4% | evidence-backed |
| 两方位时最小包围圆中心的平均/P95 误差低于面积质心 | `q4_estimator_centers_fixed16_100.json` | 100 局、1530 个两方位观测 | 方法 | 20 m 命中率略低 | evidence-backed |
| 可行域半径不超过 75 m 时质心 20 m 内命中率约 90.97% | 同上 | 条件统计 | 方法 | 非确定性保证 | evidence-backed |
| 当前测量主要消耗在未知频道搜索 | `q4_action_split_*_100.json` | 100+100 局 | 诊断 | 分布相关 | evidence-backed |
| 神经策略不值得在当前两天赛程内训练 | 单决策反事实与上限分解 | 决策性推断 | 讨论 | 不是“神经网络永远无效” | plausible-inference |
| Q3 单局墙钟在 20 局中最大为 39.76 s，20/20 全清 | `artifacts/benchmarks/q3_final_20.json` | 独立仿真 20 局 | 神经可行性、限时 | 样本较小 | evidence-backed |
