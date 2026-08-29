# 防松线合成数据训练消融 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在相同步数和真实val下完成YOLOv8s-P2真实control与25%防松线合成批次的公平消融。

**Architecture:** 新增纯Python混合COCO与确定性批次规划逻辑，训练时用Ultralytics 8.2.40自定义单卡DataLoader执行3-real+1-synthetic批次。合成资产先通过现有严格审计，再进入Git外训练目录。

**Tech Stack:** Python 3.11、pytest、PyTorch 2.7.1、Ultralytics 8.2.40、COCO/YOLO。

---

### Task 1: 混合COCO与批次计划

**Files:**
- Create: `ml/src/crrc_vision/synthetic_ablation.py`
- Create: `ml/tests/test_synthetic_ablation.py`

- [ ] 先写失败测试，覆盖ID重映射、合成仅进train、全图占比不超过30%、39个等步数batch及每批最多1张合成图。
- [ ] 运行`pytest ml/tests/test_synthetic_ablation.py -q`确认因接口缺失而失败。
- [ ] 实现最小确定性合并和批次规划函数。
- [ ] 重跑聚焦测试并确认通过。

### Task 2: 训练入口

**Files:**
- Create: `ml/src/crrc_vision/ultralytics_ablation.py`
- Create: `ml/scripts/build_synthetic_ablation_coco.py`
- Modify: `ml/scripts/train_p2_high_accuracy.py`
- Modify: `ml/tests/test_p2_training.py`

- [ ] 先写失败测试，要求只有显式合成消融模式才能启用25%批次采样。
- [ ] 实现Git外合并脚本、严格审核调用和Ultralytics单卡DataLoader适配。
- [ ] 在training manifest记录真实/合成数量、配比上限、审核包哈希和批次计划。
- [ ] 运行聚焦测试与全量测试。

### Task 3: 运行与评价

**Files:**
- Create: `docs/validation/2026-08-29-synthetic-training-ablation.md`
- Modify: `PROJECT_STATUS.md`

- [ ] 用batch 4完成control与synthetic各20 epoch。
- [ ] 从两个`results.csv`和best checkpoint读取同一真实val指标及SHA-256。
- [ ] 验证formal truth哈希未变、sealed-test未访问、训练输出位于Git外。
- [ ] 写入客观结论，运行全量测试和代码复审后提交。
