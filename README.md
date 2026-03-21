# JCC Assistance

Repository: <https://github.com/Lychee721/JCC_assistance>

中文 | [English](#english)

---

## 中文说明

### 项目简介

`JCC Assistance` 是一个面向《金铲铲之战》出装辅助场景的课程项目。这个仓库将 CNN 视觉识别、规则约束的装备合成逻辑、以及 LLM 解释层组合在一起，构成一条可以演示、评估、并继续扩展的混合系统链路。

核心流程为：

`截图 / 屏幕捕获 -> 装备栏裁剪 -> CNN 散件识别 -> 可合成装备枚举 -> 规则排序 -> LLM 解释`

项目的重点不是做一个成熟的商业产品，而是验证以下问题：

- 如何从真实游戏截图中稳定识别散件。
- 如何在本地 item graph 约束下枚举可合成装备。
- 如何让 LLM 负责解释与交互，而不是作为无约束的决策器。

### 功能概览

当前仓库包含：

- CNN 训练与评估脚本
- 基于截图和 replay 的视觉 demo
- 规则驱动的装备推荐引擎
- LLM 编排与解释模块
- FastAPI 服务接口
- 课程报告与实验图表

### 仓库结构

```text
app/                     API、推荐引擎、LLM 编排、视觉推理
configs/                 配置文件
data/                    原始数据、处理产物、训练 artifact
docs/                    设计文档与演示说明
examples/                请求 / 响应示例
prompts/                 LLM prompt 模板
references/              参考资料与笔记
Report/                  课程论文与 LaTeX 文件
schemas/                 JSON Schema
scripts/                 数据处理、训练、评估、demo 脚本
sql/                     存储结构
tests/                   单元测试
```

### 快速开始

#### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install -r requirements-train.txt
```

#### 2. 构建装备图和合成数据

```bash
python scripts/fetch_cdragon_snapshot.py --locale zh_cn
python scripts/fetch_cdragon_snapshot.py --locale en_us
python scripts/normalize_item_graph.py
python scripts/generate_synthetic_dataset.py
python -m unittest tests/test_recommendation_engine.py
```

#### 3. CNN 训练与评估

```bash
python scripts/build_synthetic_slot_dataset.py
python scripts/bootstrap_annotation_csv.py
python scripts/train_cnn.py
python scripts/evaluate_cnn.py
```

#### 4. 启动 API

```bash
uvicorn app.main:app --reload
```

可用接口：

- `GET /healthz`
- `POST /v1/recommend/items`
- `GET /v1/demo/scenarios`
- `GET /v1/demo/run/{scenario_id}`
- `POST /v1/recommend/replay-local`

Swagger 文档：<http://127.0.0.1:8000/docs>

#### 5. 运行 demo

CLI 推荐 demo：

```bash
python scripts/demo_cli.py --scenario ad_carry_stage_3
```

CNN replay demo：

```bash
python scripts/run_replay_demo.py --screenshot your_screenshot.png --target-champion main_carry --intent carry_ad --stage 4-1
```

实时 overlay demo：

```bash
python scripts/live_overlay_demo.py --use-llm
```

若需启用 LLM，请先配置对应的 API key，例如：

```bash
set GEMINI_API_KEY=your_key_here
```

### 核心文档

- [docs/architecture.md](docs/architecture.md)
- [docs/cnn-llm-pipeline.md](docs/cnn-llm-pipeline.md)
- [docs/data-sources.md](docs/data-sources.md)
- [docs/demo-guide.md](docs/demo-guide.md)
- [docs/modeling-focus.md](docs/modeling-focus.md)
- [docs/vision-training.md](docs/vision-training.md)
- [docs/roadmap.md](docs/roadmap.md)

### 外部资源

- Riot TFT Developer Docs: <https://developer.riotgames.com/docs/tft>
- CommunityDragon assets: <https://communitydragon.org>
- CommunityDragon raw snapshot: <https://raw.communitydragon.org/latest/cdragon/tft/en_us.json>
- tacticians-academy `academy-library`: <https://github.com/tacticians-academy/academy-library>

### 说明

- 当前项目的核心目标是“散件识别 -> 装备推荐 -> LLM 解释”这条完整链路。
- `completed_item` 和 `other_unknown` 这类开放 / 异质标签在当前版本中采用保守处理，后续可作为优化方向。
- 课程报告与图表主要位于 `Report/` 和 `data/vision/artifacts/` 目录下。

---

## English

### Overview

`JCC Assistance` is a coursework project for a Golden Spatula / TFT-style item assistant. It combines CNN-based vision recognition, a constrained item-graph recommendation layer, and an LLM explanation module into one end-to-end pipeline.

Core pipeline:

`screenshot / screen capture -> slot cropping -> CNN component recognition -> craftable-item enumeration -> rule-based ranking -> LLM explanation`

The repository is not intended as a polished commercial product. Instead, it is a structured hybrid system for studying:

- how to recognize item components from real gameplay screenshots,
- how to constrain recommendations with a local item graph,
- and how to use an LLM for explanation and interaction rather than unconstrained decision making.

### Features

The repository currently includes:

- CNN training and evaluation scripts
- screenshot and replay based demos
- a rule-based recommendation engine
- an LLM orchestration layer
- a FastAPI service
- the course report and experiment figures

### Repository Structure

```text
app/                     API, recommendation engine, LLM orchestration, vision inference
configs/                 Configuration files
data/                    Raw data, processed data, and training artifacts
docs/                    Design notes and demo guides
examples/                Request / response examples
prompts/                 LLM prompt templates
references/              References and notes
Report/                  Paper and LaTeX sources
schemas/                 JSON Schema files
scripts/                 Data processing, training, evaluation, and demo scripts
sql/                     Storage schemas
tests/                   Unit tests
```

### Quick Start

#### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-train.txt
```

#### 2. Build the item graph and synthetic recommendation data

```bash
python scripts/fetch_cdragon_snapshot.py --locale zh_cn
python scripts/fetch_cdragon_snapshot.py --locale en_us
python scripts/normalize_item_graph.py
python scripts/generate_synthetic_dataset.py
python -m unittest tests/test_recommendation_engine.py
```

#### 3. Train and evaluate the CNN

```bash
python scripts/build_synthetic_slot_dataset.py
python scripts/bootstrap_annotation_csv.py
python scripts/train_cnn.py
python scripts/evaluate_cnn.py
```

#### 4. Start the API

```bash
uvicorn app.main:app --reload
```

Available endpoints:

- `GET /healthz`
- `POST /v1/recommend/items`
- `GET /v1/demo/scenarios`
- `GET /v1/demo/run/{scenario_id}`
- `POST /v1/recommend/replay-local`

Swagger UI: <http://127.0.0.1:8000/docs>

#### 5. Run demos

CLI recommendation demo:

```bash
python scripts/demo_cli.py --scenario ad_carry_stage_3
```

CNN replay demo:

```bash
python scripts/run_replay_demo.py --screenshot your_screenshot.png --target-champion main_carry --intent carry_ad --stage 4-1
```

Live overlay demo:

```bash
python scripts/live_overlay_demo.py --use-llm
```

To enable the LLM path, set the corresponding API key first, for example:

```bash
set GEMINI_API_KEY=your_key_here
```

### Key Documentation

- [docs/architecture.md](docs/architecture.md)
- [docs/cnn-llm-pipeline.md](docs/cnn-llm-pipeline.md)
- [docs/data-sources.md](docs/data-sources.md)
- [docs/demo-guide.md](docs/demo-guide.md)
- [docs/modeling-focus.md](docs/modeling-focus.md)
- [docs/vision-training.md](docs/vision-training.md)
- [docs/roadmap.md](docs/roadmap.md)

### External Resources

- Riot TFT Developer Docs: <https://developer.riotgames.com/docs/tft>
- CommunityDragon assets: <https://communitydragon.org>
- CommunityDragon raw snapshot: <https://raw.communitydragon.org/latest/cdragon/tft/en_us.json>
- tacticians-academy `academy-library`: <https://github.com/tacticians-academy/academy-library>

### Notes

- The current project focuses on the full chain of component recognition, craft recommendation, and LLM explanation.
- Open or heterogeneous labels such as `completed_item` and `other_unknown` are handled conservatively in the current version and remain future work.
- The paper and experiment figures are mainly stored under `Report/` and `data/vision/artifacts/`.
