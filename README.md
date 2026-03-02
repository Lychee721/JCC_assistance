# 金铲铲出装助手

一个面向“散件识别 -> 可合成成装推荐 -> LLM 对话解释”的项目骨架。

当前仓库不是成品应用，而是一个可直接继续开发的底座，重点把 5 件事先整理好：

1. 数据源与生成链路
2. CNN 与 LLM 的职责边界
3. 统一的数据 Schema
4. 可运行的推荐引擎骨架
5. 后续训练、API、前端联调所需的配置与示例

已核对并纳入设计的外部基座：

- Riot TFT 开发文档: <https://developer.riotgames.com/docs/tft>
- CommunityDragon: <https://raw.communitydragon.org/latest/cdragon/tft/en_us.json>
- CommunityDragon 资源站: <https://communitydragon.org>
- tacticians-academy `academy-library`: <https://github.com/tacticians-academy/academy-library>

注意：

- `data/seed/*.example.json` 现在只保留为兜底示例。
- 当前默认运行时图谱已经改为由 `CommunityDragon latest` 快照生成。
- 当前 patch 的真实图谱输出文件是 `data/processed/item_graph.runtime.json`。

## 目录

```text
app/                     FastAPI + 推荐引擎 + LLM 编排骨架
configs/                 应用、模型、数据源配置
data/
  manifests/             外部数据源清单与路径规则
  processed/             运行时生成数据
  seed/                  演示用 seed 数据
docs/                    设计文档
examples/                请求/响应示例
prompts/                 LLM 系统提示词与用户模板
schemas/                 JSON Schema
scripts/                 拉取、标准化、生成数据集脚本
sql/                     推荐日志与会话存储表结构
tests/                   核心推荐引擎测试
```

## 核心流程

1. `scripts/fetch_cdragon_snapshot.py` 拉取 `CommunityDragon` 快照。
2. `scripts/normalize_item_graph.py` 从当前 patch 快照生成统一 `item_graph.runtime.json`。
3. `scripts/generate_synthetic_dataset.py` 基于 runtime component catalog 生成 CNN 训练清单。
4. `app/recommendation_engine.py` 根据散件背包算出所有可合成成装并打分。
5. `app/llm_orchestrator.py` 把结构化推荐结果组织成 LLM 可控回复。
6. `app/main.py` 提供 `/v1/recommend/items` API。

## 快速开始

```bash
python scripts/fetch_cdragon_snapshot.py --locale zh_cn
python scripts/fetch_cdragon_snapshot.py --locale en_us
python scripts/normalize_item_graph.py
python scripts/generate_synthetic_dataset.py
python -m unittest tests/test_recommendation_engine.py
```

如需跑 API：

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 作业展示

最稳的演示方式：

```bash
python scripts/demo_cli.py --scenario ad_carry_stage_3
```

如果要展示完整“CNN 识别回放界面 -> 推荐成装”链路，按这个顺序：

```bash
pip install -r requirements.txt
pip install -r requirements-train.txt
python scripts/build_synthetic_slot_dataset.py
python scripts/train_cnn.py
python scripts/run_replay_demo.py --screenshot 你的截图路径.png --target-champion 物理主C --intent carry_ad --stage 4-1
```

如果要展示接口：

1. 启动 `uvicorn app.main:app --reload`
2. 打开 `http://127.0.0.1:8000/docs`
3. 依次演示：
   `GET /healthz`
   `GET /v1/demo/scenarios`
   `GET /v1/demo/run/ad_carry_stage_3`
   `POST /v1/recommend/items`

详细演示话术见：

- [docs/demo-guide.md](/c:/Users/ASUS/Desktop/金铲铲助手/docs/demo-guide.md)
- [docs/modeling-focus.md](/c:/Users/ASUS/Desktop/金铲铲助手/docs/modeling-focus.md)
- [docs/vision-training.md](/c:/Users/ASUS/Desktop/金铲铲助手/docs/vision-training.md)

## 建议开发顺序

1. 先跑通规则推荐链路，不依赖 LLM。
2. 再接入截图识别，把 CNN 输出限定成结构化散件清单。
3. 最后接 LLM，只负责解释、追问、交互，不负责“发明配方”。

详细设计见：

- [docs/architecture.md](/c:/Users/ASUS/Desktop/金铲铲助手/docs/architecture.md)
- [docs/data-sources.md](/c:/Users/ASUS/Desktop/金铲铲助手/docs/data-sources.md)
- [docs/cnn-llm-pipeline.md](/c:/Users/ASUS/Desktop/金铲铲助手/docs/cnn-llm-pipeline.md)
