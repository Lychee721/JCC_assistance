# Vision Data

目录说明：

- `raw/screenshots/`
  存放你的游戏回放截图。
- `annotations/slot_labels.template.csv`
  标注模板。你需要把每张截图里每个槽位的真实类别写进去。
- `datasets/classifier_slots/`
  训练脚本使用的最终分类数据集目录，按 ImageFolder 结构组织。
- `artifacts/classic_cnn/`
  训练输出的权重、类别映射和指标。

类别建议：

- `bf_sword`
- `chain_vest`
- `frying_pan`
- `giants_belt`
- `needlessly_large_rod`
- `negatron_cloak`
- `recurve_bow`
- `sparring_gloves`
- `spatula`
- `tear_of_the_goddess`
- `empty_slot`

建议：

1. 先用合成数据预训练。
2. 再用你自己的真实回放截图裁切结果做微调。
3. 真实截图每类至少 80 到 150 张，效果会明显更稳。
