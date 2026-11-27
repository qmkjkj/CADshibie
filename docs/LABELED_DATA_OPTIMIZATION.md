# 恢复带标签数据后的优化记录（2026-11-26）

> 后续第二轮已取消每类 15 个 ROI 的额外截断，历史口径回归 PQ 32.856%→32.923%；见 [候选优化记录](ROUND2_CANDIDATE_OPTIMIZATION.md)。本文保留第一轮训练与冻结验收结果。

上传目录中的旧代码、旧模型、预测、提交包、DXF 项目和无标签旧 dev 已按用户要求清除。保留位置为 `人工智能比赛/datasets/`：

| 内容 | 数量 |
|---|---:|
| `train_final/` PNG/SVG 配对 | 4000 |
| `test_final/` PNG/SVG 配对 | 1000 |
| `test_json/` 旧环境导出的评估标签 | 1000 |
| `train_json/` 本轮从原 SVG 转出的训练标签 | 4000 |
| 原始 manifest | 2 |
| 数据集说明 README | 1 |

清理前比较了压缩包与已解压数据的 10002 个 PNG/SVG/CSV 文件，SHA256 全部相同；清理后再比较一次，无缺失、无变化。旧评估标签的抽样 20 份与当前 SVG 转换结果完全相同。其他旧 GT 子集都是完整 `test_json` 的重复副本；`train_subset` 中的 SVG 均能在原始训练/评估目录找到。清理记录位于 `_work/uploaded_data_preservation.json`，大体积数据不加入 Git。

## 数据隔离

联合检查 4000 张训练图、1000 张评估图和新 dev 的 809 张图：

- 98 个精确图像组同时包含训练与评估图，涉及 109 张评估图；另有近似图候选。
- 仅按精确图像与 dHash 候选联合分组后，可用于图像隔离评估的图有 824 张。
- 原始 manifest 提供 `source` 和 `stem`（例如 `test-00/0003-0010`）。加入“来源＋首段页编号”作为保守同源候选边，并与图像重复边联合取连通组后，得到更严格的来源隔离：训练 1020、训练验证 543；评估 calib 42、hold 53、final 83。其余数据保留，排除出本轮新增训练与来源隔离评估。

页编号前缀是保守来源假设，不表示这些裁片内容完全一样。跨 source 的同源关系仍可能被图像相似性漏掉。历史 YOLO/ROI/门控权重的全部训练暴露范围也未重新核实；新划分能隔离新细节头的训练，不能抹去旧模型历史。因此应报告“固定旧模型下的新增量”，不能把来源隔离 PQ 称为整个系统完全未见数据的泛化成绩。

固定划分与审计在 `_work/labeled_data_overlap.json`、`_work/image_grouped_splits.json`、`_work/source_grouped_splits.json`。新训练读取后者的副本 `_work/clean_splits.json`，不会直接扫描整个训练目录来纳入已排除页面。

## 已完成的带标签小规模消融

使用历史 calib 的前 100 张，冻结相同权重和分支候选，严格 1000px/PIL、IoU>0.5、累计 micro PQ：

| 路径 | PQ | TP / FP / FN |
|---|---:|---|
| 历史融合逻辑 | 28.453% | 236 / 215 / 450 |
| 来源身份隔离 | 28.361% | 235 / 215 / 451 |
| 来源隔离＋canonical 融合栅格 | 27.834% | 236 / 229 / 450 |
| 来源隔离＋原图平移精修 | 26.915% | 224 / 226 / 462 |

这些是同一小子集成绩，不可与历史全量 24.348% 相比。三个改动均未通过这轮验收，因此默认保留历史融合身份、旧栅格、关闭平移精修；实验开关仍可用。原图墨迹的拟合改善可能破坏评分几何，尤其该子集 class 18 的 TP 从 7 降至 1，不能依赖“更贴图像”验收。

历史兼容路径已逐文件验证与 `a9bef5f` 原 `merge_final` 在该 100 张上的输出相同。新增阶段快照与重放工具允许复用固定网络候选；推理阶段读取 PNG，标签仅用于离线评估。

另将评分器栅格过程改为每个实例一次 PIL 绘制，避免每个图元分配整张画布。20 张真实评估图的 10006 个图元与旧实现逐像素相同，100 张评估的 TP/FP/FN/ΣIoU 完全相同；这是等价加速，不改变评分规则。

## 新训练分支

新增原分辨率残差细节头，复用冻结 YOLO 与三套 ROI 粗概率：

- 新头输入 RGB＋目标实例的粗概率；监督直接使用 SVG 的正式评分栅格，不使用 PNG 骨架或 dev 预测作为标签。
- 原图 320px 重叠分块，不缩小细节标签；训练数据以小文件流式读取，不一次载入全部裁片。
- 训练使用真实检测框，并少量补充 GT 框正例；裁片验证只使用真实检测框。未匹配框包含明确难负例，歧义匹配跳过。
- 最后一层零初始化，初始输出复现 coarse；回接冻结旧 ROI 筛选决策，空细化结果回退原几何。类别和实例身份由旧系统提供。
- 实际选取来源隔离 train 中 800 页、val 中 120 页。得到 9660 个训练裁片（7444 正、2216 负）和 1043 个验证裁片（765 正、278 负）；固定随机取其中 500 个验证裁片选 checkpoint。裁片严格矢量指标仅作代理，不能当成整页 PQ。

训练 10 轮，batch=8、base=16、AdamW、lr=0.0005、seed=73。仅按训练侧固定验证裁片代理选中 epoch 7：coarse proxy=19.734%、detail proxy=35.625%。没有用 test/calib/hold/final 选择 epoch 或逐类开关。第一次训练期间曾在 calib 观察 epoch 1 的整页结果（29.911%→37.204%）；此后没有据此改训练参数，最终仍按预定的训练侧验证规则选模。

## 冻结后的整页验收

以下均使用同一权重、阈值、完整 hybrid 流程和严格 1000px/PIL、IoU>0.5、累计 micro PQ。legacy400 融合、legacy 身份、关闭几何平移保持一致。每组的 baseline 与 detail 使用完全相同图纸，三个子集之间没有来源候选连通组重叠。

| 数据范围 | 页数 | baseline PQ | detail PQ | 增量（百分点） | baseline TP/FP/FN | detail TP/FP/FN |
|---|---:|---:|---:|---:|---|---|
| calib | 42 | 29.911% | 37.801% | +7.890 | 151/141/257 | 180/105/228 |
| hold | 53 | 21.794% | 31.558% | +9.764 | 149/226/410 | 201/169/358 |
| final，方案冻结后评估 | 83 | 19.810% | 28.760% | +8.950 | 225/381/746 | 315/284/656 |
| 三组累计（描述性汇总） | 178 | 22.589% | 31.537% | +8.948 | 525/748/1413 | 696/558/1242 |

不能把 42 页的 37.801% 与历史全量 1000 页的约 24% 相比。178 页累计也不是独立 final 估计：它包含已观察的 calib。全量历史口径另行记录。

全量同 1000 张历史口径对照已完成，所有图片使用相同冻结权重与配置：

| 路径 | PQ | SQ | RQ | TP / FP / FN |
|---|---:|---:|---:|---|
| 本机旧基线 | 24.359% | 68.099% | 35.769% | 2188 / 2717 / 5141 |
| 默认原分辨率细节头 | **32.856%** | 71.621% | 45.875% | **2794 / 2058 / 4535** |

净增 **8.497 个 PQ 百分点、606 TP**，FP 减少 **659**。历史文档的 24.348% 与本机重跑的 24.359% 接近但不完全相同（历史 TP/FP/FN=2186/2720/5143）；未定位该微小差异的具体运行环境来源，因此对照使用本机同口径基线。新结果由已完成的 178 页和同权重剩余 822 页合成，1000 页无异常、全部 schema 校验通过。822 页批次 workers=3，推理 762.561 秒；各小批次启动和耗时不混称单次全量运行。

全量 1000 页只作历史口径对照，不能绕过前文的旧模型历史暴露与同源重叠限制。完整机器可读记录见 [NATIVE_DETAIL_RESULTS.json](NATIVE_DETAIL_RESULTS.json)，详细逐页输出位于 `_work/native_detail_eval/all/pred/`。

按候选来源连通组配对 bootstrap 5000 次、每次重新累计 TP/FP/FN/ΣIoU，增量的 95% percentile 区间：calib **[4.883,11.795]**、hold **[7.701,11.767]**、final **[7.217,10.645]** 个百分点。三组分别有 39/48/69 个来源候选组。区间只描述固定模型和现有分组下的抽样不确定性，不能消除旧模型历史暴露、未识别同源关系和稀有类别样本少的问题。

final 原有 225 个 GT 匹配保留 220 个，损坏 5 个，新增 95 个，净增 90 TP；FP 从 381 降至 284。calib/final 各类 PQ 均未下降；hold 的 class 22 损失 1 TP、PQ 降 2.217 个百分点，未按该结果调类开关。匹配诊断使用正式评分的同类别、阈值屏蔽、带虚拟节点的 Hungarian 一对一匹配，不直接比较两种预测的 instanceId。

## 默认接线与复现

三组验收后已将 `weights/native_detail.pt` 接入默认 hybrid，并加入模型清单；总审计参数量从 23,077,263 增至 **23,561,566**（新增头 state-dict 元素 484,303）。旧权重保留。

```bash
python infer.py --input <PNG目录> --output results_cad
python infer.py --input <PNG目录> --output baseline --disable-native-detail --roi-max-instances 15
python tools/evaluate.py --gt-dir <同一组真值目录> --pred-dir results_cad --json-out eval.json
```

也可显式使用 `--native-detail-weights <权重>`。其他引擎不会自动启用新头。推理不读取 SVG/JSON 标签；数据集构建、选模、评分工具独立于推理。现有 809 张 dev 不参与新增训练或选模，本轮没有为优化而重跑 dev。

冻结权重 SHA256：`076cfcb059f4baff57787b41d3b16a3e10244ae2260a40cccee918c655765ef6`。划分 SHA256：`4b3978fcb72b6a8ff62d3d4dd94c77ae42bb852112d6dd9995228c6a67457593`。训练入口：

```bash
python tools/build_native_detail.py --data-root 人工智能比赛/datasets --splits _work/clean_splits.json --out _work/native_detail_data --train-pages 800 --val-pages 120 --tile 320
python tools/train_native_detail.py --data _work/native_detail_data --out _work/native_detail.pt --epochs 10 --batch-size 8 --base 16 --validation-crops 500
```

划分生成器是 `tools/audit_dataset_overlap.py`、`tools/build_splits.py`、`tools/build_clean_splits.py`；最后一个需传 `--data-root 人工智能比赛/datasets` 以启用 manifest 来源家族。中间目录应放在 `_work`，不能当成推理输入。元数据保留所选图纸和粗模型哈希，checkpoint 保留训练数据索引 SHA256。

本机证据：`_work/native_detail.training.json`、`_work/native_detail_freeze.json`、`_work/native_detail_eval/{calib,hold,final}/`、`_work/native_detail_eval/isolated_comparison.json`。测试 102 项通过；离线检查无问题。

## 距离 50% 的剩余瓶颈

这轮实现了架构方案中的原分辨率几何和评分目标监督，未完整复现工作笔记中的原生模板迁移、RootSIFT、多视图和双检测器。细节头只修正已通过旧 ROI 筛选的候选，无法恢复检测漏掉、错类或被门控删除的实例。

下一轮应在训练侧验证集按阶段统计正确类别候选覆盖、NMS/门控删除、几何 IoU 跨 0.5 的数量，区分召回不足与几何不准。优先分析高 FN 的 class 2/3/8/22/18，以及 class 13/17/23 等零 TP 类的映射、样本量和混淆；再比较候选恢复、类别均衡检测训练与实例归属，不能只放宽门限。已查看的 final 此后不再作为调参后的独立最终证据。
