# 第四轮：实例笔画归属诊断与目标条件细节头

本轮完成阶段诊断、20 页原生模板试验、目标条件头配对训练及独立 class 2 后几何质量评分。各候选均未通过预定采用门槛，默认权重不变；没有追加 108 页或 1000 页回归，也没有看到开发结果后重新调参。机器记录见 [ROUND4_RESULTS.json](ROUND4_RESULTS.json)。裁片代理成绩不作为整页 PQ。

第三轮默认细节头是本轮冻结基线，SHA256 为 `dd885ae635ac546a465c1622d1f99b64cec637494e78142888459b514bade907`。实验副本位于 `_work/round4/baseline.pt`。本次配对训练没有据此更新默认细节头权重。

## 最新开发集失分诊断

使用此前已观察的 155 页、88 个来源组。完整评分仍采用同类、严格栅格 IoU > 0.5、一对一 Hungarian 匹配；Thing 为 class 0–29，Stuff 为 class 30–34。

全类结果为 PQ **35.85807%**、TP **518**、FP **333**、FN **768**。其中 Thing 有 **1221** 个 GT，**504** 个 TP、**717** 个 FN。

| Thing FN 的检测框覆盖描述 | 数量 |
|---|---:|
| 有同类框且框 IoU ≥ 0.5 | 535 |
| 没有此阈值同类框，但有异类框 | 31 |
| 没有达到此阈值的框 | 151 |

同类框独立存在覆盖 1030 个 Thing；限制候选一对一使用后，覆盖 1022 个。任意类别框的一对一覆盖为 1053 个。框覆盖诊断优先最大匹配数量、再最大框 IoU；严格几何评分保持评测器的 sum-IoU 目标，两者不能互换。框 IoU 使用 GT 原生内容投影到评分栅格后的非空半开包围框，使细横线、细竖线有非零框面积。

535 个“已有同类框”的 FN 可以互斥地描述为：

| 当前阶段表现 | 数量 |
|---|---:|
| 当前 segdet 已严格匹配，最终输出失配 | 48 |
| 已接受 ROI 有严格匹配潜力，但 segdet 一对一分配后未匹配 | 3 |
| 仅被拒绝的 ROI 有严格匹配潜力 | 79 |
| 已有原生 ROI 路由，但当前候选均不能严格匹配 | 336 |
| 类别尚未进入原生 ROI 路由 | 69 |

最后一行指原生 ROI 分支的覆盖缺口，其他几何分支仍可能处理这些类别。336 个已路由实例是本轮目标条件训练的主要依据，不能将全部 405 个“没有原生 ROI 严格潜力”都解释成网络分割失败。

全部 717 个 Thing FN 中，**49** 个在当前 segdet 阶段已匹配，到最终阶段失配；其中 **48** 个属于上表的同类框子集，另 1 个不满足该框阈值。49 与 48 的分母不同。49 中 class 2 占 33，class 14、28 各占 5，class 8 占 3。最终阶段同时恢复了 11 个 segdet 未匹配的 Thing，因此 Thing TP 为 segdet 的 542 到最终的 504。这里的“失配”可以由删除、几何修改、重分组或一对一竞争造成，尚不能统一叫作“被物理删除”。

717 个 FN 中，555 个与最终任何同类输出的栅格 IoU 为零，47 个在 (0, 0.25)，57 个在 [0.25, 0.4)，55 个在 [0.4, 0.5]，另有 3 个虽存在 IoU > 0.5 的输出，却因一对一竞争没有匹配。大量损失发生在实例恢复和筛选过程，不能仅靠把接近阈值的边界推过线解决。

### 当前 ROI 阶段追踪

`tools/trace_roi_stages.py` 只使用冻结检测框、PNG 和模型，不读取标签。对全部非空 coarse 候选计算当前 native 结果，包括生产路径拒绝的候选；原生矢量为空时，按生产逻辑回退 coarse。每页最终输出必须与第三轮最新预测完整相等，才能保存 trace。155 页均通过精确一致性检查，见 `_work/round4/roi_traces_manifest.json`。

| ROI 决策 | 候选数 | 能严格匹配至少一个 GT 的独立潜力覆盖数 |
|---|---:|---:|
| accepted | 811 | 544 |
| quality 拒绝 | 498 | 87 |
| NMS 拒绝 | 4 | 2 |
| coarse 为空 | 9 | 0 |

全部候选一对一潜力为 628，accepted 候选为 541，被拒候选为 89。不同集合的最优分配会竞争同一个 GT，**这些数字不能相加作为可获得 TP**。quality 与 NMS 独立潜力也不能简单相加推算部署收益。83 个最终 FN 仅在被拒候选中有潜力，其中 79 个属于已有同类框的子集。直接放开 498 个 quality 拒绝候选存在明显 FP 代价，本轮不这样改动。

诊断产物为 `_work/round4/diagnosis.json`（逐类、逐页、逐 GT）和 `_work/round4/diagnosis_summary.json`（含生产路由分解）。历史 `_work/round2/development/stages` 的 segdet/final 属于旧运行；它与当前 final 的差异不能归因为当前后处理。当前阶段分析只采用已验证的最新 ROI trace。

检测框来自检测器内部置信度/NMS 筛选后的结果，没有原始 pre-NMS 追踪。因此“框 IoU < 0.5”只是候选可用性描述，不能证明检测漏检是唯一原因；crop 扩展也可能包含框外目标笔画。这些数据已经用于开发分析，不是独立泛化证据。

### 进一步定位最终融合的损失

`tools/diagnose_fusion_losses.py` 用当前 trace.segdet 重新计算无几何质量规则的 merge、正常 merge、anchor、atoms 和 Stuff；没有复用旧快照中的 merged/final。155 页最终输出再次全部精确一致。

49 个 Thing 匹配损失中，47 个由 hybrid 的几何质量规则删除，1 个由 class 8 锚点归属修改造成，1 个由 class 16 固定类别抑制造成。class 2 的 33 个全因 `total >= 22` 删除，分布于 15 页、14 个来源组，实测总长度为 4.614–20.998。class 28 的 5 个来自 `mean >= 0.59`，实际均值为 0.5714–0.5881；class 14 的 5 个全在单一来源组。

这些规则存在误删，但不能因此整体关闭：在相同 merge 阶段，全部类别的规则使 TP 547→500、FP 486→335，PQ 34.874%→35.254%。anchor 也同时减少 FP、恢复其他匹配；atoms 新增 5 TP，没有损伤既有 Thing 匹配。该诊断支持未来单独验证与新几何匹配的质量评分，本轮不根据开发标签直接降低规则门槛。完整逐阶段权衡和丢失实例见 `_work/round4/fusion_losses.json`、`fusion_losses_summary.json`。

## 完整原生模板的 20 页小试验

该 CPU 试验与训练头实验分开，未叠加到训练候选，也未接入默认推理。使用原 800 页、542 个来源组建立完整原生模板，来源组与全部 155 页开发数据的 88 组无交集。目标 class 2、3、8、18、22 共 4059 个模板；保留完整图元、SVG 坐标系及来源 SHA。

先固定 seed 73 的 20 页与参数，只用真实 YOLO 框、类别、分数；每页最多 8 框、每框最多 8 模板，用局部互相 RootSIFT 匹配和相似变换迁移，再用 PNG 支持和几何条件验收。仅替换已有同类实例，不能补回缺失实例。预测全部落盘后才复制对应标签用于评分。

| 20 页结果 | 当前默认 | 模板替换 |
|---|---:|---:|
| PQ | 32.46165% | 31.99457% |
| TP | 58 | 57 |
| FP | 47 | 48 |
| FN | 98 | 99 |

69 个框中 36 个能关联已有同类预测，尝试 288 个模板，4 个通过局部特征匹配，1 个最终被采用。唯一替换位于 `5xpyl` 的 class 2：PNG 支持 F1 从 0.6043 提高到 0.7502，但对原匹配 GT 的严格 IoU 从 0.6096 降到 0.2949，丢失一个 TP。该分析在冻结候选和预测之后完成，未据此重选参数。

这说明此处的 PNG 支持度不能充分验证实例身份和笔画归属。小样本、候选上限、仅替换已有实例、相似变换与椭圆表示限制，都限制了本试验范围；**负结果只适用于这个固定配置，不能否定全部模板路线**。20 页 baseline 均精确复现当前默认，40 份输出通过 schema/哈希核查。产物见 `_work/round4/template_probe/`；此次配置不采用。

## 原裁片的目标框上下文对齐

`tools/build_instance_context.py` 复用原 `_work/native_detail_data` 的像素、coarse 概率、GT 及索引，只重跑冻结 YOLO，恢复目标检测框相对 crop 的位置和 tile 原点。检测参数为 PNG 路径输入、imgsz 640、conf 0.2、iou 0.5，完整复制原构建器的类别路由、每类最多 4 框、歧义跳过、训练侧 GT supplement 和 route 内候选编号。

920 页全部完成，10703 个裁片按原索引精确覆盖，train 9660、val 1043，原验证顺序不变；class、source、instance、confidence 全部对齐，confidence 最大差值为零。每份 NPZ 的全部 GT 目标像素及有效尺寸均重新核对。没有重算 coarse、重写 NPZ 或新增/删除样本。

| 原裁片来源 | 正例 | 负例 |
|---|---:|---:|
| train detector | 6610 | 2175 |
| train GT supplement | 834 | 41 |
| val detector | 765 | 278 |

GT supplement 生成的是完整目标框，切片后仍可能为空，因此 41 个负裁片必须保留。1 个训练补充框为零高度的细线；坐标提示仅将每轴归一化分母限制为至少 1 个原生像素，不改变检测框、标签或样本数量。

上下文文件 `_work/round4/instance_context.json` 保存 `target_box`（crop 内原生像素坐标）、`tile_origin` 以及 PNG/GT、检测权重、原索引和 metadata 的 SHA。文件 SHA256 为 `3e1af5aab1bf7c6df96504ddad0abe8454b3edb4d9d449354fac27da5e6daa4d`。训练时从这些元数据生成 dx/dy，不存重复像素。detector 负例仍使用检测器预测类别，没有换成匹配后的 GT 类别。

## 本轮模型与提前冻结的配对规则

新 `native-residual-v3` 在第三轮边界先验头上加入类别 embedding 和目标框空间提示。dx/dy 为原生像素中心相对检测框中心的偏移，分别除以框宽/高（至少 1px），截断到 [-2, 2]，padding 为零；旋转和翻转时同步变换坐标分量。

类别 embedding 为 36×128 参数，2 通道提示到第一层的投影为 2×16 参数，共新增 **4640** 个参数。它只提供当前目标的类别与框位置，尚无其他候选提示、候选间竞争或统一多实例解码，不能称为完整多候选竞争模型。

规则于 `2026-11-27T01:17:35.350175+00:00` 冻结在 `_work/round4/protocol.json`：

- control 和 instance 都从相同第三轮权重 warmstart，保留相同的边界先验；新参数零初始化。
- 两臂使用相同 800 页、9660 train 裁片，120 页/1043 val 裁片中的固定 500 裁片代理；不引入扩训页。
- 固定 4 个 epoch、batch 8、base 16、seed 73、学习率 0.0001、AdamW weight decay 0.01、cosine 调度、bf16、梯度范数 1。
- 模型构造后重置 Python/NumPy/Torch 随机数，DataLoader 显式使用相同种子的 generator，避免新增模块消耗构造随机数而改变增强/采样顺序。
- checkpoint 按固定 500 裁片代理最大值选取，epoch 0 参与选择，同分保留较早 epoch。看到结果后不延长训练预算。
- 155 页开发集须比冻结基线至少增加 0.5 个 PQ 百分点、TP 不下降，并且优于同预算 control；报告 5000 次来源组配对 bootstrap 区间，即使区间跨零。
- 裁片代理提高本身不足以部署。任何默认更新还需整页回归一致性、逐类退化记录及来源暴露说明。

epoch 0 的 control 与 instance 在固定 500 裁片上均为代理 PQ **36.7262894%**，TP 211、FP 269、FN 145。该值仅说明初始化行为一致，不能算本轮训练收益。零初始化 instance 在 16 页 ROI 重放中逐页精确复现冻结基线，`_work/round4/zero_development/replay_provenance.json` 记录 baseline/candidate 各验证 16 页。

### 配对训练与整页结果

两组均按协议完成 4 轮，随后冻结全部 checkpoint 再运行开发集比较。

| 方案 | 固定代理所选 epoch | 裁片代理 PQ | 同一 155 页 PQ | TP / FP / FN |
|---|---:|---:|---:|---|
| 第三轮默认 | 原训练 9 | 36.7263% | **35.8581%** | **518 / 333 / 768** |
| 普通续训 control | 4 | **37.0817%** | 35.5561% | 512 / 338 / 774 |
| 目标框＋类别提示 instance | 0 | 36.7263% | 35.8581% | 518 / 333 / 768 |

control 的代理上升，但整页下降 **0.301986 个百分点**，TP −6、FP +5。88 个来源组配对 bootstrap 的增益 95% 区间为 **[−0.753964, 0.088185]** 个百分点，不能据此采用；原 518 个匹配保留 507 个、丢失 11 个、新增 5 个。instance 的 4 个训练 epoch 均未超过起点，按预定规则选回新增参数仍为零值的 epoch 0；该权重实际重放的 155 份输出均与当前默认逐字段相同，零增益区间为 [0,0]。这只否定此固定预算和接线的候选，不能证明所有目标条件或候选竞争方法无效。

开发结果见 `_work/round4/development_result.json`。两次回放各自的 155 份 baseline 都精确复现第三轮当前默认；没有为负结果继续修改 epoch、学习率或提示结构。两个训练候选都不进入后续大规模成绩筛选，也不覆盖默认权重。

### 独立 class 2 后几何质量评分

该试验保持第三轮 boundary 头，不叠加 instance。先冻结 `_work/round4/post_quality_probe/protocol.json`：只用原 native 的 val120，按 seed 2735 将 94 个来源组分成 fit 78 页/62 组、calib 42 页/32 组，与 155 页开发来源组无交集。fit 有 60 TP、44 FP，calib 有 32 TP、19 FP，均达到每类标签至少 10 个的固定要求。

只使用已接受 native 实例的 `n,total,mean,max,w,h,area,aspect,density`；除 aspect 外固定 log1p 变换，再用仅 fit 拟合的 StandardScaler 和 C=0.1、无 class weight 的逻辑回归。对固定 19 个阈值 0.05–0.95，逐阈值重做整页 class 2 严格 Hungarian，最大化 `ΣIoU − 0.3378543626×0.5×预测实例数`，同分取更严阈值。校准选出 **0.10**，随后冻结模型；没有使用开发标签拟合或选阈值。

120 页候选采集时 coarse 推理按最多 8 框分批，4 页超过此上限：`2HZ92` 21 框、`MjSrQ` 14 框、`RVbaE` 和 `Xjlwa` 各 9 框。生产可能整批处理，**未验证不同 batch 形状下候选逐像素相同**。155 页开发采用已验证的生产 trace，原规则 wrapper 全字段精确复现；这一核对不能消除 fit/calib 的分批差异。

开发时仅将来源签名匹配为 segdet native 的 class 2 原 `total >= 22` 替换为学习 gate，其余 class 2 和其他类别保持旧规则。全部预测落盘后才评分，仅运行一次开发比较。

| 同一 155 页结果 | 冻结默认 | 独立质量 gate |
|---|---:|---:|
| PQ | **35.85807%** | **36.32524%** |
| TP | 518 | 537 |
| FP | 333 | 356 |
| FN | 768 | 749 |

增益 **+0.467174 个百分点**，88 来源组配对 bootstrap 95% CI 为 **[−0.054301, 1.112560]** 个百分点。独立协议扩展预定须增加至少 0.5 个百分点且 TP 不降，**未过门槛，不采用、不追加回归、不重选阈值**。原 518 个匹配全部保留，新增 19 个；同时增加 23 FP。只有 class 2 变化，27 页指标变化中 12 页 PQ 上升、13 页下降、2 页不变。

310 份预测的 schema/哈希、120 页候选及冻结源码/权重 SHA 均通过。374 条便携打分与从冻结系数直接构造的 scikit 对象最大误差为 1.11e-16，所有保留决定相同；没有再次拟合。结果、匹配变化和便携 JSON 系数见 `_work/round4/post_quality_probe/`。这些是已观察开发证据，120 页此前也参与过 native checkpoint 选择，不能称为新确认。

当前 **135 项自动化测试通过，16.19 秒**。默认入口另跑两页完整推理，由 Python 子进程核实退出码为 0、错误数为 0，输出逐字段等于第三轮保存结果；原始 PowerShell 重定向曾返回 1，但对应推理摘要错误数为 0，因此没有将该 shell 返回值冒充成功证明。正式核查见 `_work/round4/default_smoke_verified/check.json`。默认离线、参数审计通过，总参数 23,561,566；默认权重 SHA 保持 `dd885ae6…907`。

## 数据暴露与结果状态

`_work/round4/data_audit.json` 审计到 1304 个官方训练来源组，已知实验/标签评估事件覆盖后，没有未暴露的官方训练来源组或页面。即使扩训候选没有部署，其看过的来源也计为实验暴露。旧基础模型的历史训练来源仍不完全清楚，保守来源分组也可能遗漏隐藏联系。

因此，已经观察的旧 92 页、108 页确认集和历史 1000 页在本轮**只用于回归，不是新的独立确认**。换一个 split seed 无法恢复独立性；从已看过所有来源的 head 初始化，也无法将来源折重新包装为完全独立确认。本轮开发收益只能作为探索性证据。未使用无标签 dev 生成训练伪标签或评估模板。

| 后续项目 | 状态 |
|---|---|
| control 固定 4 epoch 训练及 checkpoint 选择 | 完成，epoch 4；整页下降，不采用 |
| instance 固定 4 epoch 训练及 checkpoint 选择 | 完成，epoch 0；无增益，不采用 |
| 155 页 baseline/control/instance 整页配对结果 | 完成，baseline 逐页精确核对 |
| 来源组不确定性、逐类与实例退化分析 | 完成；逐类指标保存在两套完整评测文件 |
| class 2 后几何质量评分独立试验 | 完成；+0.467174 点未达预定 +0.5 点，不采用 |

本轮没有提高默认 PQ。保留第三轮历史 1000 页回归 PQ **33.785%** 的默认版本；本轮没有重新跑该历史集，仍未达到 50%。

## 复现命令

以下 PowerShell 命令使用新输出目录，避免覆盖已冻结的实验产物。GPU 命令应顺序执行。

```powershell
$env:YOLO_CONFIG_DIR = "$PWD/_work/ultralytics"
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_OFFLINE = 'true'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

.venv/Scripts/python.exe tools/trace_roi_stages.py --snapshots _work/round2/development/stages --png-dir _work/round2/development/png --verify-final _work/round3/boundary_development/boundary --native-weights _work/round4/baseline.pt --out _work/round4/reproduction/roi_traces
.venv/Scripts/python.exe tools/diagnose_instance_stages.py --stages _work/round2/development/stages --gt-dir _work/round2/development/gt --pred-dir _work/round3/boundary_development/boundary --roi-traces _work/round4/reproduction/roi_traces --out _work/round4/reproduction/diagnosis.json
.venv/Scripts/python.exe tools/build_instance_context.py --base-data _work/native_detail_data --data-root 人工智能比赛/datasets --out _work/round4/reproduction/instance_context.json

.venv/Scripts/python.exe tools/train_native_detail.py --data _work/native_detail_data --out _work/round4/reproduction/native_control.pt --init _work/round4/baseline.pt --prior-mode boundary --conditioning none --instance-context _work/round4/reproduction/instance_context.json --epochs 4 --batch-size 8 --base 16 --lr 0.0001 --seed 73 --validation-crops 500 --train-crops-per-epoch 9660
.venv/Scripts/python.exe tools/train_native_detail.py --data _work/native_detail_data --out _work/round4/reproduction/native_instance.pt --init _work/round4/baseline.pt --prior-mode boundary --conditioning instance --instance-context _work/round4/reproduction/instance_context.json --epochs 4 --batch-size 8 --base 16 --lr 0.0001 --seed 73 --validation-crops 500 --train-crops-per-epoch 9660
```

复现模板小试验及其固定参数见 `_work/round4/template_probe/README.md`。上述实验文件位于本地 `_work`，没有提交训练数据或上传结果。
