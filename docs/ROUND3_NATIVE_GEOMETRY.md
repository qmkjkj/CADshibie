# 第三轮：原分辨率几何训练

本轮沿用 CAD PQ 的正式本地口径：1000px、PIL 1px、严格 IoU>0.5、同类一对一匹配，汇总 micro PQ。开始时默认历史回归值为 **32.923%**；没有使用无标签的 809 张 dev 训练或选型。

## 受控假设

原细节头输出为 `logit(coarse) + residual`，coarse 截断到 `[0.001,0.999]`。极端概率形成约 ±6.9 的固定先验，使小幅几何增删需要较大的残差。本轮只在 coarse 的 0.5 二值边界两侧各 2 个原生像素内，将固定先验限制到 ±2；远处背景保留原先验。阈值、网络宽度、损失、数据、筛选和融合不变，从与第一轮相同的随机种子重新训练。

边界模式属于独立实验权重，不能给旧 checkpoint 直接更换公式。`native-residual-v2` 将模式、半径、截断值写入 checkpoint，加载器拒绝以 v1 声明边界模式。原 v1 默认模型的公式保持一致。

训练侧固定随机抽取 1000 个正裁片，670261 个 GT 像素中：75612 个 coarse<0.5，占 11.28%；12197 个 coarse<0.001，占 1.82%；20751 个同时处在 coarse<0.5 和所定义的边界带内，占 3.10%。这说明边界软化的直接补线范围有限，不能将强 coarse 先验称为已确认的主要瓶颈。此统计没有涵盖删除多余笔画的作用，也不能将像素比例直接换算为 PQ 增量。

## 数据、选型和确认隔离

- 训练仍是原来的 800 页、9660 个裁片，不扩入任何新确认页。
- 仍用原来的 120 页中的固定 seed=73 的 500 个验证裁片选 checkpoint：10 epochs 中代理 PQ 最大者；相同时保留较早 epoch。
- 整页开发使用第二轮已反复查看的 155 页。这是开发数据，不再作为独立证据。
- 从上一轮仅观察过检测器计数的 109 页候选中，排除 `3UElq` 的整个来源组，并核对与训练、proxy、第二轮开发/确认和完整 test1000 的来源组不相交，保留 **108 页、99 个来源组**作为本轮增量确认。
- 确认 PNG 的输入统计曾被观察，标签未用于新模型训练或选型。旧基础权重训练暴露仍未知，不能称整个系统完全未见。

预先登记的采用规则：开发集整页 PQ 至少提高 0.5 个百分点且 TP 不降，才冻结权重和全部配置并打开新确认标签；确认集 PQ 上升且 TP 不降才能采用。按来源组配对 bootstrap 5000 次，报告实际区间；区间包含零时不宣称显著泛化收益。确认之后不再据其选择类别、阈值或 checkpoint。

划分和协议：`_work/round3/protocol.json`。第二轮的 92 页确认和历史 test1000 此后仅作回归。

## 表示能力诊断

固定 500 个验证裁片包含 356 个正裁片。将 GT 墨迹直接送入现有矢量化及同一局部栅格器，356 个正例全部匹配，代理 PQ 为 **89.564%**；coarse 为 19.734%，当前默认细节头为 35.625%。这验证了这些裁片上的表示仍有余量，同时显示矢量化往返仍损失约 10.44 个代理点。

这是一项使用真值、基于局部裁片的 oracle 诊断，既不是程序成绩，也不是整页 PQ 或整个任务上限。不能以此承诺可以达到 89.564%。

## 复现入口

本轮固定基线的完整字节副本保留在 `_work/round3/previous_default.pt`，SHA 为 `076cfcb059f4baff57787b41d3b16a3e10244ae2260a40cccee918c655765ef6`。复现对照应使用它，避免默认权重更新后改变实验起点。
已有实验目录保留作为冻结证据，重跑时请使用新的 `--out` 目录。

```powershell
python tools/train_native_detail.py --data _work/native_detail_data --out _work/round3/native_boundary.pt --epochs 10 --batch-size 8 --base 16 --validation-crops 500 --prior-mode boundary
python tools/diagnose_native_roundtrip.py --data _work/native_detail_data --out _work/round3/roundtrip.json
python tools/compare_native_geometry.py --snapshots _work/round2/development/stages --png-dir _work/round2/development/png --out _work/round3/boundary_development --variant baseline=_work/round3/previous_default.pt --variant boundary=_work/round3/native_boundary.pt --verify-baseline _work/round2/development/unlimited --gt-dir _work/round2/development/gt --source-splits _work/source_grouped_splits.json
```

整页比较工具只替换 native checkpoint，逐页复用检测器和 coarse ROI 结果；其余冻结分支从快照回接。它使用生产 `refine`、合并及归属代码，先逐页验证当前基线与第二轮取消上限的输出一致，全部预测写完后才读取评分标签。两种方案共享的计算使计时不能直接用于比较端到端速度。

## 训练数据扩展的独立对照

在查看边界模型的整页开发成绩、新确认标签之前，另行登记扩训实验：不使用边界模式，从当前固定先验的默认权重出发，比较原数据继续训练与扩充官方训练来源后的继续训练。两个模型互为对照，也与边界模型分别比较，不将改动叠加。

原联合审计已经包含全部 4000 张官方训练 PNG、1000 张历史测试 PNG 和 809 张新 dev，来源闭包没有只在旧训练子集上重算。保护原来的完整 543 页训练验证、108 页新确认及 `3UElq` 的整个组，并排除已供原细节头训练的来源组后，还剩 2437 张合法的官方训练页、422 个新增来源组。按 seed=2731 每组随机取一页，避免大项目的重复裁片压倒其他来源，新增 **422 页、4667 个裁片（3635 正、1032 负）**。

扩展后共有 **1222 页、14327 个训练裁片**；原有 1043 个验证裁片的内容、顺序全部保留，因此固定随机抽出的 500 个选模裁片也完全相同。`tools/combine_native_detail.py` 只组合训练裁片索引，不复制或重新生成验证标签。

两个继续训练模型均从相同默认 checkpoint 开始：固定先验、base16、6 epochs、lr=0.0001、batch=8、seed=73；每轮都随机无放回读取 **9660** 个训练裁片，优化步数相同。原始 epoch0 也可被选为最佳，防止强制选择退化的继续训练结果。各自仍按固定裁片代理选 checkpoint。协议见 `_work/round3/expansion_protocol.json`。

**必须保留的数据暴露说明：**422 个新增官方训练来源组中，409 个与历史 test1000 的候选来源组相同。训练仅读取 `train_final` 的 SVG 转换标签，没有读取 `test_json` 或新 dev 标签；这不是历史测试标签扩训，但会使历史测试来源与新头训练重叠。因此如采用扩训模型，历史 test1000 分数只能作带新增来源暴露的回归。108 页确认仍与所有新增训练来源组不相交，且在全部候选方案选定前保持标签关闭。旧基础权重训练历史未知的限制仍然成立。

所有候选模型固定后，在原 155 页开发集选整体 PQ 最高者；仍要求相比当前默认至少提高 0.5 个百分点且 TP 不降，再冻结方案并进行一次 108 页确认。新增训练预算及这一次开发选型都不能使已使用的开发页重新成为独立测试。

```powershell
python tools/build_expansion_splits.py --source-splits _work/source_grouped_splits.json --native-metadata _work/native_detail_data/metadata.json --protected-protocol _work/round3/protocol.json --data-root 人工智能比赛/datasets --out _work/round3/expansion_splits.json
python tools/build_native_detail.py --data-root 人工智能比赛/datasets --splits _work/round3/expansion_splits.json --out _work/round3/additional_data --train-pages 0 --val-pages 0 --tile 320
python tools/combine_native_detail.py --base _work/native_detail_data --additional _work/round3/additional_data --source-splits _work/source_grouped_splits.json --out _work/round3/expanded_data
python tools/train_native_detail.py --data _work/native_detail_data --out _work/round3/native_control.pt --epochs 6 --batch-size 8 --base 16 --validation-crops 500 --init _work/round3/previous_default.pt --lr 0.0001 --train-crops-per-epoch 9660
python tools/train_native_detail.py --data _work/round3/expanded_data --out _work/round3/native_expanded.pt --epochs 6 --batch-size 8 --base 16 --validation-crops 500 --init _work/round3/previous_default.pt --lr 0.0001 --train-crops-per-epoch 9660
```

## 训练完成与开发集选择

| 方案 | 固定裁片规则所选 epoch | 裁片代理 PQ | 155 页开发 PQ | TP / FP / FN |
|---|---:|---:|---:|---|
| 当前默认 | 原训练 7 | 35.625% | 35.075% | 511 / 335 / 775 |
| 边界先验软化，重新训练 10 轮 | 9 | 36.726% | **35.858%** | **518 / 333 / 768** |
| 原数据继续训练 6 轮 | 0（起始权重） | 35.625% | 35.075% | 511 / 335 / 775 |
| 固定预算扩充数据、继续训练 6 轮 | 0（起始权重） | 35.625% | 35.075% | 511 / 335 / 775 |

两个续训实验都完成了 6 轮，但没有选出超过起始 checkpoint 的结果。已逐 tensor、类覆盖、tile 大小、先验公式核对，epoch0 的有效推理状态与基线完全相同。因此它们复用已经逐页验证的 155 份基线输出和评测文件；这不是两次新的神经网络整页推理。复用文件 SHA 见 `_work/round3/continued_development/state_equality_reuse.json`。

边界模型开发集提高 **0.782930 个百分点**，TP +7、FP −2；旧 511 个 TP 中保留 494 个、损失 17 个、恢复新的 24 个。收益与退化同时存在。88 个来源组配对 bootstrap 5000 次，增量 95% 区间为 **[−0.089127, 1.613489]** 个百分点，包含零，不能称统计显著。开发集满足预登记门槛，因此仅该模型进入确认。

冻结权重 SHA256：`dd885ae635ac546a465c1622d1f99b64cec637494e78142888459b514bade907`；参数及 buffer 元素数仍为 **484303**，未增加网络宽度或类别门限。`_work/round3/freeze.json` 记录四个候选与全部配置/代码哈希。新确认标签在冻结并校验哈希后才打开，时间戳与 108 个标签哈希见 `confirmation/label_open_audit.json`。

训练日志中的墙钟时间包括机器暂停/休眠，其中续训第 3 轮约 22000 秒不能当作有效 GPU 训练耗时。

## 冻结后确认与默认采用

108 页确认的两套预测全部完成后统一评分。逐页核对旧模型的 ROI 回接输出与完整默认基线一致，使用完全相同的检测候选、coarse 细化、门控、解码和其他冻结分支。

| 方案 | PQ | SQ | RQ | TP / FP / FN |
|---|---:|---:|---:|---|
| 上一默认 | 37.7958% | 72.3981% | 52.2055% | 503 / 294 / 627 |
| 边界修正 | **38.7982%** | **72.5406%** | **53.4848%** | **518 / 289 / 612** |

增量 **1.002376 个 PQ 百分点**，TP +15、FP −5、FN −15。99 个来源组配对 bootstrap 5000 次，增量 95% 区间为 **[0.021999, 1.909485]** 个百分点。旧 503 个 TP 保留 491 个、损失 12 个、新增 27 个。确认只评已冻结的唯一候选，没有在本确认集上选择类别、门限或 epoch。

结果满足预登记采用规则，已将冻结 checkpoint 原字节复制到 `weights/native_detail.pt`，默认加载其 v2 先验元数据。两个续训模型均未采用，新增 422 页的训练暴露没有进入默认权重；采用模型仍只使用原来的 800 个训练页面。模型及 buffer 总审计元素数仍为 **23561566**，其中细节头 **484303**。

整体改善不代表所有类别改善：确认集 class 8 的 TP 从 145 降至 141。上述 12 个旧匹配损失与 27 个新增匹配都保留在逐页记录中；没有依据确认结果另行调整类别门槛。

默认入口另做了两页完整推理，退出码 0、推理错误 0，输出逐字段等于冻结后确认的候选预测，权重 SHA 与冻结值一致。108 项测试通过，108 份确认预测的 schema 与离线检查通过。

回退到上一轮几何权重：

```powershell
python infer.py --input <PNG目录> --output <输出目录> --native-detail-weights _work/round3/previous_default.pt
```

本确认集的 PNG 输入统计曾被观察，旧基础模型的历史训练范围仍未知；这里支持的是固定旧系统下的新细节头增量，不能称整个系统从未见过这些图。108 页此后仅作已观察回归，不再作为下一轮“新确认”。所有分数都是本地 CAD 结果，新 809 张 dev 仍无真值，尚未达到 50% 目标。

## 历史 1000 页回归的缓存核查

历史回归使用 `tools/replay_native_geometry.py`：复用检测框、非 ROI 类别几何、primary/GAT/Stuff 分支，重新计算 coarse ROI 和两个 native 细节头，再调用生产细化、融合和归属代码。它不是 1000 页从 PNG 开始的全管线重新推理，运行耗时不能当作部署吞吐量。

启用这条路径前，108 页确认集的两套输出均与原来重新运行检测器的配对路径逐字段相等；另核对过 25 页历史样本。本次完整回归的 **1000 份旧模型输出**全部等于第二轮保存的基线；**117 份新模型输出**与较慢路径已经完成的历史预测相等。两类核查范围不同，不将新模型抽样核对写成全部 1000 页的全流程验证。完整来源和输入 SHA 见 `_work/round3/historical_fast/replay_provenance.json`。

| 方案 | PQ | SQ | RQ | TP / FP / FN |
|---|---:|---:|---:|---|
| 第二轮默认 | 32.9231% | 71.6015% | 45.9810% | 2803 / 2060 / 4526 |
| 第三轮默认 | **33.7854%** | **72.3982%** | **46.6661%** | **2852 / 2042 / 4477** |

同集增益为 **0.862382 个 PQ 百分点**，TP +49、FP −18、FN −49。565 个来源组配对 bootstrap 5000 次的增益 95% 区间为 **[0.380857, 1.369346]** 个百分点；这仍是已观察的历史回归，不能替代新确认。旧 2803 个匹配保留 2670 个、损失 133 个、新增 182 个。class 18 的 TP 从 112 降至 107，class 27 从 45 降至 31；逐类指标和逐页损失/恢复记录均保留，没有按这些历史结果重新选择类别或参数。

1000 份候选输出的 schema、离线和参数审计通过；总审计元素数仍为 23561566。全部 108 项自动化测试通过，冻结的生产代码哈希在收尾时仍一致。默认模型的权重、开发/确认/回归指标、训练历史与核查摘要见 [机器可读结果](ROUND3_RESULTS.json)，总入口同时更新 `validation.json`、`environment.json` 和权重清单。

```powershell
python tools/replay_native_geometry.py --snapshots _work/labeled_eval/all/stages --png-dir 人工智能比赛/datasets/test_final --out _work/round3/historical_rerun --baseline _work/round3/previous_default.pt --candidate _work/round3/native_boundary.pt --verify-baseline _work/round2/historical/pred --gt-dir 人工智能比赛/datasets/test_json --source-splits _work/source_grouped_splits.json --group-role test_final
```
