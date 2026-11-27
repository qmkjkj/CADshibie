# 第五轮：从 PNG 直接预测完整实例集合

本轮实现独立的 `query-ink-set-v1`，先检查模型能否在少量训练页上分开完整实例。尚未证明提分，未接入 `infer.py`，也没有替换默认权重。

## 与原程序的结构区别

原 hybrid 先产生检测框、粗掩码及多个分支候选，再修补和筛选实例；新模型直接读取整页 PNG，以共享特征和 100 个联合解码的 query 预测实例，不输入旧检测框、coarse 概率或质量规则。

- 编码器复用本地 CAD YOLO 的前 10 层权重并参与训练；取 stride 8/16/32 特征融合。
- 3 层 Transformer decoder，维度 128、4 个 attention head；每个 query 输出 30 个 Thing 类别加 null，以及辅助框和 32 维 mask embedding。
- 原生 RGB、共享多尺度特征和 XY 坐标生成原分辨率 32 通道特征；query 与特征相乘得到各自独立的 sigmoid mask，允许重叠笔画。
- 一对一 Hungarian 配对后计算类别、BCE、Dice、框 L1 和 GIoU 损失。每步采样 8192 个像素，包含实例前景、输入墨迹和均匀背景。

这参考了 MaskFormer 的“带类别标签的掩码集合预测”思路，属于本项目的轻量原型，不是官方模型复现，不能借用其论文成绩作为本项目证据。[MaskFormer 论文](https://arxiv.org/abs/2107.06278)

## 文件与接口

| 文件 | 作用 |
|---|---|
| `cadvec/query_instances.py` | `QueryInstanceModel`、采样/完整 mask 解码及集合损失 |
| `cadvec/query_data.py` | 完整实例目标、非方形增强、右下 padding 与训练点采样 |
| `tools/build_query_instances.py` | 只从固定训练/验证名单构建目标，保存文件 SHA |
| `tools/train_query_instances.py` | batch 1 训练与训练能力诊断，不按开发成绩选 checkpoint |
| `tools/infer_query_instances.py` | 独立 PNG 推理，输出赛事 schema；metadata 存在输出旁目录 |

模型输出为 `class_logits: B×Q×31`、`mask_embed: B×Q×32`、`pixel_features: B×32×H×W`、`boxes: B×Q×4`。完整 mask 按 query 分块解码以控制内存，不是候选数量限制。推理统一使用类别分数 >0.5 且超过 null、mask sigmoid >0.5，不做旧门控或 NMS；不同 query 保留不同实例号。

## 数据与几何口径

固定复用原 native 数据名单，共 920 页（train 800、val 120），已观察来源；每页最大 Thing 数为 96，当前 100 queries 足以容纳这份 train/val。无标签新 dev 的实例数量未知，不能保证其每页也不超过 100。

目标保留全部原生 PIL 1px Thing 实例掩码，包括同类相交、重叠笔画和单像素目标。输入保持整张 PNG 分辨率，不缩成 512px；1000×1000 仅在右、下补白到 1024×1024。框使用非空像素的 exclusive-edge 边界，增强时同步旋转、翻转，归一化到补边后的画布。推理先裁掉 padding，再把原生像素映射为 100×100 矢量坐标。

几何口径仍有限制：893 页为 1000×1000，另有 14 页为 999×1000、13 页为 1000×999。训练目标按 PNG 原生大小栅格化，999px 页面与官方固定 1000×1000 栅格的像素相位不完全一致。当前诊断还在矢量导出之前，只统计 Thing mask；不能当作最终 CAD PQ，也不能与历史不同数据及含 Stuff 的默认成绩直接比较。

旧 CAD backbone 的历史标签暴露不完全清楚。即使后续按来源组划分，仍须保留这一限制；本轮小样本训练能力检查本身不是泛化证据。

## 已完成的 200 step 能力检查

模型共 **2,362,595** 个参数。固定前 8 个训练页，不做增强；batch 1、8192 points、seed 173，AdamW 主学习率 0.0003、backbone 学习率 0.00003，取最后一步 checkpoint。

训练 200 step 用时 **41.805 秒**，PyTorch 峰值已分配显存约 **1680.8 MiB**。同一 8 页有 46 个 Thing GT；初始及最终在固定 0.5 阈值下均无 query 输出，**TP=0、FP=0、FN=46**。这没有证明完整实例恢复能力，更没有证明 PQ 增益。

产物为 `_work/round5/capacity200/`，权重 SHA 为 `e0e60f2d51f98fdb547e8f399b5c60610b1a67d67a82656f826fc0b581bcb2c8`。

## 已固定的补充预算与停止条件

在任何开发集评分之前，另固定至多 800 step，仍只用同一 8 页、相同学习率与采样设置，使两段参数更新总数上限为 1000。第二段从 `capacity200/last.pt` 加载权重，但**重置 AdamW 状态和采样随机数**；这是另一独立 session，不能称为恢复优化器后的连续 1000 step。

第二段已完成，用时 **157.431 秒**，两段训练合计 **199.236 秒**，峰值已分配显存仍为 **1680.8 MiB**。同一 8 页在 CUDA bf16、固定 0.5 阈值下最终为 **TP=0、FP=1、FN=46**。因此停止扩大训练，不跑 155 页开发集，不降低阈值。最终权重 SHA 为 `f5a3e1b0cf9fbc404b7a2457470a37d961c11b7dd119cf781889e81f7c42dc3c`。

为区分“类别置信度不足”和“笔画未学会”，另对这 8 个训练页的全部 100 queries 做了一次 GT oracle 诊断，不改变模型或推理规则：忽略类别和分数时，一对一可恢复 **10/46** 个严格 IoU>0.5 实例；限制为 query 预测的最佳非 null 类别后为 **5/46**。全部 GT 的最佳任意类别 IoU 均值为 0.3248。因此不只是阈值挡住了正确预测，几何与语义均未达到可靠学习要求；不能通过放宽门限声称提分。该 oracle 不具有部署条件，不报告为实际 PQ。

这只说明当前轻量实现和有限预算没有通过学习能力检查，不能据此否定所有实例查询方法。下次投入前，应优先解决实例查询的初始化和收敛问题，而非增加训练时长或按开发成绩试门限。诊断工具为 `tools/diagnose_query_capacity.py`，产物在 `_work/round5/capacity1000/oracle_diagnosis.json`。

独立 PNG 推理入口已用最终 checkpoint 完成一页 CPU FP32 冒烟测试，退出码 0、schema 通过。该页 CUDA bf16 的最高类别分数约 0.5013，CPU FP32 则没有 query 通过门限，显示临界分类对计算精度敏感；两种模式均没有正确实例，不能混作一致的预测证明。完整 **152 项测试通过，23.36 秒**；默认模型、权重和历史 PQ **33.785%** 保持不变。

下一阶段门槛仍是先在小样本上分开完整实例，再按来源组开展训练和真实矢量评分。当前没有进行 155 页开发筛选或默认部署。机器记录见 [第五轮结果](ROUND5_RESULTS.json)。

## 复现命令

以下为新目录 `_repro` 的复现，GPU 命令顺序执行。跨独立 session 时重新设置环境变量；第二段只恢复权重，不恢复优化器。

```powershell
$env:YOLO_CONFIG_DIR = "$PWD/_work/ultralytics"
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_OFFLINE = 'true'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

.venv/Scripts/python.exe tools/build_query_instances.py --data-root 人工智能比赛/datasets --selection _work/native_detail_data/metadata.json --queries 100 --out _work/round5/_repro/query_data
.venv/Scripts/python.exe tools/train_query_instances.py --data _work/round5/_repro/query_data --out _work/round5/_repro/capacity200 --backbone weights/yolo_seg.pt --train-pages 8 --evaluate-pages 8 --overfit --steps 200 --queries 100 --points 8192 --lr 0.0003 --seed 173 --device cuda
.venv/Scripts/python.exe tools/train_query_instances.py --data _work/round5/_repro/query_data --out _work/round5/_repro/capacity1000 --backbone weights/yolo_seg.pt --resume _work/round5/_repro/capacity200/last.pt --train-pages 8 --evaluate-pages 8 --overfit --steps 800 --queries 100 --points 8192 --lr 0.0003 --seed 173 --device cuda
.venv/Scripts/python.exe tools/infer_query_instances.py --weights _work/round5/_repro/capacity1000/last.pt --input 人工智能比赛/datasets/train_final/9WwF0.png --output _work/round5/_repro/png_probe --device cuda --score-threshold 0.5 --mask-threshold 0.5
```
