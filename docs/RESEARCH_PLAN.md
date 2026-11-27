# 如何把 PQ 做到 50%：文献调研与路线（2026-11-21）

## 1. 关键发现：数据来自公开基准 FloorPlanCAD

赛题的 35 类语义表（single door … railing）、SVG 的
`semantic-id` / `instance-id` 属性、Thing/Stuff 划分、PQ 计算方式，与
**FloorPlanCAD**（ICCV 2021，阿里达摩院 + SFU）完全一致。该数据集公开：
15,663 张 CAD 图纸，每张含 SVG（逐图元标注）、PNG 预览、`coco_vis`。
我们的 4000/1000/1000 很可能是它的一个子集拆分。

FloorPlanCAD 的官方任务叫 **panoptic symbol spotting（全景符号检测）**，
与赛题目标一致；差别在于：

| 维度 | FloorPlanCAD 基准 | 本赛题 |
|---|---|---|
| 输入 | **SVG 图元**（line/arc/circle/ellipse） | **只有 PNG** |
| 标注 | 逐图元 semantic + instance | 同 |
| 指标 | 图元级 PQ，IoU 按长度加权 `log(1+L)` | 栅格化到 1000px，像素 IoU>0.5 |
| 忽略类 | 未标注图元不计 | 类别 35，不计 |

## 2. 文献里的 SOTA（FloorPlanCAD test PQ）

| 方法 | 年份 | 输入表示 | PQ |
|---|---|---|---:|
| PanCADNet（数据集基线） | 2021 | CNN+GCN 图元 | 56.1 |
| CADTransformer | CVPR 2022 | 图元 token + HRNetV2 图像特征 | 68.5 |
| GAT-CADNet | CVPR 2022 | 图元图 + 8 层 GAT + 邻接矩阵 | 73.7 |
| SymPoint | ICLR 2024 | 图元点云 + Point Transformer + Mask2Former 头 | 83.3（300 轮 79.6） |
| SymPoint-V2 | 2024 | 同上 + 图层特征增强 | 83.2–90.1 |
| CADSpotting | 2025 | 点表示 + 数据增强/跨集训练 | 87.4–88.9 |
| SegCAD | 2026 | 图元点云 + 统一全景解码器 | 87.7 |
| VecFormer | 2025 | 线段采样 + PTv3 + 双分支解码器 | **91.1** |
| DPSS / ArchCAD-400K | NeurIPS 2025 | 图元+图像双通路融合 | 86.2 |

**相反地，纯像素/栅格方法明显落后**：DeepLabv3+ 在同样数据上 F1 约
68–72，而图元级方法 91–96；小符号、细线、抽象符号上差距最大
（SegCAD 论文明确做了“栅格预测投影回图元”的公平对比）。

**这意味着：只做 dense segmentation / 像素分类，上限大概就在 10–20 PQ；
要到 50%，必须走“图元级表示 + 全景检测模型”这条路。**

## 3. 我们 1.884% 与 SOTA 的差距拆解

1. **输入链路**：SOTA 假设 SVG 图元已知；我们只有 PNG，需要先做
   raster→vector。我们用骨架折线近似，图元保真度和数量都还有差距。
2. **模型**：我们用的是“分量分组 + 单图元 ViT 分类”，没有图元间注意力、
   没有实例查询/邻接预测，也没有图像特征与图元特征的融合。
3. **训练数据**：我们只用了 4000 张；公开集有 15,663 张（且允许使用
   开源数据集，需在方案中说明）。
4. **指标差异**：赛题的 1px 栅格 IoU 比 FloorPlanCAD 的长度加权 IoU 更
   苛刻，但模型能力的排序应当一致。

## 4. 到 50% 的路线（按文献证据排序）

### 阶段 0：图元恢复（必做，决定上限）

目标：从 PNG 恢复接近 SVG 的图元集合（位置误差 <0.3px）。

- 现有：颜色反混合 + 骨架折线 + 4 点采样（复用 `cadvec/layers.py`、
  `skeleton.py`）。
- 需要补：
  - 亚像素中心线（alpha 加权质心 / 灰度矩），解决 0.5px 抖动；
  - 交叉点/端点拆分与共线合并，让图元粒度接近 SVG path；
  - 圆弧/圆/椭圆的参数化恢复（现有 `strokes.py` 的拟合可复用）；
  - 图元级颜色/线宽/图层特征。
- 参考：Egiazarian et al., *Deep Vectorization of Technical Drawings*
  (ECCV 2020)；Alwan 博士论文（2021）的“直线/圆弧/交点分层 + 无监督
  拟合”流程。
- 验收：用 GT SVG 图元对恢复图元做匹配，统计召回/位置误差/实例命中
  上界（我们现有 oracle 分组上界 59.7%，阶段 0 的目标是把它推到 80%+）。

### 阶段 1：图元级全景检测模型（核心）

两条可选实现，按我们 6GB 显存的可训练性排序：

**A. GAT-CADNet 复现/精简（推荐先做）**
- 图元为节点，近邻/共线为边；8 层 GAT + 相对空间编码 + 级联边编码；
- 节点分类头（36 类，含忽略类 35）+ 邻接矩阵头（同实例连接）；
- 推理：阈值化邻接 → 同类连通分量 → 实例；
- 论文设置：100 轮、顶点上限 ~4096（显存所限），单卡可训；
- 预期：图元恢复良好的前提下，FloorPlanCAD 口径可达 60–70；赛题栅格
  口径按经验打 0.5–0.7 折，约 **30–50**。
- 代码参考：GAT-CADNet（CVPR 2022, arXiv:2201.00625）。

**B. SymPoint / VecFormer 风格查询解码器（上限更高，工程量更大）**
- 图元点云/线段 token → Point Transformer → Mask2Former 式查询头，
  预测实例掩码 + 语义；VecFormer 的 Line Sampling + 双分支 + BFR 融合
  是当前 SOTA；
- 论文设置：SymPoint 8×A100、1000 轮（300 轮 79.6 PQ），我们硬件无法
  按原规模复现；需要缩小模型/数据并接受更长训练时间；
- 预期：若能在 15k 数据上训练充分，**45–60**；否则 25–40。
- 代码参考：SymPoint (ICLR 2024, arXiv:2401.10556)、VecFormer
  (arXiv:2505.23395)、CADTransformer (CVPR 2022)。

**建议**：先做 A（GAT-CADNet-lite），因为它的输入正是我们已经有的图元，
训练预算可控；达到 20–30 后再决定是否升级到 B。

### 阶段 2：数据与训练策略

- **本地 4000 张**：PNG→图元→用同名 SVG 图元匹配生成标签；
- **公开 FloorPlanCAD train 集**（约 6–10k 张）：按赛题 FAQ“允许开源数据集、
  需在方案中说明”使用；只使用其 **train split**，绝不使用 dev/test 的
  真值（避免答案查找）；
- 训练目标与赛题口径对齐：未标注图元归入 35；语义+邻接联合损失；
- 类别不平衡：按图元数加权 + 稀有类过采样；
- 数据增强：旋转/翻转/缩放、图层随机重组（CADTransformer 的 Random
  Layer）、图元随机丢弃/加噪，模拟 PNG 恢复误差。

### 阶段 3：指标对齐与集成

- 在本地 test_final 上用赛题的栅格 PQ 调阈值/后处理；
- 图元级预测与 v3 的像素管线做互补（例如 v3 的 Stuff 段 + 图元模型的
  Thing 实例）；
- 对 Stuff（30–34）单独优化：每类每图 1 段，收益高、代价低。

## 5. 算力与时间评估（RTX 3060 Laptop 6GB）

| 阶段 | 估算 |
|---|---|
| 图元恢复 + 特征 | 1–2 天（CPU 为主） |
| GAT-CADNet-lite 训练（4000 张，100 轮） | 约 1–2 天（6GB 可跑，顶点上限 2–4k） |
| 扩到 15k 公开数据再训 | 约 3–5 天 |
| SymPoint/VecFormer 级模型 | 6GB 不可按论文复现；需 ≥24GB 显存或云算力（论文 8×A100） |

## 6. 风险与诚实结论

1. **50% 是可达区间但不是保底**：文献里图元输入 + 强模型的 PQ 是 56–91；
   我们多了 PNG→图元 的损失和“忽略类”的歧义，现实目标先定 **25–40**，
   再冲 50。
2. **纯 dense segmentation 到不了 50**：文献证据明确（栅格方法落后
   20+ 个百分点），这也是我们实测 dense 只到 fg IoU 0.09 的原因。
3. **答案查找红线**：FloorPlanCAD 公开且 dev 图可能来自其 test split；
   只能用 train split 训练，绝不能用公开 test 标注反查 dev 答案。
4. **指标差异**：FloorPlanCAD 的长度加权 IoU 与赛题 1px 栅格 IoU 不同，
   论文数字不能直接当赛题分数，必须在本地重新评测。

## 6.1 已落地（2026-11-21）

已在本机实现 GAT-CADNet-lite 并接入 `--engine gat / hybrid`（只使用本地
4000 张与 3060 6GB）：

- 全量 1000 张：GAT v2（图像特征+极坐标边+增强）1.627%，v3 1.884%，
  **hybrid 2.190%**（class 8 取 v3∪GAT 并集）；
- 泛化：calib 2.700%（v3 2.320%）、holdout 1.687%（v3 1.456%）；
- 这验证了“图元级表示”的收益，但受限于训练数据与算力，离文献的 60–90
  还有很大距离。要冲 50%，需要按第 5 节扩到公开 train split 并训练更久。

## 5.1 2025–2026 新方法（继续调研）

- **PolarSym**（arXiv:2608.11793）：在注意力中显式加入极坐标方向（SF-RoPE）
  与径向距离（RDB）约束，同规模下 PQ +1.7、收敛快 3–4 倍。→ 我们已把
  方向/多尺度距离编码加入 GAT 的边偏置。
- **CADSpotting**（arXiv:2412.07377）：坐标点云 + PTv3 + 混合池化
  （max+avg，+5.5 PQ）+ RGB 先验（+1.5 PQ）+ 滑窗聚合/SWA。→ 我们的节点
  特征已含 RGB；多尺度池化可在实例特征上借鉴。
- **VecFormer**（NeurIPS 2025）：线段采样表示 + 双分支解码 + 训练无关的
  BFR 融合（+2 PQ）。→ 属于重型改造，暂未实现。
- **SegCAD / ArchCAD**（2026）：自适应点云离散 + 多池化，指出长尾类别是
  主要瓶颈。→ 与我们的 class imbalance 观察一致。
- **CADTransformer**：图元 token + HRNet 图像特征。→ 我们已用 ResNet18
  layer3 特征在笔画中点采样，作为节点图像特征（`cadvec/image_features.py`）。

## 6.2 第三轮实测（2026-11-21）

- YOLOv8n（4000 张、32,855 框）验证 mAP50 **0.764**（c2 0.978、c3 0.951、
  c8 0.861）；检测框质量足够。
- 模板拟合上界：GT 候选 **62.3%**（中位 IoU 1.0）、连通分量 9.0%、
  v3 实例 13.7% → 候选完整性是决定性瓶颈。
- 检测框→掩码仍缺一层；标注歧义（未标注门窗外观相同）是精度硬上限。

## 7. 下一步（建议立即做）

1. 把图元恢复升级为“SVG-like 图元”（亚像素 + 拆分 + 圆弧），并用 GT
   匹配统计召回与 oracle 分组上界；
2. 实现 GAT-CADNet-lite（节点分类 + 邻接矩阵），在本地 4000 张上训练；
3. 用本地栅格 PQ 评测，目标先到 **10–20**，再扩数据冲 30–50；
4. 若显存允许，再升级到 SymPoint/VecFormer 风格查询头。

## 参考

- FloorPlanCAD (ICCV 2021): https://floorplancad.github.io/
- GAT-CADNet (CVPR 2022): https://arxiv.org/abs/2201.00625
- CADTransformer (CVPR 2022): CVF Open Access
- SymPoint (ICLR 2024): https://arxiv.org/abs/2401.10556
- VecFormer (2025): https://arxiv.org/abs/2505.23395
- SegCAD / ArchCAD (2026): ScienceDirect S0926580526005273
- CADSpotting (2025): IEEE Xplore 11679560
- Deep Vectorization of Technical Drawings (ECCV 2020)
