# CADshibie — 工程图纸 PNG → 矢量 JSON

> 赛题：从建筑/工程平面图 PNG 中识别门、窗、家具、墙体等类别，输出带几何、
> 语义类别与实例号的 12 字段矢量 JSON，按 PQ（Panoptic Quality）评分。

## 仓库说明

本仓库只包含**代码、离线权重与文档**；竞赛数据集（`train_final`/`test_final`/
`dev_final`）、SVG 真值、评测输出与实验中间产物**均未包含**（受 CC BY-NC 4.0
与赛题规则约束，且体积过大）。运行时请把数据放到 `../datasets/` 或通过
`--input/--output` 指定路径。

- 最新本地历史口径回归（test_final 同 1000 张，2026-11-27）：**PQ 33.785%**，
  TP/FP/FN = **2852/2042/4477**。本轮采用重新训练的原生边界残差细节头，
  相对上一轮 32.923% 提高 **0.862 个百分点**，TP +49、FP −18。
  回归使用经一致性核查的 ROI 缓存重放，1000 份旧模型输出全部复现上一轮。
  另在冻结后的 108 页、99 个来源组增量确认中，PQ **37.796% → 38.798%**，
  TP +15、FP −5；部分旧匹配及类别仍有退化。历史进展为
  24.359% → 32.856% → 32.923% → 33.785%（原仓库历史记录为 24.348%）。
  这些是本地结果，新 dev 无标签；旧基础权重训练范围尚未核实。尚未达到 50% 目标。
- 默认推理：`python infer.py --input <dev_final目录> --output results`
  （hybrid 引擎：v3 像素管线 ∪ GAT class8 ∪ YOLO-seg 类别 + PQ 类别门控
  + class 8 检测掩码门控 + ROI 实例墨迹细化 + 512 统一类别条件模型
  + 全图 Stuff 语义分支 + 原分辨率残差细节头）。可用
  `--roi-max-instances 15` 对照额外截断；加 `--disable-native-detail` 对照历史几何基线。
- 方法、消融和文献路线见 [方法说明](docs/METHOD.md)、[实验记录](docs/OPTIMIZATION.md)、[研究计划](docs/RESEARCH_PLAN.md)。
- 工作笔记逐项实现核查、企业新资料与后续验证顺序见 [工作笔记核查](docs/WORK_NOTE_AUDIT.md)。
- 本机环境、完整模板库与数据重复检查的运行方式见 [数据准备](docs/PREPARATION.md)。
- 新 dev 的 809 张全量基线、结果包、检查图和评测限制见 [dev 运行记录](docs/DEV_BASELINE.md)。
- 融合身份、评分栅格和可选原图几何精修的实现见 [原型记录](docs/GEOMETRY_REFINEMENT.md)，真实数据验收结论见下一条。
- 恢复带标签训练/评估数据后的清理、来源隔离、真实小规模消融与新细节头训练见 [带标签优化记录](docs/LABELED_DATA_OPTIMIZATION.md)。
- 第二轮候选召回、密集实例截断修复与负收益消融见 [候选优化记录](docs/ROUND2_CANDIDATE_OPTIMIZATION.md)。
- 第三轮边界先验训练、扩训对照、冻结确认与回退方法见 [原生几何优化记录](docs/ROUND3_NATIVE_GEOMETRY.md)，完整指标见 [第三轮结果](docs/ROUND3_RESULTS.json)。
- 第四轮目标提示、模板与质量评分试验已完成，均未达到采用标准；默认仍为 **33.785%**。见 [第四轮记录](docs/ROUND4_INSTANCE_OWNERSHIP.md) 和 [完整结果](docs/ROUND4_RESULTS.json)。
- 第五轮新增整图实例集合模型，直接从 PNG 预测实例笔画；当前小样本能力检查未通过，尚未扩大训练或替换默认。实现与限制见 [新架构试验](docs/ROUND5_QUERY_INSTANCES.md)。
- 四项优化方向的实际尝试范围、外部论文比较与下一轮建议见 [优化路线重新评估](docs/OPTIMIZATION_REASSESSMENT.md)。
- 暂停实验后的详细执行路线、数据重建、两条主线、验收和停止条件见 [后续优化实施方案](docs/NEXT_OPTIMIZATION_PLAN.md)。
- 依赖：`pip install -r requirements.txt`（GPU 可选，自动回退 CPU）。
- 离线合规：`python tools/check_compliance.py`。

---

# 工程图纸 PNG 图元解析（参赛提交包）

本目录是独立、离线、可复现的参赛交付物：输入 `dev_final` 中的 PNG 平面图，
为每张图输出同名 12 字段 JSON（Line / Arc / Circle / Ellipse 图元、语义类别、
实例号、画布尺寸、实例中心与包围盒等）。评测器只读取本目录代码与
`weights/`，不依赖仓库中旧有的 Web/3D 项目。

> 提交前请填写：**队伍名称、成员、单位、联系人**。  
> 方法说明即本文第 1、4、5、10 节，控制在 3 页内；更完整的验证记录见
> `validation.json`、`environment.json`、`weights/params_report.json`。

## 1. 任务与输出契约

| 字段 | 含义 |
|---|---|
| `commands` | `0=Line`、`1=Arc`、`2=Circle`、`3=Ellipse` |
| `args` | 每个图元 4 个二维采样点，展平为 8 个数 |
| `lengths` | 图元长度（画布坐标单位） |
| `semanticIds` | `0–34` 前景类别；`35` 忽略 |
| `instanceIds` | Thing(0–29) 非负；Stuff(30–34)/忽略(35) 为 `-1` |
| `width` / `height` | JSON 坐标系宽高，默认 `100 × 100` |
| `obj_cts` | `M×4`，`[center_x, center_y, 0, instance_id]` |
| `boxes` | `M×5`，`[xmin, ymin, xmax, ymax, semantic_id]` |
| `rgb` | 图元描边颜色，`N×3`，0–255 |
| `layerIds` | 图元所属颜色层编号 |
| `widths` | 图元线宽 |

所有字段长度一致；`args` 必须为 8 个有限数；同一实例只能有一个语义类别。
写文件前由 `cadvec/schema.py::validate_output` 严格校验，任何不合规输出会直接
报错，不会静默写出。输出文件命名为 `<PNG 文件名去扩展名>.json`。

## 2. 目录结构

```text
submission/
├── infer.py                  # 评测入口：只读 PNG，写 results/<id>.json
├── run.sh                    # 本地/容器通用启动脚本
├── requirements.txt          # 固定依赖版本（CPU-only）
├── Dockerfile                # 64-bit Linux 离线镜像
├── environment.json          # 运行环境、依赖、资源与网络声明
├── validation.json           # 验证记录与输出契约
├── docs/METHOD.md            # 方法说明（提交材料）
├── weights/
│   ├── manifest.json         # 模型清单、参数量、离线声明
│   ├── color_prior.json      # 训练集统计的颜色→语义先验（需构建）
│   ├── primitive_classifier.joblib  # 可选分类器（需训练）
│   └── params_report.json    # check_compliance 生成的参数审计
├── cadvec/
│   ├── config.py             # 赛题常量与确定性配置
│   ├── image.py              # v1 PNG 读取、颜色量化分层
│   ├── layers.py             # v2 精确描边色恢复 + 颜色层
│   ├── skeleton.py           # v2 骨架折线追踪与 4 点采样
│   ├── strokes.py            # 直线/圆/椭圆/圆弧拟合（笔画级流水线）
│   ├── parallel.py           # 平行线带、端点对齐、轴向吸附
│   ├── objects.py            # 对象假设与裁剪/栅格化
│   ├── dataset.py            # 训练样本构建（假设 ↔ GT 匹配）
│   ├── crops.py              # 4 通道对象裁剪渲染
│   ├── vit.py                # 小型 ViT 与 GPU 批量推理
│   ├── pipeline_v2.py        # v2 单图推理（默认引擎）
│   ├── vectorize.py          # v1 几何提取
│   ├── prior.py              # v1 颜色先验与可选 sklearn 分类器
│   ├── external.py           # 预留的 Ollama/Qwen 外部接口（默认关闭）
│   ├── instances.py          # v1 Thing 实例聚合
│   ├── pipeline.py           # v1 单图完整流程
│   ├── schema.py             # 12 字段装配与严格校验
│   ├── svg.py                # 训练/本地评测用 SVG 解析（推理不调用）
│   └── evaluate.py           # 与参考脚本一致的 PQ/SQ/RQ
├── compliance/
│   ├── net_guard.py          # 进程级离线防护
│   └── params.py             # 模型清单与 5B 参数量审计
├── tools/                    # 构建先验、导出 GT、训练、评估、自检
└── tests/                    # 单元/集成测试（最近记录 152 项通过）
```

## 3. 快速开始

### 3.1 本地（CPU）

```bash
cd submission
pip install -r requirements.txt

# 1) 从训练集 SVG 统计颜色→语义先验（只需做一次）
python tools/build_prior.py --svg-dir /data/train_final \
  --out weights/color_prior.json

# 2) 推理 dev_final；只读取 PNG
python infer.py --input /data/dev_final --output results \
  --weights weights --workers 4

# 3) 合规自检（模型清单、参数量、网络依赖、输出格式）
python tools/check_compliance.py --results results \
  --out weights/params_report.json
```

### 3.2 容器（离线）

```bash
docker build -t cadvec-submission .
docker run --rm --network none \
  -v /path/to/dev_final:/data/dev_final:ro \
  -v /path/to/results:/results \
  cadvec-submission
```

容器默认命令等价于：

```bash
python infer.py --input /data/dev_final --output /results \
  --weights /app/submission/weights --workers 4
```

如平台要求不同的挂载路径，直接覆盖参数即可；`--weights` 默认指向镜像内
`/app/submission/weights`。

### 3.3 常用参数

| 参数 | 说明 |
|---|---|
| `--engine hybrid` | 默认/推荐：v3 像素管线 ∪ GAT(class 8) ∪ YOLO-seg(指定类别) + PQ 类别门控 + class 8 检测掩码门控 |
| `--engine gat` | 图元级 GAT 全景检测（`weights/gat_primitive.pt`） |
| `--engine v2` | 像素管线：精确颜色恢复 + 平行线优化 + ViT（需 `weights/vit_strokes.pt`）；hybrid 的底引擎 |
| `--engine v1` | 旧引擎：颜色先验 + sklearn 树模型，保留用于对比 |
| `--palette PATH` | v2 描边色调色板，默认 `weights/stroke_palette.json` |
| `--vit PATH` | v2 ViT 权重，默认 `weights/vit_strokes.pt` |
| `--vit-device auto/cuda/cpu` | ViT 推理设备，默认自动使用 CUDA |
| `--confidence 0.35` | ViT 置信度阈值；低于该值输出忽略类 35 |
| `--color-prior PATH` | 描边色→类别先验表，默认 `weights/color_class_prior.json` |
| `--color-prior-weight 0.6` | 颜色先验与 ViT 概率的融合权重；0 关闭 |
| `--canvas 100x100` | 输出坐标宽高；与数据 SVG `viewBox` 一致时最准确 |
| `--workers N` | 线程并行；单进程确定，不影响输出内容 |
| `--shard-index/--shard-count` | 多进程分片；各分片输出合并后覆盖全部输入（GIL 下推荐用多进程而非多线程） |
| `--limit N` | 只处理前 N 张，便于调试 |
| `--fail-fast` | 任一张失败立即退出；默认写合法空 JSON 并继续 |
| `--strict-exit` | 单图失败时以非零码退出（默认仍以 0 退出，保证产物齐全） |
| `--check-only` | 只审计模型清单/参数量后退出 |
| `--allow-network` | 仅本地准备数据时使用；**评测不要开启** |
| `--external-api` | 开发期可选：调用 OpenAI 兼容端点（Ollama/Qwen）复核语义；必须与 `--allow-network` 同用，正式评测禁止 |
| `--external-url` | OpenAI 兼容基础地址，默认 `http://10.11.64.137:11434/v1` |
| `--external-model` | 模型名，默认 `qwen2.5vl:3b` |
| `--max-detection-size 1200` | 检测前最长边上限；1000px 图默认不降采样，避免细线断裂 |
| `--min-primitive-length 1.5` | 过滤短图元（画布单位），去除文字/尺寸标注碎片 |
| `--classifier-min-confidence 0.5` | 分类器置信度低于该值时回退忽略类 35 |
| `--instance-gap 3.0` | Thing 实例端点聚合容差（画布单位） |
| `--segdet-mode whole/local` | YOLO 笔画归属：`whole` 整条认领（默认）；`local` 按掩码边界切分（实验项） |
| `--hybrid-gate-classes 8` | 实例级门控类别：class 8 实例必须被 YOLO-seg 同类检测掩码覆盖，否则丢弃；空串关闭 |
| `--hybrid-gate-threshold 0.1` | 门控最小重叠比例 |
| `--hybrid-gate-metric cover/iou` | 重叠度量：`cover`=交集/较小面积（默认），`iou`=标准 IoU |
| `--roi-ink-weights PATH` | ROI 实例墨迹细化权重，默认 `weights/roi_ink_c2.pt`；缺失则关闭 |
| `--roi-ink-classes 2` | 主 ROI 权重（class 2 专用 192 模型）负责的类别 |
| `--roi-ink-extra-classes 8,12,14,15,18,19,20,21,25,27,28` | 256 ROI 权重负责的类别 |
| `--roi-ink-device auto/cuda/cpu` | 细化器设备，默认自动 CUDA |
| `--roi-ink-quality PATH` | 细化实例几何质量门限 JSON，默认 `weights/roi_ink_quality.json`；缺失则不过滤 |
| `--roi-ink-weights-aux PATH` | 512 统一类别条件 ROI 权重，默认 `weights/roi_ink_512.pt` |
| `--roi-ink-aux-classes 3,4,22,26,29` | 512 统一类别模型默认负责的类别（可扩到 `0..29`） |
| `--roi-quality-model PATH` | 学习式实例质量门控，默认 `weights/roi_quality_512c_all.json` |
| `--stuff-weights PATH` | 全图 Stuff 语义分支权重，默认 `weights/stuff_unet.pt`；缺失则关闭 |
| `--stuff-thresholds PATH` | Stuff 逐类概率阈值/最小面积，默认 `weights/stuff_thresholds.json` |
| `--stuff-device auto/cuda/cpu` | Stuff 分支设备，默认自动 CUDA |
| `--anchor-assign-classes 3,8,18,22,28` | B-lite 锚点整实例重归属类别；空串关闭 |
| `--anchor-assign-inside 0.6` | 实例原子平均落入 YOLO 框的比例阈值 |
| `--atom-assign-classes 3,28` | B-lite v3 原子级锚点归属类别；空串关闭 |
| `--atom-affinity-model PATH` | 原子级亲和力 JSON，默认 `weights/atom_affinity.json` |

## 4. 方法（3 页内说明）

v2 像素分支分五步，详细说明见 `docs/METHOD.md`；当前默认引擎为 hybrid：

1. **精确描边色恢复**（`cadvec/layers.py`）：PNG 是透明底 + 抗锯齿细线。
   对每个前景像素在“白 → 调色板色”的射线集合中取最近者，恢复精确描边色与
   覆盖率；无彩色像素按反混合亮度分桶。调色板来自训练集 SVG 的 197 种精确
   `stroke` 色（`weights/stroke_palette.json`）。旧实现把 RGBA 合成到白底，
   会得到大量混合色，颜色先验因此失真——这是 v2 最大的修复点。
2. **骨架折线几何**（`cadvec/skeleton.py`）：每个颜色层骨架化，按 2 像素
   膨胀连通后标号，每个连通分量就是一个对象假设；沿骨架在交叉点/端点处
   断开成折线，按弧长取 4 个采样点（与 GT 采样一致）。实测 4 点折线仍有
   约 18% 的真值实例达到 IoU>0.5（此前 Hough 重建约 4%）。
3. **平行线优化**（`cadvec/parallel.py`，对应“强化平行线优化”）：
   - 平行线对判定（方向差、垂距、投影重叠）并聚成 `ParallelRibbon`；
   - 按垂距排序重新切分，避免传递闭包把整张图的尺寸线串成一条带；
   - 带内端点对齐到中位数，规整窗户/家具轮廓；
   - 近水平/竖直直线吸附到精确方向。
   该模块用于笔画级流水线与跨色对象补全；默认分量对象不强制合并，因为
   消融显示合并会降低实例 IoU。
4. **ViT 语义**（`cadvec/vit.py`）：每个假设渲染 4 通道 224×224 裁剪（白底
   RGB + 假设掩码），送入 291 万参数的小型 ViT（patch16/宽192/6层/6头）
   在 GPU 上分类；ViT 概率再与训练集统计的“描边色→类别”先验融合
   （`--color-prior-weight`）。训练样本由同一几何流水线在 `train_final`
   上生成，用同名 SVG 真值按 IoU/覆盖率打标签。置信度低于 `--confidence`
   的假设输出忽略类 35。
5. **输出与校验**：同类别相接的假设合并为实例；Stuff 类别按类合并、实例号
   `-1`；`assemble_output` 生成 12 字段，`validate_output` 逐项校验后原子
   写盘。单图异常时写合法空 JSON（或 `--fail-fast` 退出）。

旧的 v1 引擎（颜色分层 + 概率 Hough + 颜色先验/树模型）保留在
`--engine v1`，便于对比。v2 的几何部分是确定性的；ViT 推理是确定性的前向
计算，`--workers` 只影响速度、不影响输出内容。

## 5. 合规对照

| 规则要求 | 本提交的做法 |
|---|---|
| 64 位 Linux、离线运行 | `Dockerfile` 基于 `python:3.13-slim`；`infer.py` 启动即安装 socket 防护；`manifest` 声明 `external_network_required=false`；无外部 API |
| 模型参数总量 ≤ 5B | 当前清单审计总量 23,561,566（原 23,077,263 + 原生细节头 484,303）；见 `weights/manifest.json`；`compliance/params.py` 启动审计 |
| 容器、启动命令、依赖版本、模型清单 | `Dockerfile`、`run.sh`、`requirements.txt`、`weights/manifest.json`、`environment.json` |
| 验证 JSON | `validation.json` 记录输出契约、测试/自检结果；`infer.py` 输出逐图 JSON 并由 `validate_output` 校验 |
| 数据与开源权重 | 训练/校准只使用主办方 `train_final`/`test_final`；使用公开开源预训练权重：torchvision ResNet18（ImageNet1K）作 GAT 特征主干，YOLOv8n-seg 的 COCO/ImageNet 初始化仅用于 A/B 实验并在 `docs/OPTIMIZATION.md` §20 声明；未使用私有数据集；`dev_final` 只读 PNG |
| 禁止测试真值/答案查找 | `infer.py` 只 glob `*.png`，不导入 `cadvec.svg`；无按文件名/ID 的硬编码映射；`tools/check_compliance.py` 会扫描并阻断 |
| 结果可复现 | 固定依赖版本、固定种子、确定性流程；单图输出与线程数无关 |

**离线防护说明**：`compliance/net_guard.py` 替换
`socket.socket.connect`、`connect_ex`、`socket.create_connection` 与
`socket.getaddrinfo`，任何联网尝试都会抛出 `OfflineViolation` 并留痕。
`--allow-network` 仅用于赛前准备数据，正式评测不会使用。

## 6. 模型清单与参数量

`weights/manifest.json` 当前列出：

- `stroke-vit`：v2 小型 ViT 对象分类器（291 万参数，4 通道 224×224，
  patch16/宽192/6层/6头），训练与推理均可用 GPU；
- `stroke-palette`：197 种精确描边色，0 神经网络参数；
- `color-prior`：v1 颜色查找表，0 参数；
- `primitive-classifier`：v1 可选 sklearn 随机森林（848,532 节点），v2 不使用。

`python infer.py --check-only` 或 `tools/check_compliance.py` 会重新计算
文件大小/参数量并核对 `total_parameters`；超过 5B 或缺清单会拒绝启动。

## 7. 数据使用与隐私

- `train_final`/`test_final` 的同名 SVG 只用于统计颜色先验、训练可选分类器
  和本地 PQ 评估；
- `dev_final` 推理只读取 PNG，不读取任何 SVG/真值；
- 本提交包不包含任何测试集真值、他人数据或按样本索引的答案；
- `tools/make_synthetic.py` 生成的合成图仅用于测试链路，已在文件名与文档中
  明确标注为合成数据。

## 8. 本地验证

```bash
# v2 训练链路（GPU）
python tools/build_palette.py --svg-dir /data/train_final --out weights/stroke_palette.json
python tools/build_candidates.py --png-dir /data/train_final --svg-dir /data/train_final \
  --palette weights/stroke_palette.json --out candidates --workers 8
python tools/train_vit.py --candidates candidates --image-dir /data/train_final \
  --out weights/vit_strokes.pt --epochs 12 --batch-size 96 --workers 3
python tools/vit_report.py --candidates candidates --weights weights/vit_strokes.pt

# v2 推理 + 本地 PQ（test_final 有 SVG 真值）
python infer.py --input /data/test_final --output pred_json --engine v2 \
  --palette weights/stroke_palette.json --vit weights/vit_strokes.pt --workers 4
python tools/svg_to_json.py --svg-dir /data/test_final --out gt_json
python -m cadvec.evaluate --gt-dir gt_json --pred-dir pred_json --raster-size 1000

# 单元/集成测试：schema、SVG 解析、实例、评估、离线防护
python -m pytest tests -q

# 合成图端到端
python tools/make_synthetic.py --out synthetic --count 8
python tools/build_prior.py --svg-dir synthetic --out synthetic/weights/color_prior.json
cp weights/manifest.json synthetic/weights/manifest.json
python infer.py --input synthetic --output synthetic/results \
  --weights synthetic/weights --workers 2

# 用合成 SVG 导出 GT 后本地评估
python tools/svg_to_json.py --svg-dir synthetic --out synthetic/gt
python -m cadvec.evaluate --gt-dir synthetic/gt \
  --pred-dir synthetic/results --raster-size 1000

# 多进程分片（推荐；6 个进程各处理 1/6，结果目录合并后即为全量）
for i in 0 1 2 3 4 5; do
  python infer.py --input /data/dev_final --output results --weights weights \
    --workers 1 --shard-index "$i" --shard-count 6 &
done
wait

# 真实本地测试集（test_final 有 SVG）
python infer.py --input /data/test_final --output pred_json --weights weights --workers 4
python tools/svg_to_json.py --svg-dir /data/test_final --out gt_json
python -m cadvec.evaluate --gt-dir gt_json --pred-dir pred_json --raster-size 1000
```

参考评分脚本用 1 像素线栅格化，坐标的亚像素偏差会显著影响 IoU，因此
`cadvec.evaluate` 额外提供 `--dilate 1/2/3` 诊断开关（默认 0，与参考脚本
一致）；正式比较请使用默认值。合成数据上的结果见 `validation.json`，它不是
官方成绩。

## 9. 权重与本地评测

`weights/` 已包含从 `train_final` 生成的离线权重：

- `stroke_palette.json`：训练集 SVG 的 197 种精确描边色（约 3KB，0 参数）；
- `vit_strokes.pt`：小型 ViT 对象分类器（2,911,525 参数，约 12MB）；
- `yolo_seg.pt` / `gat_primitive.pt` / `resnet18_layer3.pt`：hybrid 的实例分割、图元 GAT 与图像特征主干；
- `roi_ink_c2.pt` / `roi_ink_256.pt`：ROI 实例墨迹 U-Net（各 1.95M 参数）；
  默认 class 2 专用模型负责 class 2，256 模型负责 8/12/14/15/18/19/20/21/22/25/27/28；
- `roi_ink_512.pt`：512 统一类别条件 ROI 墨迹 U-Net（30 个 Thing 类，1.95M 参数），
  默认接管 class 3/4/26/29；
- `stuff_unet.pt`：全图 Stuff 语义 U-Net（30–34，1.93M 参数），512 分块推理；
  全量 1000 张相对 22.696% 提升到 **23.313%**；
- `roi_ink_unified.pt`：256 统一类别模型（保留用于消融与 512 微调起点）；
- `roi_quality_512_all.json`：默认学习式质量门控（旧 13 类 + class 3/4/26/29）；
- `roi_quality_all.json`：256 模型版本的质量门控（保留用于消融）；
- `label_qa.json`：GT 在 300/400/1000px 栅格下的稳定性诊断；
- `roi_ink.pt` / `roi_ink_all.pt` / `roi_quality.json` 保留用于消融；
- `color_prior.json`：v1 颜色查找表（0 参数，保留兼容）。

训练/重建命令：

```bash
python tools/build_palette.py --svg-dir /data/train_final --out weights/stroke_palette.json
python tools/build_candidates.py --png-dir /data/train_final --svg-dir /data/train_final \
  --palette weights/stroke_palette.json --out candidates --workers 8
python tools/train_vit.py --candidates candidates --image-dir /data/train_final \
  --out weights/vit_strokes.pt --epochs 12 --batch-size 96 --workers 3
python tools/check_compliance.py --results results --out weights/params_report.json
# ROI 实例墨迹细化（GPU）
python tools/build_roi_ink.py --png-dir /data/train_final --svg-dir /data/train_final   --out _work/roi_ink --workers 6
python tools/train_roi_ink.py --shards _work/roi_ink --out weights/roi_ink.pt --epochs 20

# R1 数据卫生：按感知哈希分组划分 / 标签栅格稳定性 / 难负例
python tools/build_splits.py --train-dir /data/train_final --test-dir /data/test_final \
  --out weights/splits.json
python tools/label_qa.py --gt-dir gt_json --out weights/label_qa.json
python tools/mine_negatives.py --png-dir /data/train_final --svg-dir /data/train_final \
  --out _work/roi_negatives --split-file weights/splits.json --split-role train

# R2 统一类别条件 ROI 模型（30 个 Thing 类，256；含空白/难负例）
python tools/build_roi_ink.py --png-dir /data/train_final --svg-dir /data/train_final \
  --out _work/roi_unified/train --size 256 --classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 \
  --split-file weights/splits.json --split-role train --negatives 1 --workers 32
python tools/train_roi_ink.py --shards _work/roi_unified/train \
  --val-shards _work/roi_unified/val --classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 \
  --cap 1500 --size 256 --epochs 24 --out weights/roi_ink_unified.pt

# R3 512 等比缩放模型（几何往返显示 256→512 命中率 74%→86%）
python tools/build_roi_ink.py --png-dir /data/train_final --svg-dir /data/train_final \
  --out _work/roi_512/train --size 512 --classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 \
  --split-file weights/splits.json --split-role train --negatives 1 --workers 32
python tools/train_roi_ink.py --shards _work/roi_512/train --val-shards _work/roi_512/val \
  --classes 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29 \
  --cap 800 --size 512 --epochs 10 --lr 0.0004 --init _work/roi_ink_unified.pt \
  --out weights/roi_ink_512.pt
```

训练标签与评估口径对齐：Thing（0-29）且 `instance-id=-1` 的图元评测时会整条
跳过，因此统一归入忽略类 35；假设与真值实例按 IoU/覆盖率匹配后打标签。

本地 `test_final` 全量 1000 张（按参考口径 raster 1000、IoU>0.5、匈牙利匹配）：

| 引擎 | PQ | SQ | RQ | TP / FP / FN |
|---|---:|---:|---:|---:|
| **hybrid + 512 + Stuff + 锚点重归属 + 原子级归属（当前默认，最终）** | **24.348%** | 68.137% | 35.734% | 2186 / 2720 / 5143 |
| hybrid + 512 + Stuff + 精度再平衡 + 锚点重归属 | 24.068% | 68.112% | 35.336% | 2158 / 2727 / 5171 |
| hybrid + 512 + Stuff + 精度再平衡 | 23.896% | 68.843% | 34.711% | 2162 / 2966 / 5167 |
| hybrid + 512 统一模型 + 全图 Stuff 分支 | 23.313% | 68.766% | 33.902% | 2158 / 3244 / 5171 |
| hybrid + 512 统一模型接管 3/4/26/29 | 22.696% | 68.733% | 33.020% | 2038 / 2977 / 5291 |
| hybrid + 统一类别 ROI 接管 class 26（上一版） | 18.768% | 68.754% | 27.298% | 1687 / 3344 / 5642 |
| hybrid + class2 专用/256 + 学习式质量门控 | 18.235% | 68.557% | 26.598% | 1648 / 3415 / 5681 |
| hybrid + 192/256 路由细化 + 几何门控 | 16.250% | 67.426% | 24.101% | 1591 / 4283 / 5738 |
| hybrid + 双模型（192）细化 + 质量门控 | 13.995% | 65.891% | 21.239% | 1392 / 4387 / 5937 |
| hybrid + 单模型 ROI 细化 | 11.804% | 66.240% | 17.820% | 1318 / 6145 / 6011 |
| hybrid + class8 检测掩码门控（上一版） | 2.893% | 62.177% | 4.653% | 238 / 2664 / 7091 |
| hybrid（v3 ∪ GAT class8 + YOLO-seg 类别 + PQ 门控，上一版） | 2.597% | 62.647% | 4.145% | 244 / 4201 / 7085 |
| hybrid（v3 ∪ GAT class8） | 2.190% | 63.856% | 3.430% | 234 / 6083 / 7095 |
| v3（+ 类抑制 / NMS / 每图 top-12） | 1.884% | 63.482% | 2.968% | 187 / 5084 / 7142 |
| YOLO-seg + 笔画（segdet） | 1.673% | 58.787% | 2.846% | 175 / 4794 / 7154 |
| GAT v2（图像特征 + 极坐标边 + 增强，top-8） | 1.627% | 63.255% | 2.572% | 166 / 5413 / 7163 |
| v2（骨架折线 + ViT + 颜色先验） | 1.575% | 63.040% | 2.499% | 210 / 9271 / 7119 |
| v1（旧流程，同 1000 张） | 0.000% | 0.000% | 0.000% | 0 / 36767 / 7329 |

泛化校验（R1 感知哈希分组划分，test_final 300/300/400）：calib
**25.819%**（基线 19.202%）、hold **23.168%**（基线 16.727%）、从未参与任何选择
的 final **24.083%**（基线 18.508%），三个子集同向。上一版按旧 500/500
随机划分的 18.403% / 18.069% 仅作参考。完整消融见 `docs/OPTIMIZATION.md`
第 7、11、14、15、16、17 节，文献与路线见 `docs/RESEARCH_PLAN.md`。

补充诊断（前 60 张子集）：v2 PQ=1.186%（TP/FP/FN=11/556/431），v1 PQ=0；
把 ViT 类别替换为真值的上界为 PQ=3.164%。该子集比全量更难，小样本波动大，
以全量 1000 张为准。最终配置相对 v1：误检下降 86%（36767→5084），并首次
取得 187 个匹配实例。

该结果只用于本地链路诊断，不是官方 `dev_final` 成绩（官方真值不公开）。
剩余误差主要来自实例边界（骨架分量与 GT 实例不完全重合）与 ViT 语义精度。

## 10. 已知限制与改进方向

- 实例聚合仅用端点邻近，复杂符号（门窗、设备组合）可能过分割/欠分割；
- 圆弧依赖骨架角度覆盖，严重遮挡或极短圆弧可能漏检；
- 文字/尺寸标注/填充线会被部分误检为对象，当前用短图元过滤 + 置信度阈值
  抑制，但仍是最主要的 FP 来源；
- 抗锯齿会把同一图层拆成多种灰度，训练用的 SVG 原色与 PNG 图层色存在域偏移；
- 参考评估的 1 像素栅格对亚像素中心线极敏感，后续可加入灰度加权中心线、
  抗锯齿亚像素拟合来提升 IoU；
- 尚未使用文字/OCR（本任务不需要文字内容）。

## 11. 外部模型接口（默认关闭）

为便于后续在另一台装有 Ollama 的机器上使用小尺寸 Qwen，本提交包预留了
外部语义复核接口 `cadvec/external.py`，但**默认不启用、不发起任何网络请求**：

- 默认推理不创建 `ExternalRefiner`，仍是纯几何 + 颜色先验/分类器；
- 网络库只在 `ExternalRefiner._request` 函数内部延迟导入，模块顶层不含
  联网导入；默认推理路径不会加载网络代码，`tools/check_compliance.py`
  对 `cadvec/external.py` 仅放行函数内延迟导入，顶层导入仍会报错；
- 只有同时指定 `--external-api --allow-network` 才会创建复核器；缺少
  `--allow-network` 时 `infer.py` 直接以返回码 2 拒绝启动；
- 外部调用失败只打印一次警告，不会中断推理或产生非法 JSON。

在其他机器上的准备步骤（仅开发期）：

```bash
ollama pull qwen2.5vl:3b
OLLAMA_HOST=0.0.0.0 ollama serve

python infer.py --input /data/dev_final --output results \
  --external-api --allow-network \
  --external-url http://10.11.64.137:11434/v1 --external-model qwen2.5vl:3b
```

接口使用 OpenAI 兼容的 `/v1/chat/completions`：把 PNG（base64 data URL）与
图元列表作为多模态消息发送，要求模型返回
`{"labels": {"<序号>": <0-34>}}`，仅采纳合法类别，然后重新做实例聚合。
`--external-url` 缺 `/v1` 时会自动补齐。

**合规边界**：开发期用 Qwen 辅助复核/生成弱标签不进入评测运行时；推荐把
结论固化为 `weights/color_prior.json` 或 `weights/primitive_classifier.joblib`，
正式评测使用这些离线权重。如果要在正式评测时调用模型，必须把 Ollama 与
模型权重一起打包进离线容器、在 `weights/manifest.json` 申报参数量
（Qwen2.5-VL-3B 约 3.75B，仍低于 5B 上限）并确认平台允许容器内回环服务；
否则属于违规访问外部模型 API。

## 12. 提交前检查清单

- [ ] 填写队伍信息，确认 `docs/METHOD.md` 为 3 页内方法说明；
- [ ] 已运行 `python tools/build_prior.py` 生成 `weights/color_prior.json`；
- [ ] `python -m pytest tests -q` 全部通过；
- [ ] `python tools/check_compliance.py --results <results>` 返回 `ok: true`；
- [ ] `python infer.py --check-only` 显示总参数 848,532（≤ 5B）；
- [ ] `results/` 中 JSON 数量与 `dev_final` PNG 数量一致、文件名一一对应；
- [ ] 压缩包 ≤ 50MB、不含旧项目、数据集、测试真值或缓存文件；
- [ ] Docker 在 `--network none` 下可完成一次推理。
