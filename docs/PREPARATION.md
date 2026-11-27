# 等待数据期间的准备与运行方式

2026-11-26。本轮完成环境和基础工具，没有新的竞赛数据 PQ 结果，默认 hybrid 配置未改。

## 已完成

1. 创建项目内 `.venv`，复用本机 CUDA PyTorch，补装 Ultralytics 并对齐 `requirements.txt` 的直接依赖。当前为 Python 3.13.9、PyTorch 2.6.0+cu124、RTX 4060 Laptop GPU，OpenCV 4.11.0、NumPy 2.2.6、SciPy 1.15.3。
2. 修正源码自检扫描范围：不再扫描 `.venv`、缓存和实验目录；仍检查项目代码的联网导入。模型与源码自检通过，总参数 23,077,263。
3. 新增 [完整原生模板库构建](../tools/build_native_templates.py) 和 [实现模块](../cadvec/template_build.py)。保存全部四点图元、类型、宽度、颜色、长度、来源 SVG 路径、文件 SHA256、类别/实例号和原生画布信息。描述子只用于检索，去重使用完整图元与画布的精确内容；目前不会把平移/缩放后的形状合并。输出兼容现有 `TemplateLibrary`。
4. 新增 [跨数据集重复审计](../tools/audit_dataset_overlap.py)。比较解码、白底合成后的 RGB 与尺寸，不因 PNG 压缩、元数据、改名而漏掉相同图；另用 64 位 dHash 生成近似候选。近似候选不等于重复页，也不能证明图纸家族划分安全。报告保留文件与像素哈希；候选过多会显式报告截断并返回非零。
5. 新增 11 个相关测试，连同原测试共 **67 passed**。已用真实 SVG 解析、模板序列化与加载测试完整性，并测试训练划分排除、粗描述子碰撞、透明底、同名异图和损坏输入。

这次新增独立构建工具，原有 `build_templates.py` 保留用于历史实验复现；旧工具的 48 图元截断等限制仍存在，后续原生模板方案请使用新工具。

## 已运行的验证及限制

- `_work/preparation/model_audit.json`：源码检查、权重参数审计及 3 个输出 JSON 格式校验通过。
- 3 张 1000×1000 合成图的完整 hybrid：CUDA、3 套 ROI 与 Stuff 权重加载成功，离线防护启用，未使用外部 API，0 异常；合计推理约 23 秒，含初始化。
- **三张合成图最终都输出空预测**。这只验证了加载、执行与输出协议，尚未验证真实竞赛图上的非空实例输出和识别质量，不能当成基线复现或提分证据。
- 合成 SVG 生成 15 个完整原生模板，合成图重复检查通过。这些模板只用于工具验证，不用于参赛推理。
- `.venv` 使用 `--system-site-packages` 复用现有 CUDA 环境；`pip check` 仍报告宿主环境原有的 `s3fs` / `fsspec` 版本冲突，它们不在当前 CAD 推理依赖中。Linux 提交容器仍需单独验证。

详细报告与合成数据保存在 Git 忽略的 `_work/preparation/`，未修改 `validation.json` 的历史成绩。

## 收到数据后直接运行

以下命令在仓库根目录的 PowerShell 执行；按实际路径替换 `D:\data\...`。

```powershell
$env:YOLO_CONFIG_DIR = "$PWD\_work\ultralytics"
$env:YOLO_AUTOINSTALL = 'false'
$env:YOLO_OFFLINE = 'true'

# 先检查训练图和新验证图之间的完全重复与近似候选；工具不读取标签。
.venv\Scripts\python.exe tools/audit_dataset_overlap.py --dataset train=D:\data\train_final --dataset validation=D:\data\validation --output _work/overlap.json

# 保持现有默认配置复现基线，明确记录逐图异常。
.venv\Scripts\python.exe infer.py --input D:\data\validation --output _work/baseline_predictions --workers 1 --strict-exit --summary _work/baseline_summary.json

# 仅在有独立 GT JSON 时才能计算真实 PQ。
.venv\Scripts\python.exe tools/evaluate.py --gt-dir D:\data\validation_gt_json --pred-dir _work/baseline_predictions --raster-size 1000 --dilate 0 --json-out _work/baseline_metrics.json
```

构建模板前先固定训练/留出划分，并使用 `tools/build_splits.py` 格式的 JSON 限定来源。新工具只读取 `train_final.train`；所选 stem 缺失、重复或同时出现在其他划分都会报错。

```powershell
.venv\Scripts\python.exe tools/build_native_templates.py --svg-dir D:\data\train_final --splits D:\data\splits.json --out _work/native_templates.pkl
```

如果目录已经明确只包含允许用于构建的训练来源，也可显式使用 `--all-training-sources`。这只是来源声明，程序无法判断一个误标成训练目录的路径是否实际包含验证标签。模板库不能包含评估标签。

下一阶段先在固定 YOLO 框、类别、分数的条件下实现和比较原生几何检索/精修，再决定是否接入默认推理。后续新增的 `cadvec/local_match.py` 提供实验性局部 RootSIFT 候选、拒绝原因与坐标转换；已测试已知变换、实际裁剪/缩放、透明白底及空间覆盖拒绝，尚未接入推理或做真实 PQ 消融。它要求等比缩放裁片；特征匹配成功也不能代替逐图元的原图墨迹验收。完整的多页候选、整页迁移和多视图流程仍待实现。
