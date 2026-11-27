# dev 全量 CAD 基线运行记录

日期：2026-11-26。使用当前 hybrid 基线完成了新 dev 的全部推理。本轮没有竞赛真值评测结果；**不能据此判断是否达到 PQ 50%**。

## 结果与产物

| 项目 | 结果 |
|---|---:|
| 输入图片 | 809 张，全部 1000×1000 |
| 预测 JSON | 809 份，同名一一对应 |
| JSON 校验通过 | 809 / 809 |
| 推理异常 | 0 |
| 有预测 / 空输出 | 779 / 30 |
| 图元总数 | 869,297 |
| Thing 实例总数 | 6,852 |
| Stuff 类别段总数 | 691 |
| 越界图元 | 0 |
| 摘要推理耗时 | 2,410.083 秒，约 40 分 10 秒 |
| 平均耗时 | 2.979 秒/张，4 个工作线程 |
| 模型参数量 | 23,077,263 |
| 代码测试 | 82 passed |

空输出是需要抽查的模型结果，不能在没有标签的情况下直接算作漏检。图元总数描述输出规模，不代表识别准确率。

- [CAD 结果目录](../results_cad/)：只包含 809 个预测 JSON。
- [CAD 基线结果包](../_work/dev_baseline_20261126/results_cad_baseline.zip)：约 14 MB，内部为 `results_cad/<id>.json`，已检查 CRC 和全部导出文件哈希。
- [推理摘要](../_work/dev_baseline_20261126/inference_summary.json)。
- [完整输出审计与逐图统计](../_work/dev_baseline_20261126/output_audit.json)。
- [参数、代码与权重指纹](../_work/dev_baseline_20261126/run_manifest.json)。
- [输入与预测一致性检查](../_work/dev_baseline_20261126/consistency_audit.json)。
- [模型与源码自检](../_work/dev_baseline_20261126/model_source_audit.json)。
- [结果包哈希与完整性记录](../_work/dev_baseline_20261126/package_manifest.json)。

这是 CAD 单项的基线产物，不是包含 OCR、表格和部署材料的完整企业赛提交包。

## 输入与复现检查

实际图片在 `dev/dev/`。外层 `dev/__MACOSX/` 含 809 个 AppleDouble 资源文件，文件名也以 `.png` 结尾，但不是图像。已修正输入枚举与重复审计工具，直接传入外层 `dev` 也只会发现 809 张真实 PNG；原始数据未修改。

`manifest.csv` 的 809 个名称与图片逐一对应。其列为 `name`、`removed_instance_id`、`removed_primitives`，不包含计算 PQ 所需的完整语义和几何真值；这份 CSV 没有进入推理。仅凭列名不能确定删除发生在图像还是标注中，后续整页/模板迁移仍须验证当前 PNG 中的实例墨迹。

[输入审计](../_work/dev_input_audit.json) 发现 803 组不同的白底合成像素内容：6 对图片完全相同，另有 154 组近似内容候选。dHash 近似只用于人工复核，不能直接判定重复或同一图纸家族。

6 对重复图片的预测全部一致。前 8 张的单线程、4 线程试跑与全量运行结果也全部一致。全量结束后重新核对 809 张输入文件 SHA256，与运行前完全相同。

本轮复用原有权重与类别策略。新加的局部 RootSIFT 和墨迹检查模块尚未接入默认推理；输入枚举过滤不改变本次内层 `dev/dev` 的图片集合。运行开始时的 `infer.py` 快照与说明保存在运行目录中。

## 固定抽样的检查图

每张图依次展示原图、1px 评分渲染语义下的预测、红色预测叠加。没有 GT；红色不能理解为正确匹配。抽样按输出图元数选择两个空页、两个最稀疏非空页、两个最密集页，同分按文件名排序。

- 空输出：[0g98q](../_work/dev_baseline_20261126/previews/empty_0g98q.png)、[20KxI](../_work/dev_baseline_20261126/previews/empty_20KxI.png)。
- 稀疏输出：[obeII](../_work/dev_baseline_20261126/previews/sparse_obeII.png)、[DCl6o](../_work/dev_baseline_20261126/previews/sparse_DCl6o.png)。
- 密集输出：[Y3K1b](../_work/dev_baseline_20261126/previews/dense_Y3K1b.png)、[LG34i](../_work/dev_baseline_20261126/previews/dense_LG34i.png)。

抽查可以看到部分轮廓缺口，以及可见结构较多而输出很少的页面；这些现象仍需 GT 判断类别范围和真正失分原因。另一个需注意的事实是：`Y3K1b` 的 7,027 个输出图元中，6,698 个属于 Stuff class 33/34，Thing 实例只有 8 个。因此还须逐类核验语义和实例归属，不能单凭覆盖面积判断识别是否充分。

## 后续提分需要的数据

这批 dev 没有 SVG/JSON 真值，与企业 Q2 的说明一致。本机项目附近、常见数据路径及下载目录中未找到原来的 `train_final`、带标签 `test_final` 或 `gt_json`；仓库 README 也明确数据不随代码提交。因此当前无法复现历史的 24.348% PQ，或测量新方案在独立留出集上的提升。

后续需要恢复原训练环境中的训练 PNG+SVG、带标签的本地评估数据、固定划分文件及实际 CAD 评分脚本。先复现同口径基线、联合审计训练/评估重复，再使用固定 calibration 选择模板/几何方案，在独立留出集验证。

当前已准备好完整四点模板库构建、数据重复审计、局部 RootSIFT 坐标配准及逐图元墨迹诊断。后两项已通过合成变换、真实裁剪/缩放、无关键点、局部集中、分离图元和多余笔画测试，但尚无真实 PQ 增益证据，验收规则也未标定。详见 [准备记录](PREPARATION.md) 与 [工作笔记核查](WORK_NOTE_AUDIT.md)。
