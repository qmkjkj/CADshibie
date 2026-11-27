#!/usr/bin/env python3
"""离线推理入口：PNG 平面图 -> 每图 12 字段 JSON。

评测环境约定（见 README）：
- 输入目录包含若干 ``*.png``，不提供任何 SVG/真值；
- 输出目录写入 ``<文件名去扩展名>.json``；
- 全程离线，禁止外部模型 API。

用法::

    python infer.py --input data/dev_final --output results --weights weights

输出字段与校验见 ``cadvec/schema.py``。
"""
from __future__ import annotations

import os

# 在导入 numpy/skimage 前限制底层线程数，避免与进程池叠加造成过度订阅。
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
# ultralytics 的联网检测/遥测开关（离线评测必须关闭）。
os.environ.setdefault("YOLO_OFFLINE", "1")

import argparse
import hashlib
import json
import random
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from cadvec.config import CANVAS_LIMITS, DEFAULT_CANVAS_HEIGHT, DEFAULT_CANVAS_WIDTH, InferenceConfig
from cadvec.external import ExternalApiConfig, load_refiner
from cadvec.pipeline import run_image
from cadvec.prior import ColorPrior, PrimitiveClassifier
from cadvec.schema import empty_output, write_output
from compliance.net_guard import enable_offline_guard
from compliance.params import ModelComplianceError, audit_models


def parse_canvas(value: str) -> tuple[int, int]:
    text = value.lower().replace(" ", "")
    parts = text.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("画布格式应为 WxH，例如 100x100")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("画布宽高必须是整数") from error
    low, high = CANVAS_LIMITS
    if not (low <= width <= high and low <= height <= high):
        raise argparse.ArgumentTypeError(
            f"画布宽高必须在 {low}-{high} 之间"
        )
    return width, height


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CAD 平面图 PNG 到矢量图元 JSON 的离线推理",
    )
    parser.add_argument("--input", default="data/dev_final", help="PNG 文件或目录")
    parser.add_argument("--output", default="results", help="输出 JSON 目录")
    parser.add_argument(
        "--weights", default=str(SCRIPT_DIR / "weights"),
        help="权重与模型清单目录（默认提交包内的 weights/）",
    )
    parser.add_argument(
        "--canvas", type=parse_canvas,
        default=(DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT),
        help="输出坐标系宽高，默认 100x100",
    )
    parser.add_argument("--workers", type=int, default=1, help="并行线程数，默认 1")
    parser.add_argument(
        "--max-detection-size", type=int, default=1200,
        help="检测前把图缩放到最长边不超过该值；默认 1200，保留 1000px 原图",
    )
    parser.add_argument(
        "--min-primitive-length", type=float, default=1.5,
        help="过滤短于该值的图元（画布单位），用于去除文字/标注碎片",
    )
    parser.add_argument(
        "--classifier-min-confidence", type=float, default=0.5,
        help="分类器置信度低于该值时回退忽略类",
    )
    parser.add_argument(
        "--instance-gap", type=float, default=3.0,
        help="Thing 实例端点聚合容差（画布单位）",
    )
    parser.add_argument(
        "--engine", choices=("segdet", "det", "tpl", "hybrid", "gat", "seg", "v2", "v1"), default="hybrid",
        help="推理引擎：hybrid=v3 像素管线 ∪ GAT ∪ YOLO-seg（默认）；v2=像素管线+ViT；v1=旧颜色先验流程",
    )
    parser.add_argument(
        "--palette", default=str(SCRIPT_DIR / "weights" / "stroke_palette.json"),
        help="精确描边色调色板（v2）",
    )
    parser.add_argument(
        "--vit", default=str(SCRIPT_DIR / "weights" / "vit_strokes.pt"),
        help="ViT 对象分类器权重（v2；不存在时退化为全部忽略类）",
    )
    parser.add_argument(
        "--vit-device", default="auto", choices=("auto", "cuda", "cpu"),
        help="ViT 推理设备，默认自动选择 CUDA",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.35,
        help="ViT 置信度阈值，低于该值输出忽略类 35",
    )
    parser.add_argument(
        "--color-prior", default=str(SCRIPT_DIR / "weights" / "color_class_prior.json"),
        help="v2 颜色→类别先验表（与 ViT 概率融合）",
    )
    parser.add_argument(
        "--color-prior-weight", type=float, default=0.6,
        help="颜色先验融合权重，0 表示关闭",
    )
    parser.add_argument(
        "--class-policy", default=str(SCRIPT_DIR / "weights" / "class_policy.json"),
        help="逐类阈值与抑制策略（不存在时使用全局阈值）",
    )
    parser.add_argument(
        "--nms-iou", type=float, default=0.6,
        help="同类假设按掩码 IoU 去重的阈值；0 表示关闭",
    )
    parser.add_argument(
        "--objectness-threshold", type=float, default=0.0,
        help="IoU/objectness 头的门限；模型无该头时无效",
    )
    parser.add_argument(
        "--segdet-weights", default=str(SCRIPT_DIR / "weights" / "yolo_seg.pt"),
        help="YOLO-seg 实例分割权重（--engine segdet）",
    )
    parser.add_argument(
        "--segdet-templates", default=str(SCRIPT_DIR / "weights" / "symbol_templates.pkl"),
        help="segdet 可选的模板库（提供完整候选的几何补全）",
    )
    parser.add_argument(
        "--segdet-template-max-score", type=float, default=3.0,
        help="segdet 模板拟合的接受阈值",
    )
    parser.add_argument(
        "--inside-fraction", type=float, default=0.5,
        help="笔画点落在实例掩码内的最小比例（整条认领模式）",
    )
    parser.add_argument(
        "--segdet-mode", choices=("local", "whole"), default="whole",
        help="segdet 笔画归属：whole=整条笔画认领（默认，实测更好），local=掩码边界局部切分",
    )
    parser.add_argument(
        "--det-weights", default=str(SCRIPT_DIR / "weights" / "yolo_det.pt"),
        help="YOLO 检测权重（--engine det）",
    )
    parser.add_argument("--det-conf", type=float, default=0.25,
                        help="YOLO 检测置信度阈值")
    parser.add_argument("--det-iou", type=float, default=0.5,
                        help="YOLO NMS IoU 阈值")
    parser.add_argument(
        "--tpl-weights", default=str(SCRIPT_DIR / "weights" / "symbol_templates.pkl"),
        help="符号模板库（--engine tpl）",
    )
    parser.add_argument(
        "--tpl-max-distance", type=float, default=3.0,
        help="模板拟合的平均点到笔画距离阈值（像素）",
    )
    parser.add_argument(
        "--hybrid-classes", default="8",
        help="hybrid 引擎中由 GAT 接管的类别（逗号分隔）",
    )
    parser.add_argument(
        "--segdet-classes", default="26,29,15,23,16,10,12,3,4",
        help="hybrid 引擎中由 YOLO-seg 接管的类别",
    )
    parser.add_argument(
        "--suppress-classes", default="2,11,14,17,18,21,22,30,34,16,23",
        help="hybrid 引擎最终抑制的类别（PQ 感知门控）",
    )
    parser.add_argument(
        "--hybrid-gate-classes", default="8",
        help="hybrid 引擎中要求实例被 YOLO-seg 同类检测掩码支持的类别（逗号分隔，默认 8）",
    )
    parser.add_argument(
        "--anchor-assign-classes", default="3,8,18,22,28",
        help="B-lite 锚点整实例重归属的类别；空串关闭",
    )
    parser.add_argument(
        "--anchor-assign-inside", type=float, default=0.6,
        help="实例原子落入 YOLO 框的平均比例阈值",
    )
    parser.add_argument(
        "--anchor-affinity-model", default=str(SCRIPT_DIR / "weights" / "anchor_affinity.json"),
        help="学习式 atom-anchor 亲和力 JSON；不存在则回退硬阈值",
    )
    parser.add_argument(
        "--atom-assign-classes", default="3,28",
        help="B-lite v3 原子级归属类别；空串关闭",
    )
    parser.add_argument(
        "--atom-affinity-model", default=str(SCRIPT_DIR / "weights" / "atom_affinity.json"),
        help="原子级亲和力 JSON",
    )
    parser.add_argument(
        "--hybrid-gate-threshold", type=float, default=0.1,
        help="实例级门控的最小重叠比例；无同类 YOLO 检测时该实例被丢弃",
    )
    parser.add_argument(
        "--hybrid-gate-metric", choices=("cover", "iou"), default="cover",
        help="门控重叠度量：cover=交集/较小面积（默认），iou=标准 IoU",
    )
    parser.add_argument(
        "--hybrid-raster", choices=("legacy400", "canonical1000"), default="legacy400",
        help="融合栅格：canonical1000 使用评分器的 PIL/1px/圆椭圆语义；保留旧模式作对照",
    )
    parser.add_argument(
        "--hybrid-source-identity", choices=("legacy", "scoped"), default="legacy",
        help="跨分支实例身份门控；scoped 为实验修复，默认保留已验证的历史选择路径",
    )
    parser.add_argument(
        "--geometry-refine", action="store_true",
        help="hybrid 的原图局部几何精修（实验性，仅调整 Thing 实例位置，尚未验证真实 PQ）",
    )
    parser.add_argument(
        "--stage-snapshots", default="",
        help="可选：保存 hybrid 分支及后处理配置，供固定候选重放消融（不写入预测目录）",
    )
    parser.add_argument(
        "--native-detail-weights", default=None,
        help="原分辨率残差头权重；hybrid 默认使用 weights/native_detail.pt",
    )
    parser.add_argument(
        "--segdet-image-size", type=int, default=640,
        help="YOLO-seg 输入边长（32 的倍数）；hybrid/segdet 共用，默认 640",
    )
    parser.add_argument(
        "--disable-native-detail", action="store_true",
        help="关闭原分辨率残差头；完整历史基线另加 --roi-max-instances 15",
    )
    parser.add_argument(
        "--hybrid-segdet-weights", default=str(SCRIPT_DIR / "weights" / "yolo_seg.pt"),
        help="hybrid 引擎使用的 YOLO-seg 权重（不存在则跳过该分支）",
    )
    parser.add_argument(
        "--roi-ink-weights", default=str(SCRIPT_DIR / "weights" / "roi_ink_c2.pt"),
        help="ROI 实例墨迹细化权重（不存在则关闭细化分支）",
    )
    parser.add_argument(
        "--roi-ink-classes", default="2",
        help="使用 ROI 墨迹细化的类别（逗号分隔；空串关闭）",
    )
    parser.add_argument(
        "--roi-ink-device", default="auto", choices=("auto", "cuda", "cpu"),
        help="ROI 墨迹细化设备，默认自动选择 CUDA",
    )
    parser.add_argument(
        "--roi-ink-quality", default=str(SCRIPT_DIR / "weights" / "roi_ink_quality.json"),
        help="ROI 实例几何质量门限（JSON）；不存在则不过滤",
    )
    parser.add_argument(
        "--roi-quality-model", default=str(SCRIPT_DIR / "weights" / "roi_quality_512c_all.json"),
        help="学习式 ROI 实例质量模型（JSON）；不存在则关闭",
    )
    parser.add_argument(
        "--roi-max-instances", type=int, default=0,
        help="ROI 每类实例数上限；0 表示不额外截断，仍保留检测器上限和质量筛选",
    )
    parser.add_argument(
        "--stuff-weights", default=str(SCRIPT_DIR / "weights" / "stuff_unet.pt"),
        help="全图 Stuff 语义分支权重；不存在则关闭",
    )
    parser.add_argument(
        "--stuff-thresholds", default=str(SCRIPT_DIR / "weights" / "stuff_thresholds.json"),
        help="Stuff 逐类阈值 JSON（class -> threshold）",
    )
    parser.add_argument(
        "--stuff-device", default="auto", choices=("auto", "cuda", "cpu"),
    )
    parser.add_argument(
        "--roi-ink-weights-aux", default=str(SCRIPT_DIR / "weights" / "roi_ink_512.pt"),
        help="第二套 ROI 墨迹权重（512 统一类别条件模型，用于 --roi-ink-aux-classes）",
    )
    parser.add_argument(
        "--roi-ink-aux-classes", default="3,4,26,29,22",
        help="第二套 ROI 墨迹权重负责的类别",
    )
    parser.add_argument(
        "--roi-ink-extra-weights", default=str(SCRIPT_DIR / "weights" / "roi_ink_256.pt"),
        help="第三套 ROI 墨迹权重（可选，空串关闭）",
    )
    parser.add_argument(
        "--roi-ink-extra-classes", default="8,12,14,15,18,19,20,21,25,27,28",
        help="第三套 ROI 墨迹权重负责的类别",
    )
    parser.add_argument(
        "--gat-weights", default=str(SCRIPT_DIR / "weights" / "gat_primitive.pt"),
        help="图元级 GAT 权重（--engine gat）",
    )
    parser.add_argument(
        "--gat-edge-threshold", type=float, default=0.45,
        help="GAT 邻接概率阈值，高于该值视为同实例",
    )
    parser.add_argument(
        "--gat-color-weight", type=float, default=0.5,
        help="GAT 节点概率与颜色先验的融合权重，0 关闭",
    )
    parser.add_argument(
        "--seg-weights", default=str(SCRIPT_DIR / "weights" / "seg_unet.pt"),
        help="dense segmentation 权重（--engine seg）",
    )
    parser.add_argument(
        "--seg-threshold", type=float, default=0.30,
        help="像素置信度阈值，低于该值视为背景",
    )
    parser.add_argument(
        "--seg-min-pixels", type=int, default=12,
        help="实例最小像素数",
    )
    parser.add_argument(
        "--max-instances", type=int, default=12,
        help="每图最多保留的 Thing 实例数（按置信度截断）；0 表示不限制",
    )
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张，0 表示全部")
    parser.add_argument(
        "--shard-index", type=int, default=0,
        help="多进程分片：当前分片序号（0 起），配合 --shard-count 使用",
    )
    parser.add_argument(
        "--shard-count", type=int, default=1,
        help="多进程分片总数；各分片输出合起来覆盖全部输入",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--fail-fast", action="store_true", help="遇到单图错误立即退出")
    parser.add_argument(
        "--strict-exit", action="store_true",
        help="任一单图失败时以非零码退出；默认写出合法空 JSON 后继续",
    )
    parser.add_argument(
        "--allow-network", action="store_true",
        help="仅限本地准备数据时使用；正式评测不要开启",
    )
    parser.add_argument(
        "--check-only", action="store_true", help="只做模型清单审计后退出",
    )
    parser.add_argument(
        "--external-api", action="store_true",
        help="开发期可选：通过 OpenAI 兼容端点（Ollama/Qwen）复核语义；"
             "正式评测禁止开启，且必须同时指定 --allow-network",
    )
    parser.add_argument(
        "--external-url", default="http://10.11.64.137:11434/v1",
        help="OpenAI 兼容服务基础地址，通常以 /v1 结尾（仅 --external-api 时使用）",
    )
    parser.add_argument(
        "--external-model", default="qwen2.5vl:3b",
        help="模型名（仅 --external-api 时使用）",
    )
    parser.add_argument(
        "--external-timeout", type=float, default=60.0,
        help="外部接口超时秒数（仅 --external-api 时使用）",
    )
    parser.add_argument(
        "--summary", default="", help="可选：把运行摘要另存为 JSON",
    )
    return parser


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def find_pngs(input_path: Path) -> list[Path]:
    def is_metadata(path: Path) -> bool:
        return "__MACOSX" in path.parts or path.name.startswith("._")

    if input_path.is_file():
        if input_path.suffix.lower() != ".png":
            raise SystemExit(f"输入文件不是 PNG: {input_path}")
        if is_metadata(input_path):
            raise SystemExit(f"输入是 macOS 资源文件，不是 PNG 图像: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise SystemExit(f"输入路径不存在: {input_path}")
    files = sorted(
        (path for path in input_path.rglob("*") if path.is_file()
         and path.suffix.lower() == ".png" and not is_metadata(path)),
        key=lambda path: str(path.relative_to(input_path)).lower(),
    )
    if not files:
        raise SystemExit(f"输入目录没有 PNG: {input_path}")
    return files


def _prepare_environment(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    # 不在启动时导入 torch：本方案默认不用神经网络权重，提前加载 torch 还会
    # 与 scipy/skimage 的 OpenMP 运行时冲突。若后续 manifest 声明了 torch
    # 模型，由模型加载代码自行设置种子。


def _process_one(
    path: Path,
    output_dir: Path,
    config: InferenceConfig,
    prior: ColorPrior,
    classifier: PrimitiveClassifier,
    canvas: tuple[int, int],
    refiner=None,
    *,
    palette: str | None = None,
    vit=None,
    confidence: float = 0.30,
    seg: dict | None = None,
    gat: dict | None = None,
    hybrid: dict | None = None,
    tpl: dict | None = None,
    det: dict | None = None,
    segdet: dict | None = None,
    color_prior: str | None = None,
    color_prior_weight: float = 0.0,
    class_policy: str | None = None,
    nms_iou: float = 0.0,
    objectness_threshold: float = 0.0,
    max_instances: int = 0,
    ink_refiner=None,
    refiner_classes: frozenset = frozenset(),
    quality_rules=None,
    refiners=(),
    stuff_branch=None,
    anchor_classes=frozenset(),
    anchor_inside: float = 0.6,
    anchor_affinity=None,
    atom_classes=frozenset(),
    atom_affinity=None,
) -> dict[str, Any]:
    started = time.perf_counter()
    width, height = canvas
    if segdet is not None:
        from cadvec.pipeline_segdet import run_image_segdet
        result = run_image_segdet(
            path, segdet["weights"], config, segdet["palette"],
            canvas_width=width, canvas_height=height,
            det_conf=segdet["det_conf"], det_iou=segdet["det_iou"],
            image_size=segdet.get("image_size", 640),
            max_instances=segdet["max_instances"],
            inside_fraction=segdet["inside_fraction"],
            templates_path=segdet["templates"],
            template_max_score=segdet["template_max_score"],
            device=segdet.get("device"),
            local_split=segdet.get("local_split", False),
            refiner=ink_refiner,
            refiner_classes=refiner_classes,
        )
        write_output(result["data"], output_dir / f"{path.stem}.json")
        return {"file": path.name, **result["stats"]}
    if det is not None:
        from cadvec.pipeline_det import run_image_det
        result = run_image_det(
            path, det["det_weights"], det["templates"], config, det["palette"],
            canvas_width=width, canvas_height=height,
            det_conf=det["det_conf"], det_iou=det["det_iou"],
            max_distance_px=det["max_distance"], max_instances=det["max_instances"],
        )
        write_output(result["data"], output_dir / f"{path.stem}.json")
        return {"file": path.name, **result["stats"]}
    if tpl is not None:
        from cadvec.pipeline_tpl import run_image_tpl
        result = run_image_tpl(
            path, tpl["gat_weights"], tpl["templates"], tpl["device"], config,
            tpl["palette"],
            canvas_width=width, canvas_height=height,
            confidence=tpl["confidence"], edge_threshold=tpl["edge_threshold"],
            color_prior_path=tpl["color_prior"], color_prior_weight=tpl["color_weight"],
            class_policy_path=tpl["class_policy"],
            max_instances=tpl["max_instances"], max_distance_px=tpl["max_distance"],
        )
        write_output(result["data"], output_dir / f"{path.stem}.json")
        return {"file": path.name, **result["stats"]}
    if hybrid is not None:
        from cadvec.hybrid import merge_final
        from cadvec.pipeline_gat import run_image_gat
        from cadvec.pipeline_v2 import run_image_v2
        gat_options = hybrid["gat"]
        v2_result = run_image_v2(
            path, config, hybrid["palette"], hybrid["vit"],
            canvas_width=width, canvas_height=height,
            confidence_threshold=hybrid["confidence"],
            color_prior_path=hybrid["color_prior"],
            color_prior_weight=hybrid["color_weight"],
            class_policy_path=hybrid["class_policy"],
            nms_iou=hybrid["nms_iou"],
            max_instances=hybrid["max_instances"],
        )
        gat_result = run_image_gat(
            path, gat_options["weights"], gat_options["device"], config,
            gat_options["palette"],
            canvas_width=width, canvas_height=height,
            confidence=gat_options["confidence"],
            edge_threshold=gat_options["edge_threshold"],
            max_instances=gat_options["gat_max_instances"],
            color_prior_path=gat_options["color_prior"],
            color_prior_weight=gat_options["color_weight"],
            class_policy_path=gat_options["class_policy"],
        )
        segdet_data = None
        segdet_support = None
        if hybrid.get("segdet") is not None:
            from cadvec.pipeline_segdet import run_image_segdet
            segdet_options = hybrid["segdet"]
            segdet_result = run_image_segdet(
                path, segdet_options["weights"], config, segdet_options["palette"],
                canvas_width=width, canvas_height=height,
                det_conf=segdet_options["det_conf"], det_iou=segdet_options["det_iou"],
                image_size=segdet_options.get("image_size", 640),
                max_instances=segdet_options["max_instances"],
                inside_fraction=segdet_options["inside_fraction"],
                templates_path=None,
                device=segdet_options.get("device"),
                local_split=segdet_options.get("local_split", False),
                support_classes=frozenset(hybrid.get("gate_classes", frozenset())),
                refiner=ink_refiner,
                refiner_classes=refiner_classes,
                refiners=refiners,
            )
            segdet_data = segdet_result["data"]
            segdet_support = segdet_result.get("support")
        data = merge_final(
            v2_result.data, gat_result["data"], segdet_data,
            gat_classes=hybrid["classes"],
            segdet_classes=hybrid["segdet_classes"],
            gate_classes=hybrid.get("gate_classes", frozenset()),
            gate_threshold=hybrid.get("gate_threshold", 0.1),
            gate_metric=hybrid.get("gate_metric", "cover"),
            support=segdet_support,
            segdet_union_classes=refiner_classes,
            suppress=hybrid["suppress"],
            quality_rules=quality_rules,
            raster_mode=hybrid.get("raster_mode", "legacy400"),
            source_identity=hybrid.get("source_identity", "legacy"),
        )
        merged_data = data
        if anchor_classes and segdet_support:
            detections = segdet_support.get("detections") or []
            size = segdet_support.get("size") or (0, 0)
            if detections and size[0] > 0 and size[1] > 0:
                from cadvec.anchor_assign import reassign
                data = reassign(data, detections, set(anchor_classes),
                                int(size[1]), int(size[0]),
                                inside=anchor_inside,
                                affinity=anchor_affinity)
        if atom_classes and atom_affinity and segdet_support:
            detections = segdet_support.get("detections") or []
            size = segdet_support.get("size") or (0, 0)
            if detections and size[0] > 0 and size[1] > 0:
                from cadvec.atom_assign import assign_atoms
                data = assign_atoms(data, detections, set(atom_classes),
                                    int(size[1]), int(size[0]),
                                    atom_affinity)
        if stuff_branch is not None:
            from cadvec.stuff_branch import append_stuff
            stuff_items = stuff_branch.infer(path)
            if stuff_items:
                data = append_stuff(data, stuff_items, width, height)
        if hybrid.get("stage_snapshots"):
            snapshot_dir = Path(hybrid["stage_snapshots"])
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot = {
                "schema_version": 1, "file": path.name,
                "primary": v2_result.data, "gat": gat_result["data"],
                "segdet": segdet_data, "support": segdet_support,
                "merged": merged_data, "final": data,
                "stuff_items": stuff_items if stuff_branch is not None else [],
                "merge_options": {
                    "gat_classes": sorted(hybrid["classes"]),
                    "segdet_classes": sorted(hybrid["segdet_classes"]),
                    "gate_classes": sorted(hybrid.get("gate_classes", [])),
                    "gate_threshold": hybrid.get("gate_threshold", 0.1),
                    "gate_metric": hybrid.get("gate_metric", "cover"),
                    "segdet_union_classes": sorted(refiner_classes),
                    "suppress": sorted(hybrid["suppress"]),
                    "quality_rules": quality_rules,
                    "raster_mode": hybrid.get("raster_mode", "legacy400"),
                    "source_identity": hybrid.get("source_identity", "legacy"),
                },
                "assignment": {
                    "anchor_classes": sorted(anchor_classes), "anchor_inside": anchor_inside,
                    "anchor_affinity": anchor_affinity,
                    "atom_classes": sorted(atom_classes), "atom_affinity": atom_affinity,
                },
            }
            (snapshot_dir / f"{path.stem}.json").write_text(
                json.dumps(snapshot, ensure_ascii=False, allow_nan=False,
                           separators=(",", ":")), encoding="utf-8")
        geometry_decisions = []
        if hybrid.get("geometry_refine", False):
            from cadvec.geometry_refine import refine_image
            data, geometry_decisions = refine_image(data, path)
            diagnostics = output_dir.parent / f"{output_dir.name}_geometry_refine"
            diagnostics.mkdir(parents=True, exist_ok=True)
            (diagnostics / f"{path.stem}.json").write_text(
                json.dumps({"file": path.name, "has_ground_truth": False,
                            "decisions": geometry_decisions}, ensure_ascii=False,
                           allow_nan=False, indent=2) + "\n", encoding="utf-8")
        write_output(data, output_dir / f"{path.stem}.json")
        return {
            "file": path.name,
            "primitive_count": len(data["commands"]),
            "hypotheses": v2_result.hypotheses,
            "objects": gat_result["stats"].get("instances", 0),
            "seconds": round(time.perf_counter() - started, 3),
            "command_counts": {name: data["commands"].count(command)
                               for command, name in enumerate(("line", "arc", "circle", "ellipse"))},
            "geometry_refine": {"enabled": hybrid.get("geometry_refine", False),
                                "considered": len(geometry_decisions),
                                "accepted": sum(row["accepted"] for row in geometry_decisions)},
        }
    if gat is not None:
        from cadvec.pipeline_gat import run_image_gat
        result = run_image_gat(
            path, gat["weights"], gat["device"], config, gat["palette"],
            canvas_width=width, canvas_height=height,
            confidence=gat["confidence"], edge_threshold=gat["edge_threshold"],
            max_instances=gat["max_instances"],
            color_prior_path=gat["color_prior"],
            color_prior_weight=gat["color_weight"],
            class_policy_path=gat["class_policy"],
        )
        write_output(result["data"], output_dir / f"{path.stem}.json")
        return {"file": path.name, **result["stats"]}
    if seg is not None:
        from cadvec.pipeline_seg import run_image_seg
        result = run_image_seg(
            path, seg["weights"], seg["device"], config,
            canvas_width=width, canvas_height=height,
            threshold=seg["threshold"], min_pixels=seg["min_pixels"],
            max_instances=seg["max_instances"],
        )
        write_output(result["data"], output_dir / f"{path.stem}.json")
        return {"file": path.name, **result["stats"]}
    if palette is not None:
        from cadvec.pipeline_v2 import run_image_v2
        result = run_image_v2(
            path, config, palette, vit,
            canvas_width=width, canvas_height=height,
            confidence_threshold=confidence,
            color_prior_path=color_prior,
            color_prior_weight=color_prior_weight,
            class_policy_path=class_policy,
            nms_iou=nms_iou,
            objectness_threshold=objectness_threshold,
            max_instances=max_instances,
        )
    else:
        result = run_image(
            path, config, prior, classifier,
            canvas_width=width, canvas_height=height, refiner=refiner,
        )
    write_output(result.data, output_dir / f"{path.stem}.json")
    return {
        "file": path.name,
        "seconds": round(time.perf_counter() - started, 3),
        **result.stats,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.segdet_image_size < 32 or args.segdet_image_size % 32:
        print("[错误] --segdet-image-size 必须是正的 32 倍数", file=sys.stderr)
        return 2
    if args.roi_max_instances < 0:
        print("[错误] --roi-max-instances 必须大于等于 0", file=sys.stderr)
        return 2
    if args.disable_native_detail:
        args.native_detail_weights = ""
    elif args.native_detail_weights is None:
        args.native_detail_weights = (
            str(Path(args.weights) / "native_detail.pt") if args.engine == "hybrid" else "")
    if args.native_detail_weights and args.engine not in {"hybrid", "segdet"}:
        print("[错误] --native-detail-weights 只支持 hybrid 或 segdet 引擎", file=sys.stderr)
        return 2
    if args.geometry_refine and args.engine != "hybrid":
        print("[错误] --geometry-refine 当前只支持 hybrid 引擎", file=sys.stderr)
        return 2
    if args.external_api and not args.allow_network:
        print(
            "[错误] --external-api 必须与 --allow-network 同时使用；"
            "正式评测禁止调用外部模型接口",
            file=sys.stderr,
        )
        return 2
    if not args.allow_network:
        enable_offline_guard()

    _prepare_environment(args.seed)

    try:
        report = audit_models(args.weights)
        if args.native_detail_weights:
            from compliance.params import ModelEntry, _count_torch_parameters
            detail_path = Path(args.native_detail_weights).resolve()
            if not detail_path.is_file():
                raise ModelComplianceError(
                    f"原分辨率残差头权重不存在: {detail_path}；"
                    "可用 --disable-native-detail 恢复历史基线")
            listed = {(Path(args.weights)/entry.path).resolve() for entry in report.models}
            if detail_path not in listed:
                parameters = _count_torch_parameters(detail_path)
                report.total_parameters += parameters
                report.models.append(ModelEntry(
                    name="native-residual-detail",kind="cnn-native-detail",path=str(detail_path),
                    parameters=parameters,parameter_source="torch-state-dict",exists=True,
                    size_bytes=detail_path.stat().st_size))
    except ModelComplianceError as error:
        print(f"[合规检查失败] {error}", file=sys.stderr)
        return 2
    if not report.ok:
        print(f"[合规检查失败] 模型参数量 {report.total_parameters} 超过上限", file=sys.stderr)
        return 2
    if args.check_only:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return 0

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_pngs(input_path)
    if args.limit > 0:
        files = files[: args.limit]

    stems: dict[str, Path] = {}
    for path in files:
        if path.stem in stems:
            print(
                f"[错误] 输入中有重名文件 {path.name} 与 {stems[path.stem].name}，"
                "输出 JSON 会互相覆盖",
                file=sys.stderr,
            )
            return 2
        stems[path.stem] = path

    if args.shard_count > 1:
        if not 0 <= args.shard_index < args.shard_count:
            print("[错误] --shard-index 必须满足 0 <= index < shard-count", file=sys.stderr)
            return 2
        files = [
            path for index, path in enumerate(files)
            if index % args.shard_count == args.shard_index
        ]
        if not files:
            print(f"[错误] 分片 {args.shard_index}/{args.shard_count} 没有输入文件", file=sys.stderr)
            return 2

    config = InferenceConfig(
        canvas_width=args.canvas[0], canvas_height=args.canvas[1],
        seed=args.seed, weights_dir=Path(args.weights),
        max_detection_size=max(8, args.max_detection_size),
        min_primitive_length=max(0.0, args.min_primitive_length),
        classifier_min_confidence=min(1.0, max(0.0, args.classifier_min_confidence)),
        instance_gap=max(0.0, args.instance_gap),
    )
    prior = ColorPrior.load(args.weights, config)
    classifier = PrimitiveClassifier.load(args.weights)
    palette_path = None
    vit = None
    seg_options = None
    gat_options = None
    hybrid_options = None
    tpl_options = None
    det_options = None
    segdet_options = None
    if args.engine == "segdet":
        if not Path(args.segdet_weights).exists():
            print(f"[错误] 缺少 YOLO-seg 权重 {args.segdet_weights}", file=sys.stderr)
            return 2
        segdet_options = {
            "weights": args.segdet_weights, "palette": args.palette,
            "det_conf": args.det_conf, "det_iou": args.det_iou,
            "image_size": args.segdet_image_size,
            "max_instances": args.max_instances,
            "inside_fraction": args.inside_fraction,
            "templates": args.segdet_templates,
            "template_max_score": args.segdet_template_max_score,
            "device": 0 if (args.vit_device in ("auto", "cuda")
                            and _cuda_available()) else "cpu",
            "local_split": args.segdet_mode == "local",
        }
        print("实例分割引擎（YOLO-seg + 笔画矢量化）", file=sys.stderr)
    if args.engine == "det":
        if not Path(args.det_weights).exists() or not Path(args.tpl_weights).exists():
            print("[错误] det 引擎需要 YOLO 权重与模板库", file=sys.stderr)
            return 2
        det_options = {
            "det_weights": args.det_weights, "templates": args.tpl_weights,
            "palette": args.palette, "det_conf": args.det_conf, "det_iou": args.det_iou,
            "max_distance": args.tpl_max_distance, "max_instances": args.max_instances,
        }
        print("检测+模板拟合引擎（YOLO + 模板库）", file=sys.stderr)
    if args.engine == "tpl":
        if not Path(args.gat_weights).exists() or not Path(args.tpl_weights).exists():
            print("[错误] tpl 引擎需要 GAT 权重与模板库", file=sys.stderr)
            return 2
        device_name = "cuda" if (args.vit_device in ("auto", "cuda") and _cuda_available()) else "cpu"
        tpl_options = {
            "gat_weights": args.gat_weights, "templates": args.tpl_weights,
            "device": device_name, "palette": args.palette,
            "confidence": 0.3, "edge_threshold": args.gat_edge_threshold,
            "color_prior": args.color_prior, "color_weight": args.gat_color_weight,
            "class_policy": args.class_policy, "max_instances": args.max_instances,
            "max_distance": args.tpl_max_distance,
        }
        print(f"模板拟合引擎（GAT 候选 + 模板库），设备 {device_name}", file=sys.stderr)
    if args.engine == "hybrid":
        if not Path(args.gat_weights).exists() or not Path(args.vit).exists():
            print("[错误] hybrid 引擎需要 GAT 与 ViT 权重", file=sys.stderr)
            return 2
        device_name = "cuda" if (args.vit_device in ("auto", "cuda") and _cuda_available()) else "cpu"
        from cadvec.vit import VitClassifier
        vit = VitClassifier.load(args.vit, device=device_name)
        hybrid_options = {
            "palette": args.palette,
            "vit": vit,
            "confidence": args.confidence,
            "color_prior": args.color_prior,
            "color_weight": args.color_prior_weight,
            "class_policy": args.class_policy,
            "nms_iou": args.nms_iou,
            "max_instances": args.max_instances,
            "classes": {int(value) for value in args.hybrid_classes.split(",") if value.strip()},
            "segdet_classes": {int(value) for value in args.segdet_classes.split(",") if value.strip()},
            "suppress": {int(value) for value in args.suppress_classes.split(",") if value.strip()},
            "gate_classes": {int(value) for value in args.hybrid_gate_classes.split(",") if value.strip()},
            "gate_threshold": args.hybrid_gate_threshold,
            "gate_metric": args.hybrid_gate_metric,
            "raster_mode": args.hybrid_raster,
            "source_identity": args.hybrid_source_identity,
            "geometry_refine": args.geometry_refine,
            "stage_snapshots": args.stage_snapshots,
            "segdet": {
                "weights": args.hybrid_segdet_weights,
                "palette": args.palette,
                "det_conf": 0.2,
                "det_iou": 0.5,
                "image_size": args.segdet_image_size,
                "max_instances": 10,
                "inside_fraction": 0.35,
                "device": 0 if device_name == "cuda" else "cpu",
                "local_split": args.segdet_mode == "local",
            } if Path(args.hybrid_segdet_weights).exists() else None,
            "gat": {
                "weights": args.gat_weights,
                "device": device_name,
                "palette": args.palette,
                "confidence": 0.3,
                "edge_threshold": args.gat_edge_threshold,
                "gat_max_instances": 8,
                "color_prior": args.color_prior,
                "color_weight": args.gat_color_weight,
                "class_policy": args.class_policy,
            },
        }
        if hybrid_options["segdet"] is None:
            print(f"[警告] 未找到 YOLO-seg 权重 {args.hybrid_segdet_weights}，"
                  "hybrid 将跳过 YOLO-seg 分支（PQ 会下降）", file=sys.stderr)
        palette_path = args.palette
        print(f"hybrid 引擎（GAT classes {sorted(hybrid_options['classes'])}），"
              f"设备 {device_name}", file=sys.stderr)
    if args.engine == "gat":
        if not Path(args.gat_weights).exists():
            print(f"[错误] 缺少 GAT 权重 {args.gat_weights}", file=sys.stderr)
            return 2
        if not Path(args.palette).exists():
            print(f"[错误] 缺少调色板 {args.palette}", file=sys.stderr)
            return 2
        device_name = "cuda" if (args.vit_device in ("auto", "cuda") and _cuda_available()) else "cpu"
        gat_options = {
            "weights": args.gat_weights, "device": device_name,
            "palette": args.palette, "confidence": args.confidence,
            "edge_threshold": args.gat_edge_threshold,
            "max_instances": args.max_instances,
            "color_prior": args.color_prior,
            "color_weight": args.gat_color_weight,
            "class_policy": args.class_policy,
        }
        print(f"图元级 GAT 引擎，设备 {device_name}", file=sys.stderr)
    if args.engine == "seg":
        if not Path(args.seg_weights).exists():
            print(f"[错误] 缺少 segmentation 权重 {args.seg_weights}", file=sys.stderr)
            return 2
        device_name = "cuda" if (args.vit_device in ("auto", "cuda") and _cuda_available()) else "cpu"
        seg_options = {
            "weights": args.seg_weights, "device": device_name,
            "threshold": args.seg_threshold, "min_pixels": args.seg_min_pixels,
            "max_instances": args.max_instances,
        }
        print(f"dense segmentation 引擎，设备 {device_name}", file=sys.stderr)
    if args.engine == "v2":
        if Path(args.palette).exists():
            palette_path = args.palette
            from cadvec.vit import VitClassifier
            device = None
            if args.vit_device != "auto":
                device = args.vit_device
            vit = VitClassifier.load(args.vit, device=device)
            if vit is None:
                print(
                    f"[警告] 未找到 ViT 权重 {args.vit}，v2 将只做几何提取、"
                    "所有图元输出忽略类 35",
                    file=sys.stderr,
                )
            else:
                print(f"ViT 设备: {vit.device}", file=sys.stderr)
        else:
            print(
                f"[警告] 未找到调色板 {args.palette}，回退到 v1 引擎",
                file=sys.stderr,
            )
    roi_refiner = None
    roi_classes = frozenset(
        int(value) for value in args.roi_ink_classes.split(",") if value.strip())
    if roi_classes and args.engine in ("hybrid", "segdet"):
        if Path(args.roi_ink_weights).exists():
            from cadvec.roi_ink import RoiInkRefiner
            roi_refiner = RoiInkRefiner.load(
                args.roi_ink_weights, device=args.roi_ink_device,
                max_instances=args.roi_max_instances,
                quality_path=args.roi_quality_model)
            if roi_refiner is not None:
                print(f"ROI 墨迹细化：classes {sorted(roi_classes)}，"
                      f"设备 {roi_refiner.device}，阈值 {roi_refiner.threshold:.2f}",
                      file=sys.stderr)
        else:
            print(f"[警告] 未找到 ROI 墨迹权重 {args.roi_ink_weights}，"
                  "细化分支关闭", file=sys.stderr)
            roi_classes = frozenset()
    roi_plan = []
    if roi_refiner is not None and roi_classes:
        roi_plan.append((frozenset(roi_classes), roi_refiner))
    aux_classes = frozenset(
        int(value) for value in args.roi_ink_aux_classes.split(",") if value.strip())
    if aux_classes and args.engine in ("hybrid", "segdet") and Path(args.roi_ink_weights_aux).exists():
        from cadvec.roi_ink import RoiInkRefiner
        aux_refiner = RoiInkRefiner.load(
            args.roi_ink_weights_aux, device=args.roi_ink_device,
            max_instances=args.roi_max_instances,
            quality_path=args.roi_quality_model)
        if aux_refiner is not None:
            roi_plan.append((aux_classes, aux_refiner))
            roi_classes = roi_classes | aux_classes
            print(f"ROI 墨迹细化（辅助模型）：classes {sorted(aux_classes)}，"
                  f"设备 {aux_refiner.device}", file=sys.stderr)
    extra_classes = frozenset(
        int(value) for value in args.roi_ink_extra_classes.split(",") if value.strip())
    if extra_classes and args.engine in ("hybrid", "segdet") and Path(
            args.roi_ink_extra_weights).exists():
        from cadvec.roi_ink import RoiInkRefiner
        extra_refiner = RoiInkRefiner.load(
            args.roi_ink_extra_weights, device=args.roi_ink_device,
            max_instances=args.roi_max_instances,
            quality_path=args.roi_quality_model)
        if extra_refiner is not None:
            roi_plan.append((extra_classes, extra_refiner))
            roi_classes = roi_classes | extra_classes
            print(f"ROI 墨迹细化（第三模型）：classes {sorted(extra_classes)}，"
                  f"设备 {extra_refiner.device}", file=sys.stderr)
    quality_rules = {}
    if args.native_detail_weights:
        from cadvec.native_detail import NativeDetailRefiner
        native_detail = NativeDetailRefiner.load(args.native_detail_weights,
                                               device=roi_plan[0][1].device if roi_plan else "cpu")
        for _classes, model in roi_plan:
            model.native_detail = native_detail
        print(f"原分辨率残差头：{args.native_detail_weights}", file=sys.stderr)
    if roi_classes and Path(args.roi_ink_quality).exists():
        try:
            payload = json.loads(Path(args.roi_ink_quality).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        for key, value in (payload.get("rules") or {}).items():
            if isinstance(value, (list, tuple)) and len(value) == 3:
                quality_rules[int(key)] = (str(value[0]), str(value[1]), float(value[2]))
    anchor_affinity = None
    if (args.anchor_assign_classes.strip()
            and Path(args.anchor_affinity_model).exists()):
        try:
            anchor_affinity = json.loads(
                Path(args.anchor_affinity_model).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            anchor_affinity = None
    atom_affinity = None
    if (args.atom_assign_classes.strip()
            and Path(args.atom_affinity_model).exists()):
        try:
            atom_affinity = json.loads(
                Path(args.atom_affinity_model).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            atom_affinity = None
    stuff_branch = None
    if args.engine == "hybrid" and Path(args.stuff_weights).exists():
        from cadvec.stuff_branch import StuffBranch
        stuff_branch = StuffBranch.load(
            args.stuff_weights, args.stuff_thresholds, args.stuff_device)
        if stuff_branch is not None:
            print(f"Stuff 语义分支：设备 {stuff_branch.device}，"
                  f"阈值 {stuff_branch.thresholds or 'default 0.5'}",
                  file=sys.stderr)
    refiner = load_refiner(ExternalApiConfig(
        enabled=args.external_api,
        base_url=args.external_url,
        model=args.external_model,
        timeout=args.external_timeout,
    ))

    print(
        f"离线推理: {len(files)} 张 PNG，画布 {args.canvas[0]}x{args.canvas[1]}，"
        f"颜色先验 {len(prior.entries)} 条，分类器 {'启用' if classifier.available else '未启用'}，"
        f"外部接口 {'启用' if refiner is not None else '关闭'}",
        file=sys.stderr,
    )

    started = time.perf_counter()
    errors: list[dict[str, str]] = []
    processed = 0
    progress_lock = threading.Lock()
    width, height = args.canvas

    def handle(path: Path) -> dict[str, Any]:
        nonlocal processed
        stats: dict[str, Any] = {"file": path.name}
        try:
            stats = _process_one(
                path, output_dir, config, prior, classifier, args.canvas, refiner,
                palette=palette_path, vit=vit, confidence=args.confidence,
                seg=seg_options,
                gat=gat_options,
                hybrid=hybrid_options,
                tpl=tpl_options,
                det=det_options,
                segdet=segdet_options,
                color_prior=args.color_prior, color_prior_weight=args.color_prior_weight,
                class_policy=args.class_policy, nms_iou=args.nms_iou,
                objectness_threshold=args.objectness_threshold,
                max_instances=args.max_instances,
                ink_refiner=roi_refiner,
                refiner_classes=roi_classes,
                quality_rules=quality_rules,
                refiners=tuple(roi_plan),
                stuff_branch=stuff_branch,
                anchor_classes=frozenset(
                    int(value) for value in args.anchor_assign_classes.split(",")
                    if value.strip()),
                anchor_inside=args.anchor_assign_inside,
                anchor_affinity=anchor_affinity,
                atom_classes=frozenset(
                    int(value) for value in args.atom_assign_classes.split(",")
                    if value.strip()),
                atom_affinity=atom_affinity,
            )
        except Exception as error:  # noqa: BLE001 - 单图失败不应中断整批
            if args.fail_fast:
                raise
            write_output(empty_output(width, height), output_dir / f"{path.stem}.json")
            detail = {
                "file": path.name,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=3),
            }
            with progress_lock:
                errors.append(detail)
        with progress_lock:
            processed += 1
            if processed == len(files) or processed % 25 == 0:
                print(f"  进度 {processed}/{len(files)}", file=sys.stderr)
        return stats

    results: list[dict[str, Any]] = []
    if args.workers <= 1:
        for path in files:
            results.append(handle(path))
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(handle, path): path for path in files}
            for future in as_completed(futures):
                results.append(future.result())

    elapsed = time.perf_counter() - started
    commands = {"line": 0, "arc": 0, "circle": 0, "ellipse": 0}
    primitives = 0
    for item in results:
        for key, value in (item.get("command_counts") or {}).items():
            commands[key] = commands.get(key, 0) + int(value)
        primitives += int(item.get("primitive_count", 0))

    summary = {
        "input": str(input_path),
        "output": str(output_dir),
        "images": len(files),
        "errors": len(errors),
        "elapsed_seconds": round(elapsed, 3),
        "seconds_per_image": round(elapsed / max(len(files), 1), 3),
        "primitives": primitives,
        "command_counts": commands,
        "model_parameters": report.total_parameters,
        "offline_guard": not args.allow_network,
        "external_api": bool(args.external_api),
    }
    if args.engine == "hybrid":
        summary["native_detail"] = {
            "enabled": bool(args.native_detail_weights),
            "weights": Path(args.native_detail_weights).name if args.native_detail_weights else None,
            "sha256": hashlib.sha256(Path(args.native_detail_weights).read_bytes()).hexdigest()
                      if args.native_detail_weights else None,
        }
        summary["segdet_image_size"] = args.segdet_image_size
        summary["roi_max_instances"] = args.roi_max_instances
        summary["hybrid_raster"] = args.hybrid_raster
        summary["hybrid_source_identity"] = args.hybrid_source_identity
        summary["geometry_refine"] = {
            "enabled": args.geometry_refine,
            "considered": sum(item.get("geometry_refine", {}).get("considered", 0)
                              for item in results),
            "accepted": sum(item.get("geometry_refine", {}).get("accepted", 0)
                            for item in results),
        }
    if errors:
        summary["error_samples"] = errors[:5]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    return 1 if (errors and args.strict_exit) else 0


if __name__ == "__main__":
    raise SystemExit(main())
