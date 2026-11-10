from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image
from platform_presets import (
    _correct_skin_tone,
    _detect_pose_keypoints,
    _sdxl_inpaint_repair,
    clothing_mask,
    detect_garment_extent,
    force_white_background,
    foreground_mask,
    limit_mask_to_subject,
    reapply_garment_detail,
)
from prewhiten import prewhiten_input

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("atelier.engine_sdxl")

_GPU_LOCK = threading.Lock()
_PIPE_LOCK = threading.Lock()
_sdxl_state: dict = {
    "pipe": None,
    "ip_adapter_loaded": False,
    "ip_adapter_reason": "",
    "depth_loaded": False,
    "lora_loaded": False,
    "lora_install_failed": False,
    "inpaint_mode": False,
}

_REPO_ROOT = Path(__file__).resolve().parent.parent
SDXL_DIR = _REPO_ROOT / "models" / "sdxl"
_IDM_DIR = _REPO_ROOT / "IDM-VTON"
_TRAINING_DIR = _REPO_ROOT / "training"

AUDIENCE_PROMPTS = {
    "women": "a fashion model wearing this garment, full body front view, professional fashion photography",
    "men": "a male fashion model wearing this garment, full body front view, professional fashion photography",
    "girls": "a child model wearing this garment, correct proportions, realistic body, full body front view",
    "boys": "a boy child model wearing this garment, correct proportions, realistic body, full body front view",
    "baby": "a baby model wearing this garment, correct proportions, realistic body, full body front view",
}

AUDIENCE_NEGATIVE = {
    "women": "worst quality, low quality, monochrome, bad anatomy, extra limbs, missing limbs, adult body on child",
    "men": "worst quality, low quality, monochrome, bad anatomy, extra limbs, missing limbs, female body",
    "girls": "worst quality, low quality, adult body, distorted proportions, big head, small body, extra limbs",
    "boys": "worst quality, low quality, adult body, distorted proportions, big head, small body, extra limbs",
    "baby": "worst quality, low quality, adult body, extra limbs, missing limbs",
}
# 通用反向(全 audience 生效):电商主图禁项——塑料感/空气刷、有色背景、地面阴影。
# CLIP 77-token 预算(实测 token 数): "distorted body"并入"deformed"、"beige background"
# 并入"colored background"; "studio shadow"并入"floor shadow"(白底后处理本就会压
# 背景阴影); watermark/text/logo/signature 裁掉 —— 88-token 时代它们位于截断
# 丢弃区, 从未被编码过, 留在串里只烧 token 预算。
COMMON_NEGATIVE = "plastic, airbrushed, deformed, colored background, floor shadow"
# 摄影质感正向(防"假人感"):电商主图需要的写实体感与面料细节。
PHOTO_QUALITY = "photorealistic, 8k uhd, dslr, soft studio lighting, detailed fabric texture, natural fabric drape"

ETHNICITY_PROMPTS = {
    "western_europe": "a model with a slender build, long legs, relatively narrow hips, European body proportions",
    "eastern_europe": "a model with a taller body, slightly broader shoulders, Eastern European body proportions",
    "west_asia": "a model with a medium height, curvy hips, Middle Eastern body proportions",
    "central_asia": "a model with a compact build, shorter legs relative to torso, Central Asian body proportions",
    "east_asia": "a model with a slim petite frame, narrower shoulders, longer torso "
    "relative to legs, East Asian body proportions",
    "southeast_asia": "a model with a smaller and slimmer body, narrower shoulders, Southeast Asian body proportions",
    "africa": "a model with long arms and legs, athletic build, African body proportions",
    "north_america": "a model with a larger and fuller body, wider waist, North American body proportions",
    "south_america": "a model with an hourglass figure, prominent waist-hip contrast, South American body proportions",
    "oceania": "a model with slightly longer legs, fuller body, Australian and Pacific body proportions",
}

GARMENT_PROMPT_SUFFIX = {
    "upper": "shirt, t-shirt, blouse",
    "tunic": "tunic top, longline blouse, mid-length shirt with hem at the hips or upper thighs",
    "dress": "dress, gown, skirt",
    "lower": "pants, trousers, shorts, jeans",
    "onesie": "onesie, romper, baby body",
}

# 袖型反向约束: SDXL 对正向袖型描述跟随弱(09290 长袖平铺被画成肘部
# 截断), negative 按检测袖型压掉对立袖型, 比正向描述收敛快。
# 词条必须前置(组装处保证): negative 按 CLIP 77 token 截断, 旧序追加在
# 末尾时袖型词全部落在丢弃区(encoded=0, C3 与 C2 逐像素相同的根因)。
# "sleeveless"离 long 视觉太远, 不占预算。
SLEEVE_NEGATIVE_PROMPTS = {
    "long": "bare forearms, short sleeves, three-quarter sleeves",
    "short": "long sleeves, sleeveless top, bare shoulders",
    "sleeveless": "long sleeves, short sleeves, sleeve fabric",
}


def _dedupe_negative(parts: list[str]) -> str:
    """按短语去重拼接 negative 片段。

    AUDIENCE_NEGATIVE 与 COMMON_NEGATIVE 的 "bad anatomy, extra limbs" 等
    短语重复时保留首次出现 —— negative 走 CLIP 77-token 截断, 重复词条
    白烧预算还挤掉袖型词。大小写不敏感, 保序, 空片段跳过。
    """
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        for phrase in part.split(","):
            phrase = phrase.strip()
            key = phrase.lower()
            if phrase and key not in seen:
                seen.add(key)
                out.append(phrase)
    return ", ".join(out)


def infer_auto_garment_type(extent: dict | None) -> str:
    """auto 品类推断: 平铺衣料几何 → 覆盖范围。

    - aspect<0.7(矮宽) → lower(裤子平铺宽矮);
    - aspect>1.6(高瘦)且衣身本体够长 → dress(一体式覆盖躯干+腿);
    - hem 过腰(mid/long)且衣身本体宽到底 → tunic(中长款, 下摆→髋/大腿。
      旧逻辑全判 upper → 下摆被 warp 框钉在腰部, "齐腰截断"根因之一);
    - 其他 → upper。

    body_ratio 门槛挡"垂带撑高的短款": 裹身衣垂带/长腰带挂在 bbox 中心
    附近时把 bbox 撑高, hem/aspect 看着像中长款, 但衣身本体只到 ~0.8
    (04469 实测 0.84) —— 按 tunic 出图会画成过髋幽灵长衫, 挡回 upper。
    extent 缺 body_ratio 键(旧调用方/手工构造)时默认 1.0, 门槛等价旧行为。
    """
    aspect = (extent or {}).get("aspect", 1.0)
    hem = (extent or {}).get("hem", "waist")
    body_ratio = (extent or {}).get("body_ratio", 1.0)
    if aspect < 0.7:
        return "lower"
    if aspect > 1.6 and body_ratio >= 0.75:
        return "dress"
    if hem in ("mid", "long") and body_ratio >= 0.9:
        return "tunic"
    return "upper"


def intersect_mask_with_garment_alpha(
    apply_mask: Image.Image | np.ndarray | None,
    ref_alpha: Image.Image | None,
    min_keep: float = 0.4,
) -> tuple[Image.Image | np.ndarray | None, float | None]:
    """贴回 mask ∩ warp 衣料形(膨胀余量): 形外生成内容不贴回原图。

    几何 mask 是超集: 长袖手臂走廊兜底到身高 85%、hem 截线放宽到腰下
    0.28 span —— SDXL 会在超集里重画裤子/肚皮/髋部并被 blend 贴回原图
    (裤子变色/漏肚脐)。warp 参考图的衣料 alpha(衣身+袖条)是"衣服该在
    哪"的形状; 膨胀余量后与 apply_mask 求交, 形外的走廊尾巴/hem 尾巴
    不再贴回, 保留原图像素(裤子/肚皮)。

    - 椭圆核 (0.055w, 0.008h): 横向错位来自袖条列(画布基模比例)与真人
      手臂列(u2net 主体外 25%)的坐标系差, 需大横余量; 纵向锚点(肩/腕/
      hem)两侧同源, 过大竖余量会把衣料形渗到裤头(09290 裤头变色)。
    - kept < min_keep(形与 mask 明显错位)/alpha 缺失/cv2 缺失 → 原样返回
      (fail-open); VTON_MASK_ALPHA_INTERSECT=0 一键关闭(A/B 用)。
    - 类型透传: PIL 进 PIL 出, ndarray 进 ndarray 出(下游按 ndarray 消费)。

    返回 (mask, kept); kept=None 表示未应用。
    """
    if apply_mask is None or ref_alpha is None:
        return apply_mask, None
    if os.environ.get("VTON_MASK_ALPHA_INTERSECT", "1") == "0":
        return apply_mask, None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return apply_mask, None
    was_pil = isinstance(apply_mask, Image.Image)
    m = np.asarray(apply_mask.convert("L")) if was_pil else np.asarray(apply_mask)
    if m.ndim == 3:
        m = m[..., 0]
    h, w = m.shape
    ra_src = ref_alpha.getchannel("A") if ref_alpha.mode == "RGBA" else ref_alpha.convert("L")
    ra = np.asarray(ra_src, dtype=np.uint8)
    if ra.shape != m.shape:
        ra = cv2.resize(ra, (w, h), interpolation=cv2.INTER_LINEAR)
    kx = max(9, int(0.055 * w) | 1)
    ky = max(3, int(0.008 * h) | 1)
    ra_dil = cv2.dilate(ra, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kx, ky)))
    m_u8 = m.astype(np.uint8)
    denom = max(float(m_u8.mean()), 1.0)
    # 错位守卫先按原始 mask 判(并集前): alpha 与主体 mask 完全错开时
    # kept≈0 → fail-open, 不信任 warp 形。
    kept = float(np.minimum(m_u8, ra_dil).mean()) / denom
    if kept < min_keep:
        return apply_mask, None
    # 实心面料区并入 mask(守卫通过后): 宽松款(04469 罩衫)面料比模特身体
    # 宽, 腰侧面料悬垂在人体剪影之外 —— 只按主体剪影裁 mask 会把悬垂
    # 面料裁掉, blend 贴回底图白背景 = "腰间两边泛白"(实测 mask 覆盖
    # 0.00 vs ref alpha 1.00)。alpha>200 的实心衣料允许贴入。
    strict = ((ra > 200).astype(np.uint8)) * 255
    out = np.minimum(np.maximum(m_u8, strict), ra_dil)
    if was_pil:
        return Image.fromarray(out), kept
    return out, kept


# 穿着贴合层:让衣服"穿在身上"而非"贴在身上"——面料随身材垂坠、起褶皱。
FIT_PROMPTS = {
    "fitted": (
        "the garment is worn on the model's body in a slim fit, fabric hugging the figure with "
        "tight, natural wrinkles and folds that follow the body's silhouette"
    ),
    "regular": (
        "the garment is worn on the model's body in a regular fit, fabric draping naturally over "
        "the figure with realistic wrinkles and folds that conform to the body shape"
    ),
    "loose": (
        "the garment is worn on the model's body in a relaxed loose fit, fabric hanging softly "
        "over the figure with gentle natural folds"
    ),
    "oversized": (
        "the garment is worn on the model's body in an oversized fit, fabric draping wide and "
        "loose over the figure with deep natural folds"
    ),
}

FIT_NEGATIVE = (
    "flat paper-doll garment overlay, garment floating off the body, no fabric drape, stiff flat cloth pasted on skin"
)


def sdxl_available() -> tuple[bool, str]:
    # 目录存在还不够:校验关键模型文件,避免"看似可用实则加载即炸"。
    required = [
        ("sdxl_base", "model_index.json"),
        ("controlnet_openpose", "config.json"),
        ("controlnet_canny", "config.json"),
    ]
    for d, f in required:
        if not (SDXL_DIR / d).exists():
            return False, f"Missing weights: {d}"
        if not (SDXL_DIR / d / f).exists():
            return False, f"Missing {f} in {d}"
    try:
        import torch
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "No GPU available"
    return True, "ok"


_IMAGE_PROJ_BASE_KEY = "image_proj.proj.weight"
_IMAGE_PROJ_PLUS_KEY = "image_proj.layers.0.0.norm1.weight"


def _read_image_encoder_dim(ip_adapter_dir: Path) -> int | None:
    """读取本地 image_encoder 的输出维度(projection_dim,缺失时用 hidden_size)。"""
    config_path = Path(ip_adapter_dir) / "image_encoder" / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    for key in ("projection_dim", "hidden_size"):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _infer_ip_adapter_input_dim(weight_path: Path) -> int | None:
    """从 IP-Adapter 权重的 image_proj 推断其期望的图像 embedding 维度。

    base 变体的 image_proj.proj.weight 形状为 [out, in],取 in;
    plus/plus-face 变体的 image_proj.layers.0.0.norm1.weight 为 1-D,长度即维度。
    """
    try:
        if weight_path.suffix.lower() == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(weight_path), framework="numpy", device="cpu") as handle:
                keys = set(handle.keys())
                if _IMAGE_PROJ_PLUS_KEY in keys:
                    shape = tuple(handle.get_slice(_IMAGE_PROJ_PLUS_KEY).get_shape())
                    if len(shape) == 1:
                        return int(shape[0])
                if _IMAGE_PROJ_BASE_KEY in keys:
                    shape = tuple(handle.get_slice(_IMAGE_PROJ_BASE_KEY).get_shape())
                    if len(shape) == 2:
                        return int(shape[1])
            return None

        if weight_path.suffix.lower() == ".bin":
            import torch

            state = torch.load(str(weight_path), map_location="cpu")
            if not isinstance(state, dict):
                return None
            # 兼容把权重包在 state_dict/model 下的 .bin。
            for wrapper in ("state_dict", "model"):
                nested = state.get(wrapper)
                if isinstance(nested, dict):
                    state = nested
                    break
            for key, is_1d in ((_IMAGE_PROJ_PLUS_KEY, True), (_IMAGE_PROJ_BASE_KEY, False)):
                tensor = state.get(key)
                shape = getattr(tensor, "shape", None)
                if shape is None:
                    continue
                if is_1d and len(shape) == 1:
                    return int(shape[0])
                if not is_1d and len(shape) == 2:
                    return int(shape[1])
    except Exception:  # 权重文件损坏/格式不符:跳过该候选,不阻断其它
        # 损坏/格式不符的权重文件跳过,不能让它阻断其它候选。
        logger.debug("Could not inspect IP-Adapter weight %s", weight_path, exc_info=True)
    return None


def _ip_adapter_variant_rank(weight_path: Path) -> int:
    """plus 变体优先于 plus-face,两者都优先于 base。"""
    name = weight_path.name.lower()
    if "plus-face" in name:
        return 1
    if "plus" in name:
        return 0
    return 2


def _select_ip_adapter_weight(ip_adapter_dir: Path) -> tuple[Path | None, str]:
    """选择与本地 image_encoder 输出维度配套的 IP-Adapter 权重。

    返回 (权重路径, 降级原因)。找不到任何权重时原因为空字符串;
    有候选但维度不配套时返回 None 和可直接写入 ip_adapter_reason 的说明。
    """
    ip_adapter_dir = Path(ip_adapter_dir)
    # .safetensors 优先;只有在没有匹配项时才回退到 .bin。
    groups = (
        sorted(ip_adapter_dir.glob("*.safetensors")),
        sorted(ip_adapter_dir.glob("*.bin")),
    )
    # 没有安装任何权重时不读取 encoder 配置,保持「未安装 ≠ 降级」的语义。
    if not any(groups):
        return None, ""

    encoder_dim = _read_image_encoder_dim(ip_adapter_dir)
    if encoder_dim is None:
        config_path = ip_adapter_dir / "image_encoder" / "config.json"
        return None, f"Cannot read image_encoder output dim from {config_path}; skipping IP-Adapter"

    inspected: list[tuple[Path, int]] = []
    for paths in groups:
        group_inspected = [(path, dim) for path in paths if (dim := _infer_ip_adapter_input_dim(path)) is not None]
        inspected.extend(group_inspected)
        matches = [path for path, dim in group_inspected if dim == encoder_dim]
        if matches:
            matches.sort(key=lambda path: (_ip_adapter_variant_rank(path), path.name.lower()))
            return matches[0], ""

    if not inspected:
        return None, f"IP-Adapter weights in {ip_adapter_dir} expose no readable image_proj dims; skipping IP-Adapter"

    expected_dims = sorted({dim for _, dim in inspected})
    if len(expected_dims) == 1:
        reason = (
            f"IP-Adapter weight image_proj expects {expected_dims[0]}-dim embeddings "
            f"but local image_encoder outputs {encoder_dim}"
        )
    else:
        dims = ", ".join(str(dim) for dim in expected_dims)
        reason = f"IP-Adapter weights expect one of {dims}-dim embeddings but local image_encoder outputs {encoder_dim}"
    return None, reason


def _pick_ip_adapter_weight(ip_adapter_dir: Path) -> Path | None:
    """纯函数:返回与 image_encoder 配套的权重路径,无匹配时返回 None。"""
    return _select_ip_adapter_weight(ip_adapter_dir)[0]


def ip_adapter_available() -> bool:
    """IP-Adapter 能否真正加载:匹配的权重文件与 image_encoder(CLIP vision)均须存在。"""
    weight_ok = _pick_ip_adapter_weight(SDXL_DIR / "ip_adapter") is not None
    image_encoder_dir = SDXL_DIR / "ip_adapter" / "image_encoder"
    enc_ok = (image_encoder_dir / "config.json").is_file() and (
        (image_encoder_dir / "model.safetensors").is_file() or (image_encoder_dir / "pytorch_model.bin").is_file()
    )
    return weight_ok and enc_ok


_engine_status_cache: dict | None = None


def engine_status(force: bool = False) -> dict:
    """SDXL 引擎资产自检(纯文件系统, 不加载 GPU/不联网)。

    供 download_weights.sh 校验与开机 depth P0 回归确认用。返回各关键权重
    的在位布尔 + depth 链是否完整(controlnet_depth 模型 + dpt midas ckpt)。

    进程级缓存(远端复核#5: 此前 depth 未加载时**每张图**重扫一遍磁盘):
    权重在一次进程生命周期内不会热更, 结果稳定; CLI/独立进程每次都是新进程,
    不受缓存影响。force=True 可显式重扫。
    """
    global _engine_status_cache
    if _engine_status_cache is not None and not force:
        return dict(_engine_status_cache)

    def _has(p: Path) -> bool:
        return p.exists()

    annotators = SDXL_DIR / "annotators"
    depth_dir = SDXL_DIR / "controlnet_depth"
    depth_model = (depth_dir / "config.json").is_file()
    depth_midas = (annotators / "dpt_hybrid-midas-501f0c75.pt").is_file()
    _engine_status_cache = {
        "engine": "sdxl",
        "sdxl_base": _has(SDXL_DIR / "sdxl_base" / "unet"),
        "controlnet_openpose": _has(SDXL_DIR / "controlnet_openpose" / "config.json"),
        "controlnet_canny": _has(SDXL_DIR / "controlnet_canny" / "config.json"),
        "controlnet_depth": depth_model,
        "depth_midas_ckpt": depth_midas,
        # depth 只有「ControlNet 模型 + MiDaS annotator ckpt」两者都在才真能生效;
        # 缺任一都会静默退 pose+canny 或灰图@scale0(pipeline audit P0-5/P0-6)。
        "depth_ready": depth_model and depth_midas,
        "ip_adapter": ip_adapter_available(),
        "lora": _has(SDXL_DIR / "lora" / "adapter_model.safetensors"),
        "openpose_annotator": _has(annotators / "body_pose_model.pth"),
        "rtmpose": _has(_REPO_ROOT / "models" / "rtmpose" / "rtmpose-m_coco.pth"),
        "mediapipe": _has(_REPO_ROOT / "models" / "mediapipe" / "pose_landmarker_full.task"),
        "u2net_seg": _has(_REPO_ROOT / "models" / "seg" / "u2net.onnx"),
        "sdxl_inpaint": _has(SDXL_DIR / "sdxl_inpaint" / "model_index.json"),
    }
    return dict(_engine_status_cache)


_pose_detector = None
_canny_detector = None
_depth_detector = None


def _get_openpose_control(image: Image.Image) -> Image.Image:
    global _pose_detector
    from controlnet_aux import OpenposeDetector

    if _pose_detector is None:
        # 懒加载加锁(NEW-H3 残留):并发首载会重复构建检测器,用 _PIPE_LOCK 单飞。
        with _PIPE_LOCK:
            if _pose_detector is None:
                annotator_dir = SDXL_DIR / "annotators"
                if annotator_dir.exists():
                    _pose_detector = OpenposeDetector.from_pretrained(str(annotator_dir))
                else:
                    _pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    return _pose_detector(image)


def _get_canny_control(image: Image.Image) -> Image.Image:
    global _canny_detector
    from controlnet_aux import CannyDetector

    if _canny_detector is None:
        # 懒加载加锁(NEW-H3 残留):并发首载会重复构建检测器,用 _PIPE_LOCK 单飞。
        with _PIPE_LOCK:
            if _canny_detector is None:
                _canny_detector = CannyDetector()
    return _canny_detector(image, low_threshold=100, high_threshold=200, detect_resolution=1024, image_resolution=1024)


def _get_depth_control(image: Image.Image) -> Image.Image:
    """人体深度图:换衣贴合体型、产生布料褶皱与立体感的关键条件(贴合改进)。"""
    global _depth_detector
    from controlnet_aux import MidasDetector

    if _depth_detector is None:
        # 懒加载加锁:并发首载会重复构建检测器,用 _PIPE_LOCK 单飞。
        with _PIPE_LOCK:
            if _depth_detector is None:
                annotator_dir = SDXL_DIR / "annotators"
                if annotator_dir.exists():
                    _depth_detector = MidasDetector.from_pretrained(str(annotator_dir))
                else:
                    _depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
    # 显式 1024(pipeline audit 行53):MidasDetector 默认 detect/image_resolution=512,
    # 而 canny 用 1024 → depth 半分辨率再上采样, 贴合收益打折。与 canny 对齐到 1024。
    return _depth_detector(image, detect_resolution=1024, image_resolution=1024)


def _load_trained_lora(pipe, lora_dir: Path) -> str | None:
    import sys

    if str(_TRAINING_DIR) not in sys.path:
        sys.path.insert(0, str(_TRAINING_DIR))
    from lora_io import load_trained_lora, lora_weight_path

    if lora_weight_path(lora_dir) is None:
        return None
    return load_trained_lora(pipe, lora_dir)


def _remove_background(image: Image.Image) -> Image.Image:
    try:
        from rembg import remove

        return remove(image)
    except ImportError:
        return image


def fit_square(im: Image.Image, side: int, pad=(255, 255, 255)) -> Image.Image:
    """保宽高比缩进 square,短边补白。VTON_AR=0 时回退硬拉伸。"""
    if os.environ.get("VTON_AR", "1") == "0":
        return im.resize((side, side), Image.Resampling.LANCZOS)
    w, h = im.size
    scale = side / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    canvas = Image.new("RGB", (side, side), pad)
    canvas.paste(im.resize((nw, nh), Image.Resampling.LANCZOS), ((side - nw) // 2, (side - nh) // 2))
    return canvas


def _apply_lora_scale(pipe) -> None:
    """小数据 r16 LoRA 默认 0.7,避免 1.0 过拟合模板句。VTON_LORA 可覆盖。"""
    raw = os.environ.get("VTON_LORA", "0.7").strip()
    try:
        scale = float(raw)
    except ValueError:
        logger.warning("Invalid VTON_LORA=%r; keeping adapter default", raw)
        return
    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters(["default"], adapter_weights=[scale])
            logger.info("LoRA scale -> %s (env VTON_LORA)", scale)
            return
        except Exception:  # diffusers 版本差异:set_adapters 缺失时走 set_lora_scale
            logger.debug("pipe.set_adapters failed; trying fuse-less scale", exc_info=True)
    if hasattr(pipe, "set_lora_scale"):
        pipe.set_lora_scale(scale)
        logger.info("LoRA scale -> %s via set_lora_scale", scale)


def align_garment_to_pose(
    garment: Image.Image,
    audience: str,
    ethnicity: str,
    garment_type: str,
    canvas: int = 1024,
    sleeve_len: str | None = None,
    pose: dict | None = None,
    hem: str | None = None,
    arm_cols: tuple[int, int, int, int] | None = None,
) -> tuple[Image.Image, Image.Image | None]:
    """把平铺衣"掰"到人体姿态:按目标人群肩宽/轮廓缩放并做 2D 网格变形,
    得到一张"正穿在人身上"的服装参考图,供 SDXL IP-Adapter 使用。

    这是 VITON 几何对齐步骤的纯 CPU 简化版——把衣服先贴到目标体型,
    再让生成器在已对齐的参考上重绘纹理,显著提升衣服在身上的位置/比例
    准确性,尤其解决"衣服飘在身外/袖长不合/比例错位"。

    流程:crop 内容 -> fit_garment_to_body(按肩宽等比缩放) ->
    _warp_garment(水平轮廓收窄/展宽 + 垂直锚点对齐) -> 白底 RGB。

    垂直对齐(VTON_WARP_VERTICAL, 默认 1.0)按 GARMENT_BOX 调整后的躯干框
    映射领口→颈部、下摆→腰部(上衣)或膝部(连衣裙)或髋部(下装),
    修正仅按肩宽缩放导致的 logo/花纹垂直偏位。
    上衣 hem=mid/long 时把框底从腰线拉到髋/大腿(与 clothing_mask 截线同口径),
    避免中长款被压成齐腰(04469 下摆丢失/漏肚脐)。arm_cols(人物剪影左右臂列,
    compose 用 _arm_column_bands 算好传入)决定袖条横向落位, None 时 refit
    退回画布比例。

    VTON_WARP=0 时关闭(回退原平铺图);VTON_WARP_STRENGTH 可覆盖默认强度。

    返回 (白底 RGB 参考图, 带 alpha 的 RGBA 参考图或 None)。
    第二项供 pattern-lock 贴回用:alpha=0 的画布白底区域不被贴进模特
    (修"袖子白晕/脖子灰带"——白底参考图整块贴入 mask 内无衣料区)。
    未 warp(回退/关闭)时第二项为 None,调用方退回白底启发式。
    """
    if os.environ.get("VTON_WARP", "1") == "0":
        return garment, None
    try:
        from engine import (
            GARMENT_BOX,
            GARMENT_WARP_STRENGTH,
            _warp_garment,
            fit_garment_to_body,
            garment_alpha,
            refit_sleeves_to_arms,
            torso_box,
        )
    except ImportError:
        logger.debug("align_garment_to_pose: engine funcs unavailable; passthrough")
        return garment, None

    box = torso_box(audience, ethnicity)
    adj = GARMENT_BOX.get(garment_type, GARMENT_BOX["upper"])
    box["y"] = box["y"] + adj["y"] * box["h"]
    box["h"] = box["h"] * adj["h"]
    # 品类=upper 但衣料实测过腰: 框底仍钉在腰 → 中长款被压成齐腰。
    # 与 clothing_mask 截线同口径拉长(mid +0.35 相对躯干框, long +0.50)。
    if garment_type == "upper" and hem in ("mid", "long"):
        _hem_scale = 1.35 if hem == "mid" else 1.50
        box["h"] = box["h"] * _hem_scale
        logger.info(
            "warp box hem=%s type=upper → h*=%.2f (y=%.3f bottom=%.3f)",
            hem,
            _hem_scale,
            box["y"],
            box["y"] + box["h"],
        )
    # 上衣下摆兜底盖过裤头(waist 分类也至少到髋+0.03): 平铺图拍法(折袖/
    # 压缩挂拍)使 hem 分类偏短(c10 实测 09263/09290 分类 waist → 框底
    # 0.568c, alpha 下摆在肚脐上 → 漏肚脐/"只盖裤头两边"), 用户预期是
    # 上衣盖住裤头。取 pose 髋点(缺省 0.643)与 mid/long 拉长后框底的
    # 较大者。
    if garment_type in ("upper", "auto", "tunic"):
        _hip = 0.0
        _p_hip = pose.get("hip") if pose else None
        if _p_hip and len(_p_hip) >= 3 and _p_hip[2] >= 0.3:
            _hip = float(_p_hip[1])
        _hip_bottom = (_hip if _hip > 0 else 0.643) + 0.03
        if box["y"] + box["h"] < _hip_bottom:
            logger.info(
                "warp box hem floor: bottom %.3f -> %.3f (cover waistband)",
                box["y"] + box["h"],
                _hip_bottom,
            )
            box["h"] = _hip_bottom - box["y"]
    box["width_key"] = adj.get("w", "shoulder")
    base_strength = GARMENT_WARP_STRENGTH.get(garment_type, 0.42)
    raw = os.environ.get("VTON_WARP_STRENGTH", "").strip()
    try:
        strength = float(raw) if raw else base_strength
    except ValueError:
        strength = base_strength
        logger.warning("Invalid VTON_WARP_STRENGTH=%r; using default", raw)

    # fit_garment_to_body 期望能取到衣料 alpha(RGBA)或自动估计(RGB)。
    g = garment.convert("RGBA")
    if int(g.split()[-1].getextrema()[0]) >= 250:
        # 不透明图:用 garment_alpha 补一层 alpha 供 bbox/warp 使用。
        g.putalpha(garment_alpha(garment))

    fitted = fit_garment_to_body(g, box, canvas, canvas)
    if fitted.mode != "RGBA":
        fitted = fitted.convert("RGBA")
    warped = _warp_garment(fitted, box, strength)
    if warped.mode != "RGBA":
        warped = warped.convert("RGBA")

    # 统一到 canvas 方画布: 窄长平铺按宽缩放会生成超高画布(04469 实测
    # 1024x1374)。旧代码 RGB 参考图对超高图"裁切"到 canvas、alpha 却被下游
    # "resize"回 canvas —— 同一件衣服两套行坐标(1:1.343 错位): 贴回时衣料
    # 内容整体低于 alpha 形 34%, 下摆被 alpha 切掉(漏肚脐/腰间没盖), 特征
    # (结/口袋)错位(结跑到胸口)。这里统一 resize 到 canvas 方画布后再贴
    # 袖条, RGB 参考图/alpha/袖条三者行坐标完全一致; 垂直 remap 的目标行
    # 按画布比例表达, resize 后即落在 box 行上, 不产生二次压缩。
    if warped.size != (canvas, canvas):
        warped = warped.resize((canvas, canvas), Image.Resampling.LANCZOS)

    # 袖几何对齐: 2D warp 折不下袖子, 这里把平铺衣的左右袖(横向摊开/向下
    # 折放自适应, 短袖只贴肘上段)合成到画布手臂列, 让参考图在手臂上有真实
    # 袖纹(修"长袖变无袖/短袖无袖")。竖直锚点用 pose 的肩/腕归一化 y(无 pose
    # 退回全身比例), 横向按全身正面基模标定; RGB 平铺图先补 garment_alpha,
    # 白底不整块贴入。VTON_SLEEVE_REFIT=0 一键关闭本步(A/B 用)。
    if sleeve_len in ("long", "short") and os.environ.get("VTON_SLEEVE_REFIT", "1") != "0":
        try:
            warped = refit_sleeves_to_arms(warped, garment, canvas, pose, sleeve_len, arm_cols=arm_cols)
        except Exception:
            logger.exception("refit_sleeves_to_arms failed; keeping torso-only reference")

    out = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    out.paste(warped, (0, 0), warped)
    return out, warped


def crop_garment_to_content(
    image: Image.Image,
    pad_frac: float = 0.03,
    min_content_frac: float = 0.04,
) -> Image.Image:
    """把服装平铺图裁到内容边界,去掉大片白底/透明边,让 Canny 与 IP-Adapter
    的参考聚焦在服装本体上(颜色/花纹保真度更高)。

    - RGBA 输入按 alpha 判内容;RGB 输入把四角 24x24 中值当背景估计,
      与背景 L1 距离 > 40 判为内容像素。
    - 内容占比 < min_content_frac(检测不到清晰主体,如纯色图/全身浅色且无白边)
      或找不到连通区时,原样返回(安全 no-op,不会比现状更差)。
    """
    import numpy as np

    arr = np.array(image.convert("RGBA"))
    alpha = arr[..., 3]
    # 只有"真"透明图(PNG 平铺)才按 alpha 判内容;不透明 RGB 升成 RGBA 后
    # alpha 恒 255,必须走背景差分,否则会把整图误判成"全是内容"不裁。
    has_transp = image.mode in ("RGBA", "LA") and int(alpha.min()) < 200
    if has_transp:
        mask = alpha > 20
    else:
        rgb = arr[..., :3].astype(np.int16)
        c = 24
        corners = np.concatenate(
            [
                rgb[:c, :c].reshape(-1, 3),
                rgb[:c, -c:].reshape(-1, 3),
                rgb[-c:, :c].reshape(-1, 3),
                rgb[-c:, -c:].reshape(-1, 3),
            ]
        )
        bg = np.median(corners, axis=0)
        mask = np.abs(rgb - bg).sum(axis=2) > 40

    if float(mask.mean()) < min_content_frac:
        return image  # 检测不到清晰主体 → 原样返回(no-op)

    h, w = mask.shape
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return image
    r0 = int(np.where(rows)[0][0])
    r1 = int(np.where(rows)[0][-1])
    c0 = int(np.where(cols)[0][0])
    c1 = int(np.where(cols)[0][-1])

    p = int(min(h, w) * pad_frac)
    r0 = max(0, r0 - p)
    c0 = max(0, c0 - p)
    r1 = min(h - 1, r1 + p)
    c1 = min(w - 1, c1 + p)

    out = arr[r0 : r1 + 1, c0 : c1 + 1]
    if has_transp:
        return Image.fromarray(out, "RGBA")
    return Image.fromarray(out[..., :3], "RGB")


_FP16_WEIGHT_FILES = ("diffusion_pytorch_model.fp16.safetensors", "model.fp16.safetensors")
_PIPELINE_COMPONENTS = {
    "unet": "diffusion_pytorch_model",
    "vae": "diffusion_pytorch_model",
    "text_encoder": "model",
    "text_encoder_2": "model",
}


def _fp16_variant_kwargs(model_dir: Path) -> dict[str, str]:
    """仅当本地存在 fp16 权重变体时传 variant,避免硬编码 fp16 导致 OSError。

    diffusers 会把 variant 拼进权重文件名:ControlNet/UNet/VAE 为
    diffusion_pytorch_model.fp16.safetensors,CLIP text encoder 为
    model.fp16.safetensors。本地目录只有非 variant 文件时必须省略 variant。
    """
    if any((model_dir / name).is_file() for name in _FP16_WEIGHT_FILES):
        return {"variant": "fp16"}
    return {}


def _pipeline_fp16_variant_kwargs(pipeline_dir: Path) -> dict[str, str]:
    """pipeline 只有在所有带权重组件的 fp16 变体都存在时才启用 variant。

    若任一组件缺失 fp16 变体,diffusers 仍会为该组件拼出 .fp16.safetensors
    路径并报 OSError;因此宁可整条 pipeline 回退到非 variant 权重。
    """
    has_variant = False
    for subfolder, stem in _PIPELINE_COMPONENTS.items():
        component_dir = pipeline_dir / subfolder
        if not component_dir.is_dir():
            continue
        variant_file = component_dir / f"{stem}.fp16.safetensors"
        if not variant_file.is_file():
            # 该组件没有 fp16 变体 -> diffusers 会拼出不存在的路径,整条 pipeline 都不能传 variant。
            return {}
        has_variant = True
    return {"variant": "fp16"} if has_variant else {}


def _inpaint_weights_ready() -> bool:
    return (SDXL_DIR / "sdxl_inpaint" / "model_index.json").is_file()


def _mask_to_pil(mask, size: tuple[int, int]) -> Image.Image:
    """clothing_mask / ndarray / PIL → 生成分辨率的 L 图;None 则全白(整图重绘)。"""
    import numpy as np

    if mask is None:
        return Image.new("L", size, 255)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    im = Image.fromarray(arr.astype("uint8"), "L")
    if im.size != size:
        im = im.resize(size, Image.Resampling.BILINEAR)
    return im


def _collect_refit_anchors(
    garment_img: Image.Image,
    cloth_ref: Image.Image,
    human_img: Image.Image,
    garment_extent: dict | None,
    pose: dict | None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """收集贴印锚点: 平铺图(领口/肩/下摆) -> 人体(肩/髋/膝)。

    返回 (src_pts, dst_pts)。任一侧缺失返回 ([], []) 让调用方回退网格。
    """
    try:
        import numpy as _np

        g_arr = _np.asarray(garment_img.convert("RGBA"))
        alpha = g_arr[..., 3]
        # 平铺图领口 = alpha 顶部中央; 下摆 = alpha 底部中央; 肩 = 领口两侧
        cols = _np.where(alpha.sum(axis=0) > 20)[0]
        if len(cols) < 8:
            return [], []
        mid_x = int((cols.min() + cols.max()) / 2)
        rows = _np.where(alpha.sum(axis=1) > 20)[0]
        top_y = int(rows[0])
        bot_y = int(rows[-1])
        shoulder_x_l = int(cols.min() + (mid_x - cols.min()) * 0.3)
        shoulder_x_r = int(cols.max() - (cols.max() - mid_x) * 0.3)
        src_pts = [
            (float(mid_x), float(top_y)),  # 领口
            (float(shoulder_x_l), float(top_y)),  # 左肩
            (float(shoulder_x_r), float(top_y)),  # 右肩
            (float(mid_x), float(bot_y)),  # 下摆
        ]

        # 人体锚点: 从 pose 关键点取
        if pose is None:
            return [], []
        # pose 归一化 [0,1]; 映射到 human_img 像素
        hw, hh = human_img.size
        shoulder = pose.get("shoulder")
        hip = pose.get("hip")
        knee = pose.get("knee")
        if not (shoulder and hip):
            return [], []
        sy = shoulder[1] * hh
        hy = hip[1] * hh
        ky = knee[1] * hh if knee else hy + (hy - sy) * 0.5
        sx = shoulder[0] * hw
        dst_pts = [
            (sx, sy + (hy - sy) * 0.15),  # 领口(略低于肩)
            (sx - (sx - 0.05 * hw), sy),  # 左肩(外扩)
            (sx + (hw - sx) * 0.6, sy),  # 右肩(外扩)
            (sx, max(hy, ky * 0.85)),  # 下摆(髋或膝)
        ]
        return src_pts, dst_pts
    except Exception:
        logger.exception("refit anchor collection failed")
        return [], []


def _build_sdxl_infer_kwargs(
    *,
    prompt: str,
    negative_prompt: str,
    control_images,
    control_scales,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    generator,
    ip_adapter_image=None,
    inpaint_mode: bool = False,
    init_image: Image.Image | None = None,
    mask_image: Image.Image | None = None,
    strength: float = 0.99,
) -> dict:
    """组装 SDXL 采样参数:inpaint 用 image=底座 / mask=衣身 / control_image=姿态。"""
    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "generator": generator,
        "output_type": "pil",
        "controlnet_conditioning_scale": control_scales,
    }
    if inpaint_mode:
        kwargs["image"] = init_image
        kwargs["mask_image"] = mask_image
        kwargs["control_image"] = control_images
        kwargs["strength"] = float(strength)
    else:
        kwargs["image"] = control_images
    if ip_adapter_image is not None:
        kwargs["ip_adapter_image"] = ip_adapter_image
    return kwargs


def _get_sdxl_pipe(device, dtype):
    # 首次加载必须单飞(NEW-H3):并发请求同时见 pipe=None 会重复构建管线撑爆显存。
    with _PIPE_LOCK:
        if _sdxl_state["pipe"] is not None:
            return _sdxl_state["pipe"], _sdxl_state["ip_adapter_loaded"], _sdxl_state["ip_adapter_reason"]

        import torch
        from diffusers import (
            ControlNetModel,
            DDPMScheduler,
            StableDiffusionXLControlNetPipeline,
        )

        try:
            from diffusers import StableDiffusionXLControlNetInpaintPipeline as _InpaintPipeCls
        except ImportError:
            _InpaintPipeCls = None

        inpaint_dir = SDXL_DIR / "sdxl_inpaint"
        inpaint_mode = bool(_InpaintPipeCls is not None and _inpaint_weights_ready())
        pipe_cls = _InpaintPipeCls if inpaint_mode else StableDiffusionXLControlNetPipeline
        pipe_root = inpaint_dir if inpaint_mode else (SDXL_DIR / "sdxl_base")
        sched_root = pipe_root if (pipe_root / "scheduler").is_dir() else (SDXL_DIR / "sdxl_base")

        scheduler = DDPMScheduler.from_pretrained(str(sched_root), subfolder="scheduler", local_files_only=True)

        # thibaud/controlnet-openpose-sdxl-1.0 ships the weights as a single-file
        # checkpoint (OpenPoseXL2.safetensors) rather than a diffusers subfolder layout.
        openpose_ckpt = SDXL_DIR / "controlnet_openpose"
        openpose_single = openpose_ckpt / "OpenPoseXL2.safetensors"
        if openpose_single.exists():
            controlnet_pose = ControlNetModel.from_single_file(str(openpose_single), torch_dtype=dtype)
        else:
            controlnet_pose = ControlNetModel.from_pretrained(
                str(openpose_ckpt), torch_dtype=dtype, use_safetensors=True, local_files_only=True
            )

        # 本地 canny 目录可能只有非 variant 权重;有 fp16 变体才传 variant。
        def _load_canny():
            return ControlNetModel.from_pretrained(
                str(SDXL_DIR / "controlnet_canny"),
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
                **_fp16_variant_kwargs(SDXL_DIR / "controlnet_canny"),
            )

        # 贴合改进:默认只加载 pose+depth(2 个 controlnet)。3 个 controlnet 常驻
        # 会超出 24GB 显存,故不默认全加载;depth 缺失或失败时退回 pose+canny。
        depth_dir = SDXL_DIR / "controlnet_depth"
        depth_loaded = False
        if (depth_dir / "config.json").exists():
            try:
                controlnets = [
                    controlnet_pose,
                    ControlNetModel.from_pretrained(
                        str(depth_dir),
                        torch_dtype=dtype,
                        use_safetensors=True,
                        local_files_only=True,
                        **_fp16_variant_kwargs(depth_dir),
                    ),
                ]
                depth_loaded = True
            except Exception:
                logger.exception("Failed to load depth ControlNet; falling back to canny")
                controlnets = [controlnet_pose, _load_canny()]
        else:
            controlnets = [controlnet_pose, _load_canny()]

        pipe = None
        try:
            pipe = pipe_cls.from_pretrained(
                str(pipe_root),
                controlnet=controlnets,
                scheduler=scheduler,
                torch_dtype=dtype,
                use_safetensors=True,
                local_files_only=True,
                **_pipeline_fp16_variant_kwargs(pipe_root),
            )
            # SDXL base VAE 需以 fp32 解码,否则 fp16 下解码溢出产生 NaN/黑图。
            # 此刻组件在 CPU 上,直接改 dtype 即可。
            pipe.vae = pipe.vae.to(torch.float32)

            lora_dir = SDXL_DIR / "lora"
            lora_loaded = False
            lora_install_failed = False
            try:
                lora_fmt = _load_trained_lora(pipe, lora_dir)
                if lora_fmt:
                    lora_loaded = True
                    logger.info("Loaded trained LoRA (%s) from %s", lora_fmt, lora_dir)
            except Exception:
                logger.exception("Failed to load trained LoRA; continuing without it")
                # 区分"未安装"(正常态, _load_trained_lora 无权重时提前返回 None 不抛异常)
                # 与"已安装但加载失败"(真降级)。lora_io 的 sys.path 注入发生在 _load_trained_lora
                # 内部(异常来自 load 阶段时 import 已成功), 故此处可复检权重在位性。
                try:
                    from lora_io import lora_weight_path

                    lora_install_failed = lora_weight_path(lora_dir) is not None
                except Exception:  # noqa: BLE001
                    lora_install_failed = False

            # 权重必须与本地 image_encoder 的输出维度配套:base(1024)在 1280 编码器上
            # 会 shape mismatch,plus/plus-face(1280)才兼容。按实际 image_proj 维度选择。
            ip_adapter_weight, ip_adapter_reason = _select_ip_adapter_weight(SDXL_DIR / "ip_adapter")
            ip_adapter_loaded = False
            if ip_adapter_weight is not None:
                try:
                    # 以本地目录为 pretrained_model_name_or_path 并显式 subfolder="",
                    # 避免回连 HF,也避免缺省 subfolder 触发 TypeError(H-05)。
                    pipe.load_ip_adapter(
                        str(ip_adapter_weight.parent),
                        subfolder="",
                        weight_name=str(ip_adapter_weight.name),
                        download_kwargs={"local_files_only": True},
                    )
                    pipe.set_ip_adapter_scale(0.9)
                    ip_adapter_loaded = True
                except Exception as exc:
                    # 权重存在但加载失败:标记降级(N4-L6),与「未安装权重」区分。
                    logger.exception("Failed to load IP-Adapter (%s); continuing without it", exc)
                    ip_adapter_reason = (
                        f"IP-Adapter weight present ({ip_adapter_weight.name}) but failed to load: {exc}"
                    )
            elif ip_adapter_reason:
                logger.warning("%s", ip_adapter_reason)
            else:
                logger.warning(
                    "No IP-Adapter weight file found in %s; SDXL runs without image prompt", SDXL_DIR / "ip_adapter"
                )

            # IP-Adapter 激活时不启用 attention slicing:SlicedAttnProcessor 会
            # 覆盖 IPAdapter 处理器,而后者需要 unet 传入的 (text, image) 元组。
            # 无 IP-Adapter 时 slicing 仍是纯内存优化,可安全启用。
            if not ip_adapter_loaded:
                pipe.enable_attention_slicing()

            # 多 controlnet(pose+canny+depth)+IP-Adapter 超出 24GB 显存:
            # offload text_encoder/VAE/IP-Adapter image_encoder 到 CPU,只留
            # unet+controlnet 在 GPU,显著降低峰值显存(贴合改进引入 depth 后)。
            pipe.enable_model_cpu_offload(device=device)
            # VAE tiling/slicing 进一步压低解码峰值显存(多 controlnet 场景);
            # 该管线在部分 diffusers 版本无此方法,缺失时静默跳过。
            for _m in ("enable_vae_slicing", "enable_vae_tiling"):
                if hasattr(pipe, _m):
                    getattr(pipe, _m)()

            _sdxl_state["pipe"] = pipe
            _sdxl_state["ip_adapter_loaded"] = ip_adapter_loaded
            _sdxl_state["ip_adapter_reason"] = ip_adapter_reason
            _sdxl_state["depth_loaded"] = depth_loaded
            _sdxl_state["lora_loaded"] = lora_loaded
            _sdxl_state["lora_install_failed"] = lora_install_failed
            _sdxl_state["inpaint_mode"] = inpaint_mode
            if inpaint_mode:
                logger.info("SDXL main path: ControlNet Inpaint (%s)", pipe_root)
            else:
                logger.warning(
                    "SDXL Inpaint weights missing or ControlNetInpaintPipeline unavailable; "
                    "main path falls back to full-frame ControlNet (face/arms will be redrawn)"
                )
            return pipe, ip_adapter_loaded, ip_adapter_reason
        except Exception:
            # 加载失败必须释放已占显存,否则残留半成品会持续占用 VRAM(NEW-H3)。
            logger.exception("Failed to build SDXL pipeline; releasing GPU memory")
            if pipe is not None:
                with contextlib.suppress(Exception):
                    pipe.to("cpu")
                _sdxl_state["pipe"] = None
                _sdxl_state["ip_adapter_loaded"] = False
                _sdxl_state["ip_adapter_reason"] = ""
                _sdxl_state["depth_loaded"] = False
                _sdxl_state["lora_loaded"] = False
                _sdxl_state["lora_install_failed"] = False
                _sdxl_state["inpaint_mode"] = False
            try:
                import gc

                del controlnets
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to release GPU memory after pipeline error")
            raise


def compose_tryon_sdxl(
    base_path: Path,
    garment_path: Path,
    audience: str,
    garment_type: str,
    long_side: int = 2048,
    background: str = "white",
    ethnicity: str = "western_europe",
    num_inference_steps: int = 30,
    fit: str = "regular",
    crop_garment: bool = True,
) -> Image.Image:
    import torch
    from PIL import Image as PILImage

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    base_img = Image.open(base_path).convert("RGB")
    garment_img = Image.open(garment_path).convert("RGB")

    # 强制输入步骤(pipeline audit P0-3/行4 + 守卫 行41): 四角 flood-fill 预白底,
    # 治"灰 cyclorama → 头后灰块"。带 changed-pixel 比例 + u2net 主体交集守卫,
    # 越界告警并回退原图(绝不静默产出灰块/被啃的图)。VTON_PREWHITEN=0 可关。
    _pw_base_img, pw_base = prewhiten_input(base_img)
    if pw_base.get("status") not in ("applied", "already_white", "disabled"):
        logger.warning("base prewhiten not applied: %s", pw_base.get("reason"))
    base_img = _pw_base_img
    # 远端复核#4: 成衣(浅面料)预白底可被 flood-fill 啃 —— 默认开(fail-closed 守卫兜底),
    # 但允许按资产类型单独关: VTON_PREWHITEN_GARMENT=0 → 成衣侧不预白, 底图侧不受影响。
    if os.environ.get("VTON_PREWHITEN_GARMENT", "1") == "0":
        _pw_g_img, pw_garment = garment_img, {"status": "disabled", "reason": "VTON_PREWHITEN_GARMENT=0"}
    else:
        _pw_g_img, pw_garment = prewhiten_input(garment_img)
    garment_img = _pw_g_img

    if crop_garment:
        garment_img = crop_garment_to_content(garment_img)

    # 贴印参考图长边跟输出 (≥ 2048), 不缩到 1024 丢印花高频。
    # SDXL 原生生成仍锁 1024; warp/贴印/IP-Adapter 语义路用高分辨率 canvas。
    target_w, target_h = 1024, 1024
    warp_canvas = max(1024, int(long_side))
    human_img = fit_square(base_img, target_w)
    cloth_img = fit_square(garment_img, target_h)

    # 衣料属性检测(袖长/衣长/宽高比):提前算,用于 auto 类型推断 + mask 裁切。
    garment_extent = None
    try:
        garment_extent = detect_garment_extent(garment_img)
        logger.info("garment extent: %s", garment_extent)
    except Exception:  # noqa: BLE001
        logger.debug("detect_garment_extent failed; using defaults")

    # auto 类型推断:从平铺衣料宽高比判断覆盖范围。
    # aspect>1.6(高瘦)→dress(一体式覆盖躯干+腿);
    # aspect<0.7(矮宽)→lower(裤子平铺宽矮);
    # 其他→upper;hem 过腰且衣身宽到底→tunic(见 infer_auto_garment_type)。
    if garment_type == "auto":
        garment_type = infer_auto_garment_type(garment_extent)
        logger.info(
            "auto garment_type -> %s (aspect=%s hem=%s body_ratio=%s)",
            garment_type,
            (garment_extent or {}).get("aspect"),
            (garment_extent or {}).get("hem"),
            (garment_extent or {}).get("body_ratio"),
        )
    # 品类 vs hem 检测互校验: 不一致大声 warning(防止"各跑各的"——品类定画布/
    # prompt, hem 定 mask 截线, 两者打架时出图必错位)。
    _hem_x = garment_extent.get("hem", "waist") if garment_extent else "waist"
    if (
        (garment_type == "tunic" and _hem_x == "waist")
        or (garment_type in ("upper", "onesie") and _hem_x in ("mid", "long"))
        or (garment_type == "dress" and _hem_x == "waist")
    ):
        logger.warning(
            "garment_type=%s conflicts with garment extent hem=%s — warp box/prompt and "
            "mask cut line may disagree; check the garment flat-lay image or pass "
            "garment_type explicitly",
            garment_type,
            _hem_x,
        )

    # mask debug tag(VTON_DEBUG_MASK 落盘区分衣物; 串行跑场景, env 共享):
    # 在此设置, 后面 clothing_mask 的阶段图(hemcut/corridor/carve/final)可拿到。
    os.environ["VTON_DEBUG_TAG"] = os.path.basename(str(garment_path))[:32]

    # 几何对齐:把平铺衣掰到目标人群姿态,作为 IP-Adapter 参考图。
    # VTON_WARP=1(默认)时,生成器看到的是"已穿在身上的衣服",而非平铺贴图,
    # 显著改善衣服在身上的位置/比例/袖长;失败或关闭时回退原平铺图。
    warp_enabled = os.environ.get("VTON_WARP", "1") != "0"
    cloth_ref = cloth_img
    cloth_ref_alpha: Image.Image | None = None
    # warp 分支内赋值; 先给默认值, 生成后袖条合成(post-hoc)也要用。
    _sleeve = (garment_extent or {}).get("sleeve")
    _pose = None
    _arm_cols = None
    if warp_enabled:
        try:
            _sleeve = garment_extent.get("sleeve") if garment_extent else None
            # 袖几何对齐需要肩/腕竖直锚点: 长袖到腕、短袖到肘上。拿不到 pose
            # 时 refit 退回全身比例(仍安全)。
            _pose = None
            if _sleeve in ("long", "short"):
                try:
                    _pose = _detect_pose_keypoints(human_img)
                except Exception:
                    logger.exception("pose for sleeve refit failed; refit will use full-body ratios")
            # 袖条横向落位: 从人物剪影自适应左右臂列(白底画布), 硬编码画布
            # 比例在"手臂贴身/letterbox 进画布"的底图上会贴到白背景上。
            _arm_cols = None
            if _sleeve in ("long", "short"):
                try:
                    from engine import _arm_column_bands

                    _arm_cols = _arm_column_bands(human_img)
                    if _arm_cols is not None and warp_canvas != target_w:
                        _scale = warp_canvas / float(target_w)
                        _arm_cols = tuple(round(v * _scale) for v in _arm_cols)
                    if _arm_cols is not None:
                        logger.info(
                            "sleeve arm bands from silhouette: L %d-%d R %d-%d",
                            _arm_cols[0],
                            _arm_cols[1],
                            _arm_cols[2],
                            _arm_cols[3],
                        )
                    else:
                        logger.info("sleeve arm bands undetected; refit falls back to canvas ratios")
                except Exception:
                    logger.exception("arm column detection failed; refit uses canvas ratios")
            cloth_ref, cloth_ref_alpha = align_garment_to_pose(
                garment_img,
                audience,
                ethnicity,
                garment_type,
                warp_canvas,
                sleeve_len=_sleeve,
                pose=_pose,
                hem=_hem_x,
                arm_cols=_arm_cols,
            )
        except Exception:
            logger.exception("align_garment_to_pose failed; using flat cloth reference")

    prompt = AUDIENCE_PROMPTS.get(audience, AUDIENCE_PROMPTS["women"])
    eth_prompt = ETHNICITY_PROMPTS.get(ethnicity, ETHNICITY_PROMPTS["western_europe"])
    prompt = f"{prompt}, {eth_prompt}"
    garment_suffix = GARMENT_PROMPT_SUFFIX.get(garment_type, "")
    if garment_suffix:
        prompt = f"{prompt}, wearing {garment_suffix}"
    # 袖型写入 prompt:长袖须盖到腕;短袖停在上臂;无袖露整臂。
    # 不写时 SDXL 常把三种袖画成同一种, 后处理 mask 也救不回。
    _sleeve_prompt = (garment_extent or {}).get("sleeve")
    if _sleeve_prompt == "long":
        prompt = (
            f"{prompt}, the garment has long sleeves: sleeve fabric fully covers "
            f"each arm down to the wrist, no bare forearms"
        )
    elif _sleeve_prompt == "short":
        prompt = f"{prompt}, the garment has short sleeves ending at the upper arm, forearms and hands remain bare"
    elif _sleeve_prompt == "sleeveless":
        prompt = f"{prompt}, the garment is sleeveless, both arms fully bare from the shoulder down, no sleeve fabric"
    # 穿着贴合:衣服必须"穿"在模特身上——面料垂坠、褶皱随身材(不是平铺贴图)。
    prompt = f"{prompt}, {FIT_PROMPTS.get(fit, FIT_PROMPTS['regular'])}"
    # 摄影质感(防假人感):写实 dslr 质感 + 面料细节。
    prompt = f"{prompt}, {PHOTO_QUALITY}"
    # 额外 prompt 钩子(如"长袖须随手臂到手腕"):仅当设置 VTON_PROMPT_HINT 时追加,默认不变。
    _hint = os.environ.get("VTON_PROMPT_HINT", "").strip()
    if _hint:
        prompt = f"{prompt}, {_hint}"
    if background == "white":
        # 电商主图:背景要求纯白无缝影棚、无地面/阴影(后处理还会强制压白)。
        prompt = f"{prompt}, isolated on a pure white seamless studio backdrop, no floor, no shadow"
    # 袖型反向约束: 检测到袖型时压掉对立袖型(04469 短袖被画成无袖、
    # 09290 长袖被画成肘部截断的实测根因之一 —— 正向袖型词跟随弱)。
    # 袖型词放最前 + 短语去重: negative 走 CLIP tokenizer 77-token 截断
    # (truncation=True 静默截断, 无日志), 旧序追加末尾时袖型词全部被截
    # 掉(encoded=0, C3 与 C2 逐像素相同的实测根因)。去重后全 audience
    # 加全袖型组合 ≤77 token, 袖型词与核心禁项全部编码。
    _neg_sleeve = SLEEVE_NEGATIVE_PROMPTS.get(_sleeve_prompt or "")
    negative_prompt = _dedupe_negative(
        [
            _neg_sleeve,
            AUDIENCE_NEGATIVE.get(audience, AUDIENCE_NEGATIVE["women"]),
            FIT_NEGATIVE,
            COMMON_NEGATIVE,
        ]
    )

    pipe, ip_adapter_loaded, ip_adapter_reason = _get_sdxl_pipe(device, dtype)

    # 降级信号合成(pipeline audit 行8):由 IP-Adapter + depth + LoRA + mask 各环节
    # 的失败备注共同合成, 不再只编码 IP-Adapter。每个环节失败时往这里塞原因。
    degraded_reasons: list[str] = []
    if ip_adapter_reason:
        degraded_reasons.append(ip_adapter_reason)
    # LoRA 语义(远端复核#2): 「未安装」(机器没训过权重)≠ 降级, 每张图不标;
    # 只有「已安装但加载失败」才计入样式降级(与 L223「未安装 ≠ 降级」对齐)。
    if _sdxl_state.get("lora_install_failed"):
        degraded_reasons.append("trained LoRA present but failed to load (style will not match)")

    _ipa = os.environ.get("VTON_IPA")
    if ip_adapter_loaded and _ipa:
        pipe.set_ip_adapter_scale(float(_ipa))
        logger.info("IP-Adapter scale overridden -> %s (env VTON_IPA)", float(_ipa))

    if _sdxl_state.get("lora_loaded"):
        _apply_lora_scale(pipe)

    control_pose = _get_openpose_control(human_img)

    # 贴合改进:depth 存在时用 pose+depth(深度图约束衣服包裹体型、产生立体
    # 褶皱);depth 不可用(depth_loaded=False)时退回 pose+canny。两者都保证
    # images 数量与已加载的 controlnet 数量一致。
    control_images = [control_pose]
    control_scales = [0.8]
    if _sdxl_state.get("depth_loaded"):
        try:
            control_images.append(_get_depth_control(human_img))
            control_scales.append(0.95)
        except Exception:
            # fail-loud(pipeline audit P0-6/行23 + 远端复核#1): 第二条 ControlNet 此时是
            # depth 模型(L622-631)。canny 边缘图对 depth 模型是错模态输入(负贡献),
            # 不能喂 canny@0.55 —— 改喂中性灰度图 @ scale 0.0(零贡献, 等效关掉 depth 分支),
            # 并记录降级原因。canny@品类 scale 只用于第二条确为 canny 模型的分支(L883)。
            logger.exception("Depth control FAILED at runtime; depth input zeroed (DEGRADED)")
            degraded_reasons.append(
                "depth ControlNet loaded but MiDaS depth control failed at runtime; "
                "depth input zeroed (grayscale @ scale 0.0) — garment depth structure not enforced"
            )
            control_images.append(cloth_ref.convert("L"))
            control_scales.append(0.0)
    else:
        # depth 未加载。资产其实齐(controlnet_depth 模型 + dpt midas 都在)却没加载
        # = 真问题(P0-5): 标降级。资产本就缺 = 深度可选, 只 loud warning 不算降级。
        if engine_status().get("depth_ready"):
            degraded_reasons.append(
                "depth assets present (controlnet_depth + dpt midas) but depth_loaded=False — depth silently inactive"
            )
            logger.error(
                "depth assets are present but the pipeline did NOT load the depth ControlNet; "
                "the advertised dual-ControlNet is running pose+canny. INVESTIGATE."
            )
        else:
            logger.warning(
                "depth ControlNet not available (missing models/sdxl/controlnet_depth or "
                "annotators/dpt_hybrid-midas-501f0c75.pt); running pose+canny. See "
                "scripts/download_weights.sh ONLY=depth to enable depth."
            )
        control_images.append(_get_canny_control(cloth_ref))
        # 按品类自动调节 Canny 强度:实心面料(upper/dress/onesie)用 0.55 锁图案;
        # 薄纱/网眼类用 0.4 防鬼影。VTON_CANNY 显式覆盖优先级最高。
        _canny_defaults = {"upper": 0.55, "tunic": 0.55, "dress": 0.55, "onesie": 0.55, "lower": 0.45}
        _canny_default = _canny_defaults.get(garment_type, 0.55)
        control_scales.append(float(os.environ.get("VTON_CANNY", str(_canny_default))))

    import os as _os

    generator = torch.Generator(device=device).manual_seed(int(_os.environ.get("VTON_SEED", "42")))

    inpaint_mode = bool(_sdxl_state.get("inpaint_mode"))
    hole_img = None
    if cloth_ref.size != (target_w, target_h):
        ip_ref = cloth_ref.resize((target_w, target_h), PILImage.Resampling.LANCZOS)
    else:
        ip_ref = cloth_ref
    if inpaint_mode:
        try:
            import numpy as _np_hole

            _hole_subj = foreground_mask(human_img)
            _hole_mask = clothing_mask(
                human_img,
                garment_type=garment_type,
                garment_extent=garment_extent,
                subject_mask=_hole_subj,
            )
            if _hole_mask is not None and cloth_ref_alpha is not None:
                _hole_mask, _ = intersect_mask_with_garment_alpha(_hole_mask, cloth_ref_alpha)
            hole_img = _mask_to_pil(_hole_mask, (target_w, target_h))
            if float(_np_hole.asarray(hole_img).mean()) < 4.0:
                hole_img = None
                inpaint_mode = False
                degraded_reasons.append("inpaint hole empty; falling back to full-frame ControlNet")
                logger.warning("DEGRADED: inpaint hole empty; full-frame ControlNet")
        except Exception:
            logger.exception("failed to build inpaint hole; full-frame ControlNet")
            hole_img = None
            inpaint_mode = False
            degraded_reasons.append("inpaint hole build failed; falling back to full-frame ControlNet")
    elif not _inpaint_weights_ready():
        degraded_reasons.append("sdxl_inpaint weights missing; full-frame ControlNet redraws face/arms")

    pipe_kwargs = _build_sdxl_infer_kwargs(
        prompt=prompt,
        negative_prompt=negative_prompt,
        control_images=control_images,
        control_scales=control_scales,
        width=target_w,
        height=target_h,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(os.environ.get("VTON_GUIDANCE", "7.5")),
        generator=generator,
        ip_adapter_image=ip_ref if ip_adapter_loaded else None,
        inpaint_mode=inpaint_mode and hole_img is not None,
        init_image=human_img if inpaint_mode else None,
        mask_image=hole_img,
        strength=float(os.environ.get("VTON_INPAINT_STRENGTH", "0.99")),
    )

    # 采样(含 denoise)在 autocast 下跑;VAE 解码移出 autocast 并以 fp32 执行,
    # 否则 SDXL fp16 VAE 解码溢出产生 NaN/黑图。
    with _GPU_LOCK, torch.no_grad(), torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
        latents = pipe(**{**pipe_kwargs, "output_type": "latent"}).images
    with torch.no_grad():
        decoded = pipe.vae.decode(latents.to(torch.float32) / pipe.vae.config.scaling_factor, return_dict=False)[0]
    result = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    # 服装区遮罩:细节回贴(衣服纹理/颜色)与身份锁(人脸/背景)共用同一张
    # clothing_mask, 只算一次(含 rembg u2net 推理,是重操作)。
    # mask 255=重绘区(reapply 在此贴回原衣纹理颜色),0=保护区
    # (protect_inverse_refit 在此贴回源图像素保身份)。两者互补:
    # reapply 管衣服保真, 保护贴回管脸/发丝/手臂不走形。
    # 任一开关开启才算 mask,都关时跳过省一次推理。
    need_mask = (
        os.environ.get("VTON_DETAIL", "1") != "0"
        or os.environ.get("VTON_LOCK_ID", "1") != "0"
        or os.environ.get("VTON_PATTERN_LOCK", "1") != "0"
    )
    # subject_mask(rembg 前景)算一次,复用给 clothing_mask 和 force_white_background:
    # clothing_mask 用它做 cv2 5.0 启发式脸区兜底; force_white_background 用它
    # 保护主体(含胸口/脖子)不被压白——不传时 cv2 5.0 下胸口白块根因。
    subj_mask = foreground_mask(human_img) if need_mask or background == "white" else None
    # garment_extent 已在函数开头检测(用于 auto 推断),此处直接复用。
    shared_mask = (
        clothing_mask(
            human_img,
            garment_type=garment_type,
            garment_extent=garment_extent,
            subject_mask=subj_mask,
        )
        if need_mask
        else None
    )
    # need_mask 为真却拿不到 mask(pipeline audit 行6):detail/lock/seam/skin/inpaint
    # 全部后处理会被静默跳过 → 返回"裸 SDXL 帧"。必须 warning + 记降级, 不静默。
    if need_mask and shared_mask is None:
        _mask_reason = (
            "clothing_mask unavailable (subject mask missing or subject too small); "
            "detail/identity-lock/seam/skin/inpaint post-processing all SKIPPED — raw SDXL frame"
        )
        logger.warning("DEGRADED: %s", _mask_reason)
        degraded_reasons.append(_mask_reason)

    # 应用 mask = 几何 mask ∩ 生成图主体: clothing_mask 是几何外推(长袖
    # 走廊 85% 兜底/下摆截线), 生成器没画满的走廊里是生成图白背景;
    # 不求交就会把白底(被 Reinhard 染成衣色)半透明盖到原图上 ——
    # 09290 髋部粉块 + 黑矩形根因。reapply/blend/seam/inpaint 都用应用
    # mask; _correct_skin_tone 用应用 mask, 校正贴回区内的生成肤色
    # (胸口/手臂两个肤色), 衣料像素不拉。
    # 此处 result 还是裸 SDXL 帧(未染色), u2net 判主体最可靠。
    apply_mask = limit_mask_to_subject(shared_mask, foreground_mask(result)) if shared_mask is not None else None
    # 贴回 mask ∩ warp 衣料形: 走廊/hem 尾巴里的裤子/肚皮重画不贴回原图
    # (裤子变色/漏肚脐); 形错位(fail-open)或 warp 关闭时保持完整 mask。
    apply_mask, _alpha_kept = intersect_mask_with_garment_alpha(apply_mask, cloth_ref_alpha)
    if _alpha_kept is not None:
        logger.info("apply_mask ∩ warp-garment-alpha: %.0f%% of apply_mask kept", _alpha_kept * 100)
    if shared_mask is not None and apply_mask is not None:
        import numpy as _npa

        _cov = float(_npa.asarray(apply_mask).mean()) / max(float(_npa.asarray(shared_mask).mean()), 1.0)
        logger.info("apply_mask(gen-subject intersect): %.0f%% of shared_mask", _cov * 100)

    # mask 可视化(VTON_DEBUG_MASK=1, 输出到 VTON_DEBUG_DIR, 默认 .): clothing_mask
    # 叠加基模(绿=mask 区, 红=主体非 mask 区) + warp 参考图 + extent 中央条带边界 +
    # 肤色校正排除区红线(bbox 顶 12% = 下巴)。哪块被 mask 圈进/挖出, 直接看图;
    # 中间各步(hemcut/corridor/carve/final)看 clothing_mask 落的 maskstage_* 阶段图。
    if os.environ.get("VTON_DEBUG_MASK", "0") == "1" and shared_mask is not None:
        try:
            import cv2 as _cvd
            import numpy as _npd

            _ddir = Path(os.environ.get("VTON_DEBUG_DIR", "."))
            _ddir.mkdir(parents=True, exist_ok=True)
            _tag = os.environ.get("VTON_DEBUG_TAG", "img")
            _vis = _npd.array(human_img.convert("RGB"))
            _cm = _npd.asarray(shared_mask, dtype=_npd.float32)
            if _cm.shape[:2] != _vis.shape[:2]:
                _cm = _cvd.resize(_cm, (_vis.shape[1], _vis.shape[0]), interpolation=_cvd.INTER_LINEAR)
            _mz = _cm > 127
            _ov = _vis.copy()
            _g = _ov[_mz].astype(_npd.float32) * 0.55 + _npd.array([0, 190, 0], _npd.float32) * 0.45
            _ov[_mz] = _g.astype("uint8")
            if subj_mask is not None:
                _sm = _npd.asarray(subj_mask)
                if _sm.shape[:2] != _vis.shape[:2]:
                    _sm = _cvd.resize(_sm, (_vis.shape[1], _vis.shape[0]), interpolation=_cvd.INTER_NEAREST)
                _sz = (_sm > 0) & ~_mz
                _r = _ov[_sz].astype(_npd.float32) * 0.55 + _npd.array([220, 0, 0], _npd.float32) * 0.45
                _ov[_sz] = _r.astype("uint8")
            _nw = _vis.mean(axis=2) < 240
            if _nw.any():
                _ysn = _npd.where(_nw.any(axis=1))[0]
                _fb = int(_ysn[0] + (_ysn[-1] - _ysn[0]) * 0.12)
                _cvd.line(_ov, (0, _fb), (_vis.shape[1], _fb), (0, 0, 255), 3)
            _cvd.imwrite(
                str(_ddir / f"maskdebug_{_tag}.jpg"),
                _cvd.cvtColor(_ov, _cvd.COLOR_RGB2BGR),
                [int(_cvd.IMWRITE_JPEG_QUALITY), 90],
            )
            # 应用 mask(几何∩生成主体)叠加图: 红 = 几何 mask 有、但生成图
            # 没画出内容的区域(走廊白底, 被交集剔除, 不再贴到原图)。
            if apply_mask is not None:
                _am = _npd.asarray(apply_mask, dtype=_npd.float32)
                if _am.shape[:2] != _vis.shape[:2]:
                    _am = _cvd.resize(_am, (_vis.shape[1], _vis.shape[0]), interpolation=_cvd.INTER_LINEAR)
                _az = _am > 127
                _ov2 = _vis.copy()
                _g2 = _ov2[_az].astype(_npd.float32) * 0.55 + _npd.array([0, 190, 0], _npd.float32) * 0.45
                _ov2[_az] = _g2.astype("uint8")
                _dz = _mz & ~_az
                _r2 = _ov2[_dz].astype(_npd.float32) * 0.55 + _npd.array([220, 0, 0], _npd.float32) * 0.45
                _ov2[_dz] = _r2.astype("uint8")
                _cvd.imwrite(
                    str(_ddir / f"applymaskdebug_{_tag}.jpg"),
                    _cvd.cvtColor(_ov2, _cvd.COLOR_RGB2BGR),
                    [int(_cvd.IMWRITE_JPEG_QUALITY), 90],
                )
            _cvd.imwrite(
                str(_ddir / f"refdebug_{_tag}.jpg"),
                _cvd.cvtColor(_npd.array(cloth_ref.convert("RGB")), _cvd.COLOR_RGB2BGR),
                [int(_cvd.IMWRITE_JPEG_QUALITY), 90],
            )
            if garment_extent:
                _gd = _npd.array(garment_img.convert("RGB"))
                _gw = _gd.shape[1]
                _cvd.rectangle(_gd, (int(_gw * 0.22), 0), (int(_gw * 0.78), _gd.shape[0] - 1), (0, 128, 255), 2)
                _cvd.putText(_gd, f"extent={garment_extent}", (8, 24), _cvd.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)
                _cvd.imwrite(
                    str(_ddir / f"extentdebug_{_tag}.jpg"),
                    _cvd.cvtColor(_gd, _cvd.COLOR_RGB2BGR),
                    [int(_cvd.IMWRITE_JPEG_QUALITY), 90],
                )
            logger.info(
                "VTON_DEBUG_MASK dump -> %s (maskdebug/applymaskdebug/refdebug/extentdebug_%s + maskstage_* 阶段图)",
                _ddir,
                _tag,
            )
        except Exception:
            logger.warning("VTON_DEBUG_MASK dump failed", exc_info=True)

    # 确定性贴印升级: VTON_REFIT_ENGINE 选择 tps / optical_flow / grid(默认)。
    # 在 reapply_garment_detail 之前, 用 TPS/光流把 cloth_ref 的贴印像素对齐褶皱,
    # 让 pattern_lock 用更准的贴印结果。grid = 现状; tps = RBF TPS 网格升级;
    # optical_flow = tps 结果上叠加 Farneback 光流微移。
    # 注意: 必须在 reapply_garment_detail(detail_applied 计算)之前执行, 且 cloth_ref
    # 替换后 cloth_ref_alpha 也要同步更新, 否则 pattern_lock 会用旧 alpha。
    if shared_mask is not None and apply_mask is not None and warp_enabled:
        _refit_engine = os.environ.get("VTON_REFIT_ENGINE", "grid").lower()
        if _refit_engine in ("tps", "optical_flow"):
            try:
                from preserve_refit import optical_flow_refit, tps_warp

                _anchor_src, _anchor_dst = _collect_refit_anchors(
                    garment_img, cloth_ref, human_img, garment_extent, _pose
                )
                if _anchor_src and _anchor_dst:
                    _refit_canvas = (result.size[0], result.size[1])
                    _refit = tps_warp(cloth_ref, _anchor_src, _anchor_dst, _refit_canvas)
                    if _refit_engine == "optical_flow":
                        _refit = optical_flow_refit(_refit, result, apply_mask)
                    cloth_ref = _refit if _refit.mode == "RGBA" else _refit.convert("RGBA")
                    if cloth_ref_alpha is not None and cloth_ref_alpha.size != cloth_ref.size:
                        cloth_ref_alpha = cloth_ref_alpha.resize(cloth_ref.size, Image.Resampling.BILINEAR)
                    logger.info(
                        "refit engine=%s applied (%d anchors); pattern_lock will use refit pixels",
                        _refit_engine,
                        len(_anchor_src),
                    )
                else:
                    logger.warning("refit anchors unavailable; keeping grid remap")
            except Exception:
                logger.exception("refit engine %s failed; keeping grid remap", _refit_engine)

    # 商品细节回贴:VTON_PATTERN_LOCK=1 且已 warp 时, warp 衣直接覆盖衣服区,
    # 生成图只调褶皱光影;否则走高频柔光 + Reinhard 颜色迁移。
    # PATTERN_LOCK 与 DETAIL 解耦:关 DETAIL 仍可贴回颜色/纹理(配方 C 花纹丢失根因)。
    # WARP=0 时几何 overlay 会对不齐, 退回 Reinhard, 不当成"已锁花纹"。
    detail_applied = False
    _detail_on = os.environ.get("VTON_DETAIL", "1") != "0"
    _pattern_on = os.environ.get("VTON_PATTERN_LOCK", "1") != "0"
    if _pattern_on and not warp_enabled:
        logger.warning(
            "PATTERN_LOCK=1 with VTON_WARP=0: geometric overlay skipped "
            "(flat cloth would misalign); using Reinhard color/texture transfer"
        )
    if (_detail_on or _pattern_on) and shared_mask is not None:
        try:
            pattern_lock = _pattern_on and warp_enabled
            result = reapply_garment_detail(
                result,
                cloth_ref,
                apply_mask,
                pattern_lock=pattern_lock,
                ref_alpha=cloth_ref_alpha,
            )
            detail_applied = True
            logger.info(
                "reapply_garment_detail: pattern_lock=%s on garment region",
                pattern_lock,
            )
        except Exception:
            logger.exception("reapply_garment_detail failed; using raw SDXL output")

    # 身份锁: 遮罩外(人脸/发丝/手/背景)贴回源图 human_img, 遮罩内保留生成像素。
    # 替代旧 blend_tryon + _seam_repair 两步: protect_inverse_refit 一步完成
    # 保护贴回 + Telea 边界, 手臂不会变白、胸口不会变白、发丝不会糊。
    lock_identity = False
    if os.environ.get("VTON_LOCK_ID", "1") != "0" and apply_mask is not None:
        from preserve_refit import protect_inverse_refit

        _feather = int(os.environ.get("VTON_PROTECT_FEATHER", "4"))
        result = protect_inverse_refit(result, human_img, apply_mask, feather_px=_feather)
        lock_identity = True

    # 接缝修复: protect_inverse_refit 已内置 Telea 边界带, 无需再跑 _seam_repair。
    # 肤色校正: 保护贴回后手臂/胸口已贴回源图像素, 肤色校正作为兜底仍保留。

    # 肤色校正:SDXL 重画手臂皮肤常丢失原始肤色(发白发灰)。
    # 检测 result 中肤色像素,把色调向 human_img 原始肤色偏移。
    # 传新衣遮罩:长袖盖住原图手臂位置时,袖子像素不被"原图肤色兜底"
    # 强拉成肤色(袖子发黄根因)。
    if lock_identity:
        try:
            result = _correct_skin_tone(human_img, result, cloth_mask=apply_mask)
        except Exception:  # noqa: BLE001
            logger.warning("_correct_skin_tone failed; using raw blend result (DEGRADED)")
            degraded_reasons.append("skin tone correction failed (arms/hands may look washed out)")

    # SDXL Inpainting 二次修复(可选):低降噪强度(0.35)局部重绘接缝/肤色
    # 异常区。需要 SDXL Inpainting 模型 + VTON_INPAINT_REPAIR=1。
    # VTON_PRINT_INPAINT=1 时额外并入印花 ROI 边界带: 确定性贴印区
    # (字母/图案内部)不进 inpaint, 只有贴印区与生成衣料的边界带进
    # 局部 inpaint 做混合。
    if lock_identity:
        try:
            _print_roi = None
            if os.environ.get("VTON_PRINT_INPAINT", "0") == "1" and warp_enabled and apply_mask is not None:
                from preserve_refit import detect_print_roi

                _print_roi = detect_print_roi(cloth_ref, apply_mask)
                if _print_roi is not None:
                    logger.info(
                        "print ROI edge band %.1f%% of frame; included in inpaint repair",
                        float(_print_roi.mean()) / 255.0 * 100.0,
                    )
            result = _sdxl_inpaint_repair(result, apply_mask, original=human_img, print_roi=_print_roi)
        except Exception:  # noqa: BLE001
            logger.warning("_sdxl_inpaint_repair failed; using pre-inpaint result (DEGRADED)")
            degraded_reasons.append("SDXL inpaint repair failed")

    # 印花保真分 + 交付档位: 贴回区内 warp 参考与结果的灰度相关。
    # pattern_lock 生效 → 高分 auto; 花纹/字母被重画或错位 → 低分 review。
    # 写入 result.info, 由 worker 落库 job 元数据; 多 seed 排序共用此分。
    if apply_mask is not None and warp_enabled:
        try:
            from quality_gate import delivery_tier, print_fidelity_score

            _pf = print_fidelity_score(result, cloth_ref, apply_mask)
            result.info["print_fidelity"] = _pf
            result.info["delivery_tier"] = delivery_tier(_pf, sku_has_text=False)
            logger.info("print fidelity %.3f -> tier %s", _pf, result.info["delivery_tier"])
        except Exception:
            logger.exception("print fidelity scoring failed; tier not set")

    # 袖条生成后确定性合成: 走廊尾段(前臂)常被"∩生成主体"裁掉——SDXL 不把
    # 前臂画满 → 走廊白底被剔(c7 实测 cut_row 以下 in-apply=0), 袖条最多贴到
    # 肘。生成后把臂列带整段接管: 袖端以下贴回原图手臂/手(原肤色, 抹掉 SDXL
    # 多画的袖/发灰肤色), 袖条(与 refit 同源)盖在上面, 不再依赖 apply_mask。
    # VTON_SLEEVE_REFIT=0 可关。
    if warp_enabled and _sleeve in ("long", "short") and os.environ.get("VTON_SLEEVE_REFIT", "1") != "0":
        try:
            from engine import arm_refit_overlay

            _ov = arm_refit_overlay(
                garment_img, human_img, target_w, pose=_pose, sleeve_len=_sleeve, arm_cols=_arm_cols
            )
            if _ov is not None:
                _res = result.convert("RGBA")
                _res.alpha_composite(_ov)
                result = _res if result.mode == "RGBA" else _res.convert("RGB")
                logger.info("arm refit overlay composited (strips + base arms, post-hoc)")
        except Exception:
            logger.exception("post-hoc arm refit overlay failed; arms keep generated pixels")
            degraded_reasons.append("post-hoc arm overlay failed (sleeves/forearms may look washed out)")

    if background == "transparent":
        result = _remove_background(result)
    elif background == "white":
        # 电商主图硬性要求:背景必须纯白 RGB(255,255,255)。扩散模型生成的
        # 背景常偏暖米色/灰,不能直接采用——把边缘连通的近白背景压成纯白,
        # 只改背景区域,模特/衣服原样保留(边缘连通检查避免误伤身上的白色衣物)。
        if result.mode == "RGBA":
            bg = PILImage.new("RGB", result.size, (255, 255, 255))
            bg.paste(result, mask=result.split()[-1])
            result = bg
        # force_white 坑5(v0.1.8 docstring):新版内置"主体区 + 2px 抗锯齿边保留",
        # 边界处理交给函数内部;本地保留的是**并集掩膜**改进(对应 erode_v2 实验):
        # 保护掩膜 = 基模主体 并 生成图主体:
        #  - 透视款(如 01):结果图 u2net 漏判半透明胸口, 基模主体兜住躯干;
        #  - 宽松款(如 07 宽袖):生成图主体兜住超出基模轮廓的袖子。
        # 不再内缩(v0.1.7 时代为避开旧版白晕环做的 2px erode):新版函数自带
        # +2px 边保留,内缩会与之抵消,故直接用并集。
        if subj_mask is not None:
            _sg = foreground_mask(result)
            if _sg is not None:
                import numpy as _np

                subj_mask = _np.logical_or(subj_mask > 0, _sg > 0).astype("uint8")
        result = force_white_background(result, subject_mask=subj_mask)

    rw, rh = result.size
    scale = long_side / max(rw, rh)
    if abs(scale - 1) > 0.01:
        result = result.resize((int(rw * scale), int(rh * scale)), PILImage.Resampling.LANCZOS)

    # 降级信号合成(pipeline audit 行8):IP-Adapter + depth + LoRA + mask 各环节
    # 失败备注共同 OR 进 engine_degraded。「未安装权重」不视为降级,「装了但失败/
    # 被跳过」才算。worker.py 只透传这一对 flag, 故在此一次合成完整。
    degraded_reason = "; ".join(degraded_reasons)
    result.info["engine_degraded"] = bool(degraded_reason)
    result.info["degraded_reason"] = degraded_reason
    result.info["lock_identity"] = lock_identity
    result.info["detail_applied"] = detail_applied
    result.info["prewhiten"] = {"base": pw_base, "garment": pw_garment}
    # 出图后清空 CUDA 缓存:VAE 解码的 fp32 中间张量会留下大块显存碎片,
    # 不清会在下一任务/下一解码时放大峰值,触发 cgroup OOM。
    if device.type == "cuda":
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
    return result
