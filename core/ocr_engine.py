import logging
import cv2
import numpy as np
import onnxruntime as ort
import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if getattr(config, "VERBOSE_LOGGING", False) else logging.WARNING)

_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
_MODEL_PATH = "models/fast-plate/best.onnx"


class TextExtractor:
    """OCR engine backed directly by the fast-plate ONNX model (CCT)."""

    def __init__(self, lang="en"):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        self.session = ort.InferenceSession(_MODEL_PATH, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        logger.info("fast-plate ONNX engine loaded: %s", _MODEL_PATH)

    # ── internal ──────────────────────────────────────────────────────────────

    def _preprocess(self, bgr_img: np.ndarray) -> np.ndarray:
        """BGR → uint8 RGB, resized to 128×64 (W×H), NHWC."""
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (128, 64))          # (H=64, W=128, C=3)
        return rgb[np.newaxis]                     # (1, 64, 128, 3) uint8

    def _decode(self, logits: np.ndarray) -> str:
        """logits shape: (B, 10, 37) → list[str]."""
        indices = np.argmax(logits, axis=-1)       # (B, 10)
        return [
            "".join(_CHARSET[i] for i in row).replace("_", "")
            for row in indices
        ]

    def _run(self, plate_imgs: list[np.ndarray]) -> list[str]:
        batch = np.concatenate([self._preprocess(img) for img in plate_imgs]) # (B,64,128,3)
        logits = self.session.run(None, {self.input_name: batch})[0]           # (B,10,37)
        return self._decode(logits)

    # ── public API ────────────────────────────────────────────────────────────

    def extract_text(self, plate_img: np.ndarray, **_) -> str:
        if plate_img is None or plate_img.size == 0:
            return ""
        text = self._run([plate_img])[0]
        if text:
            logger.info("OCR result: %s", text)
        return text

    def extract_texts(self, plate_imgs: list[np.ndarray], **_) -> list[str]:
        if not plate_imgs:
            return []
        return self._run(plate_imgs)

    # aliases kept for call-site compatibility
    extract_plate = extract_text
    extract_plate_texts = extract_texts