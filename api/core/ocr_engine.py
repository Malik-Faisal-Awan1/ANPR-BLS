import logging
import cv2
import numpy as np
import onnxruntime as ort
import config

logger = logging.getLogger(__name__)

_CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"


class TextExtractor:
    """CCT OCR engine backed by fast-plate ONNX model."""

    def __init__(self, **_):
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        so = ort.SessionOptions()
        so.intra_op_num_threads = config.OCR_INTRA_OP_THREADS
        self.session = ort.InferenceSession(config.OCR_MODEL_PATH, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        logger.info("OCR model loaded: %s", config.OCR_MODEL_PATH)

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """BGR -> uint8 RGB 128x64 NHWC with letterbox (black padding, no stretch)."""
        target_w, target_h = 128, 64
        h, w = bgr.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[np.newaxis]

    def _decode(self, logits: np.ndarray) -> list[str]:
        # rstrip only: CCT pads trailing slots with '_'; mid-string '_' never occurs in valid output.
        return [
            "".join(_CHARSET[i] for i in np.argmax(logits[b], axis=-1)).rstrip("_")
            for b in range(logits.shape[0])
        ]

    def _run(self, imgs: list[np.ndarray]) -> list[str]:
        batch = np.concatenate([self._preprocess(img) for img in imgs])
        return self._decode(self.session.run(None, {self.input_name: batch})[0])

    def extract_text(self, img: np.ndarray, **_) -> str:
        return self._run([img])[0] if img is not None and img.size else ""

    def extract_texts(self, imgs: list[np.ndarray], **_) -> list[str]:
        return self._run(imgs) if imgs else []

    # aliases
    extract_plate = extract_text
    extract_plate_texts = extract_texts