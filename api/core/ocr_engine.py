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
        self.session = ort.InferenceSession(config.OCR_MODEL_PATH, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        logger.info("OCR model loaded: %s", config.OCR_MODEL_PATH)

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """BGR → uint8 RGB 128×64 NHWC."""
        return cv2.cvtColor(cv2.resize(bgr, (128, 64)), cv2.COLOR_BGR2RGB)[np.newaxis]

    def _decode(self, logits: np.ndarray) -> list[str]:
        return [
            "".join(_CHARSET[i] for i in np.argmax(logits[b], axis=-1)).replace("_", "")
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