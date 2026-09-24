import logging
import os
import time
from typing import List, Dict, Any

import cv2
import numpy as np
import onnxruntime as ort
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if getattr(config, "VERBOSE_LOGGING", False) else logging.WARNING)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    
    return keep


class PlateDetector:
    """YOLOv9 license plate detector using ONNX Runtime with custom post-processing."""
    
    def __init__(self):
        self.model_path = config.MODEL_PATH
        self.device = config.DEVICE
        self.imgsz = config.YOLO_IMGSZ
        self.conf_threshold = config.CONFIDENCE_THRESHOLD
        self.iou_threshold = config.IOU_THRESHOLD
        self.license_class_id = config.LICENSE_PLATE_CLASS_ID
        
        # Load class names
        self.class_names = config.YOLO_CLASSES
        
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}")
            
            # ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if self.device == "cuda":
                providers.insert(0, 'CUDAExecutionProvider')
            
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=so,
                providers=providers
            )
            
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"YOLOv9 detector loaded from {self.model_path} (ONNX Runtime)")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv9 model: {str(e)}")
            raise
    
    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple, tuple]:
        """Preprocess frame for YOLOv9 inference (Ultralytics-style letterbox)."""
        frame_h, frame_w = frame.shape[:2]
        
        # Letterbox resize to maintain aspect ratio (centered padding)
        scale = min(self.imgsz / frame_w, self.imgsz / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Centered padding (standard Ultralytics letterbox)
        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # BGR to RGB, normalize, HWC to CHW
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0
        padded = padded.transpose(2, 0, 1)  # HWC -> CHW
        padded = np.expand_dims(padded, 0)  # Add batch dim
        
        return padded, scale, (new_w, new_h), (pad_x, pad_y)
    
    def _postprocess(self, output: np.ndarray, scale: float, orig_shape: tuple, pad: tuple) -> List[Dict[str, Any]]:
        """Post-process YOLOv9 raw output."""
        # Output shape: [1, 9, anchors] -> [9, anchors]
        # 9 channels = 4 bbox (cx, cy, w, h) + 5 class scores (sigmoid)
        out = output[0]
        
        cx, cy, w, h = out[0], out[1], out[2], out[3]
        # Class scores (indices 4-8): license, car, bike, rickshaw, truck
        cls_scores = out[4:9]  # Shape: [5, anchors]
        
        # Apply sigmoid to get class probabilities
        cls_probs = 1 / (1 + np.exp(-cls_scores))
        
        # License plate class probability (class 0 = license)
        license_prob = cls_probs[0]
        
        # Confidence = license plate class probability
        conf = license_prob
        
        # Filter by confidence threshold and license plate class
        mask = conf >= self.conf_threshold
        if not mask.any():
            return []
        
        # Get filtered detections
        cx_f = cx[mask]
        cy_f = cy[mask]
        w_f = w[mask]
        h_f = h[mask]
        conf_f = conf[mask]
        
        # Convert cx,cy,w,h to x1,y1,x2,y2 (in model input space - padded 640x640)
        x1 = cx_f - w_f / 2
        y1 = cy_f - h_f / 2
        x2 = cx_f + w_f / 2
        y2 = cy_f + h_f / 2
        
        # Remove padding offsets to get coordinates in resized image space
        pad_x, pad_y = pad
        x1 = x1 - pad_x
        y1 = y1 - pad_y
        x2 = x2 - pad_x
        y2 = y2 - pad_y
        
        # Scale back to original image coordinates
        x1 = x1 / scale
        y1 = y1 / scale
        x2 = x2 / scale
        y2 = y2 / scale
        
        # Clip to image boundaries
        frame_h, frame_w = orig_shape
        x1 = np.clip(x1, 0, frame_w)
        y1 = np.clip(y1, 0, frame_h)
        x2 = np.clip(x2, 0, frame_w)
        y2 = np.clip(y2, 0, frame_h)
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # NMS
        keep_idx = nms(boxes, conf_f, self.iou_threshold)
        
        detections = []
        for idx in keep_idx:
            detections.append({
                "bbox": [int(boxes[idx, 0]), int(boxes[idx, 1]), int(boxes[idx, 2]), int(boxes[idx, 3])],
                "conf": float(conf_f[idx]),
                "class_id": self.license_class_id
            })
        
        return detections
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect license plates in a single frame."""
        return self.detect_batch([frame])[0]
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Detect license plates in a batch of frames."""
        if not frames:
            return []
        
        try:
            batch_inputs = []
            scales = []
            orig_shapes = []
            pads = []
            
            for frame in frames:
                if frame is None or frame.size == 0:
                    batch_inputs.append(None)
                    scales.append(1.0)
                    orig_shapes.append((0, 0))
                    pads.append((0, 0))
                    continue
                
                orig_h, orig_w = frame.shape[:2]
                orig_shapes.append((orig_h, orig_w))
                
                inp, scale, _, pad = self._preprocess(frame)
                batch_inputs.append(inp)
                scales.append(scale)
                pads.append(pad)
            
            # Filter valid frames
            valid_indices = [i for i, inp in enumerate(batch_inputs) if inp is not None]
            if not valid_indices:
                return [[] for _ in frames]
            
            valid_inputs = np.concatenate([batch_inputs[i] for i in valid_indices], axis=0)
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: valid_inputs})
            output = outputs[0]  # [batch, 9, 3549]
            
            # Post-process each frame
            results = []
            for batch_idx, frame_idx in enumerate(valid_indices):
                frame_output = output[batch_idx:batch_idx+1]
                scale = scales[frame_idx]
                orig_shape = orig_shapes[frame_idx]
                pad = pads[frame_idx]
                
                detections = self._postprocess(frame_output, scale, orig_shape, pad)
                
                # Apply ROI filtering (same as before)
                detections = self._apply_roi_filtering(detections, orig_shape)
                
                results.append(detections)
            
            # Fill in empty results for invalid frames
            final_results = []
            res_idx = 0
            for i in range(len(frames)):
                if i in valid_indices:
                    final_results.append(results[res_idx])
                    res_idx += 1
                else:
                    final_results.append([])
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return [[] for _ in frames]
    
    def _apply_roi_filtering(self, detections: List[Dict], frame_shape: tuple) -> List[Dict]:
        """Apply ROI and foreground filtering."""
        if not detections:
            return []
        
        frame_h, frame_w = frame_shape
        top_ignore = getattr(config, "ROI_TOP_IGNORE_RATIO", 0.12)
        bottom_ignore = getattr(config, "ROI_BOTTOM_IGNORE_RATIO", 0.05)
        min_area = getattr(config, "MIN_PLATE_AREA", 200)
        min_w = getattr(config, "MIN_PLATE_WIDTH", 15)
        min_h = getattr(config, "MIN_PLATE_HEIGHT", 10)
        
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            plate_w = x2 - x1
            plate_h = y2 - y1
            plate_area = plate_w * plate_h
            
            if plate_w < min_w or plate_h < min_h or plate_area < min_area:
                continue
            
            box_cy = (y1 + y2) / 2.0
            if box_cy < frame_h * top_ignore or box_cy > frame_h * (1.0 - bottom_ignore):
                continue
            
            filtered.append(det)
        
        if not filtered:
            return []
        
        # Filter overlapping (keep larger)
        keep = []
        sorted_dets = sorted(filtered, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]), reverse=True)
        for det in sorted_dets:
            x1, y1, x2, y2 = det['bbox']
            box_area = (x2 - x1) * (y2 - y1)
            overlap = False
            for k_det in keep:
                kx1, ky1, kx2, ky2 = k_det['bbox']
                ix1 = max(x1, kx1)
                iy1 = max(y1, ky1)
                ix2 = min(x2, kx2)
                iy2 = min(y2, ky2)
                if ix1 < ix2 and iy1 < iy2:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area / box_area > 0.70:
                        overlap = True
                        break
            if not overlap:
                keep.append(det)
        
        # Foreground filtering
        for det in keep:
            det["fg_score"] = self._calculate_foreground_score(det["bbox"], frame_h, frame_w)
            det["area"] = (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1])
        
        keep.sort(key=lambda d: d.get("fg_score", 0.0), reverse=True)
        
        if getattr(config, "ENABLE_FOREGROUND_FILTERING", True) and keep:
            if getattr(config, "SELECT_PRIMARY_FOREGROUND_ONLY", True):
                keep = [keep[0]]
            else:
                primary_area = keep[0]["area"]
                cutoff_ratio = getattr(config, "FOREGROUND_AREA_RATIO_CUTOFF", 0.25)
                min_keep_area = primary_area * cutoff_ratio
                keep = [d for d in keep if d["area"] >= min_keep_area]
        
        return keep
    
    @staticmethod
    def _calculate_foreground_score(bbox: list, frame_h: int, frame_w: int) -> float:
        x1, y1, x2, y2 = bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if frame_h <= 0 or area <= 0:
            return 0.0
        y_center = (y1 + y2) / 2.0
        y_norm = max(0.0, min(1.0, y_center / float(frame_h)))
        y_weight = 0.25 + 0.75 * y_norm
        return float(area * y_weight)