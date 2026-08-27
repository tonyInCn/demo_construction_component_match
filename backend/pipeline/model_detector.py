import time
import uuid
from typing import Optional

import cv2
import numpy as np

from backend.config import (
    CLIP_MODEL_ID,
    CONFIDENCE_THRESHOLD,
    MAX_SIDE,
    MODEL_ID,
    IOU_THRESHOLD,
)
from backend.logger import get_logger

logger = get_logger(__name__)

COMPONENT_LIBRARY = [
    {"id": "comp_001", "name": "预制柱-标准柱", "shape": "rect_vertical", "color": "gray", "threshold": 0.45},
    {"id": "comp_005", "name": "预制梁-主梁", "shape": "rect_horizontal", "color": "gray", "threshold": 0.45},
    {"id": "comp_012", "name": "预制板-楼板", "shape": "flat_large", "color": "light_gray", "threshold": 0.45},
    {"id": "comp_023", "name": "预制墙板-外墙", "shape": "rect_vertical", "color": "textured", "threshold": 0.45},
    {"id": "comp_034", "name": "预制楼梯", "shape": "stepped", "color": "gray", "threshold": 0.45},
    {"id": "comp_041", "name": "预制节点-连接", "shape": "small_circle", "color": "metallic", "threshold": 0.45},
    {"id": "comp_056", "name": "预制柱-顶层", "shape": "rect_vertical", "color": "gray", "threshold": 0.45},
    {"id": "comp_067", "name": "预制墙-内墙", "shape": "rect_vertical", "color": "light", "threshold": 0.45},
]

GROUNDING_PROMPTS = [
    "tower crane",
    "precast column",
    "precast beam",
    "precast slab",
    "precast wall",
    "precast staircase",
    "building component",
    "construction machinery",
    "concrete structure",
]


class ModelDetector:
    def __init__(self):
        self.grounding_dino = None
        self.clip_model = None
        self.clip_processor = None
        self.device = "cpu"
        self.initialized = False
        self._loading = False
        self._load_progress = 0
        self._load_step = ""
        self._load_steps = []
        self._load_error = None

    @property
    def load_status(self):
        return {
            "initialized": self.initialized,
            "grounding_dino_loaded": self.grounding_dino is not None,
            "clip_loaded": self.clip_model is not None,
            "active_methods": self._active_methods(),
            "loading": self._loading,
            "load_progress": self._load_progress,
            "load_step": self._load_step,
            "load_steps": self._load_steps,
            "load_done": self.initialized,
            "load_error": self._load_error,
        }

    def _active_methods(self):
        methods = []
        if self.grounding_dino is not None:
            methods.append("grounding_dino")
        if self.clip_model is not None:
            methods.append("clip")
        return methods

    def load_models(self, device: str = "cuda"):
        if self.initialized or self._loading:
            return
        self._loading = True
        self._load_error = None
        self.device = device

        steps = [
            "初始化环境",
            "环境就绪",
            "GroundingDINO 就绪",
            "CLIP 就绪",
            "全部就绪",
        ]
        self._load_steps = steps

        try:
            self._update_progress(10, steps[0])

            import torch
            self.device = device if torch.cuda.is_available() else "cpu"
            self._update_progress(25, steps[1])

            self._update_progress(45, steps[2])
            self.grounding_dino = self._load_grounding_dino()
            self._update_progress(70, "GroundingDINO 就绪")

            self._update_progress(85, steps[3])
            self.clip_model, self.clip_processor = self._load_clip()
            self._update_progress(95, "CLIP 就绪")

            self.initialized = True
            self._update_progress(100, steps[4])
            logger.info("所有模型加载完成")
        except Exception as e:
            self._load_error = str(e)
            logger.error(f"模型加载失败: {e}")
        finally:
            self._loading = False

    def _update_progress(self, progress: int, step: str):
        self._load_progress = progress
        self._load_step = step
        logger.info(f"加载进度: {progress}% - {step}")

    def _load_grounding_dino(self):
        try:
            from transformers import pipeline
            detector = pipeline(
                "zero-shot-object-detection",
                model=MODEL_ID,
                device=self.device,
            )
            logger.info(f"GroundingDINO 加载成功: {MODEL_ID}")
            return detector
        except Exception as e:
            logger.warning(f"GroundingDINO 加载失败: {e}")
            return None

    def _load_clip(self):
        try:
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
            processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
            model = model.to(self.device)
            model.eval()
            logger.info(f"CLIP 加载成功: {CLIP_MODEL_ID}")
            return model, processor
        except Exception as e:
            logger.warning(f"CLIP 加载失败: {e}")
            return None, None

    def detect(self, frame: np.ndarray, work_region: Optional[np.ndarray] = None):
        detections = []
        h, w = frame.shape[:2]

        detections.extend(self._detect_grounding_dino(frame, w, h))
        detections.extend(self._detect_traditional_cv(frame, w, h))

        if len(detections) == 0:
            return detections

        detections = self._apply_nms(detections)
        return detections

    def _detect_grounding_dino(self, frame: np.ndarray, w: int, h: int):
        results = []
        if self.grounding_dino is None:
            return results

        try:
            import torch
            scale = 1.0
            if max(h, w) > MAX_SIDE:
                scale = MAX_SIDE / max(h, w)
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                frame_small = frame

            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))

            prompts = GROUNDING_PROMPTS
            try:
                outputs = self.grounding_dino(
                    images=pil_image,
                    candidate_labels=prompts,
                )
            except Exception:
                prompt_text = ", ".join(prompts)
                outputs = self.grounding_dino(
                    images=pil_image,
                    text=prompt_text,
                )

            boxes = outputs.get("boxes", [])
            scores = outputs.get("scores", [])
            labels = outputs.get("labels", [])

            for box, score, label in zip(boxes, scores, labels):
                if score < 0.3:
                    continue
                box = [float(b) / scale for b in box] if scale != 1.0 else [float(b) for b in box]
                x1, y1, x2, y2 = box
                results.append({
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "confidence": float(score),
                    "label": label,
                    "detector": "grounding_dino",
                })

            del outputs, pil_image, frame_small
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"GroundingDINO 推理异常: {e}")

        return results

    def _detect_traditional_cv(self, frame: np.ndarray, w: int, h: int):
        results = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_area = (w * h) * 0.005
            max_area = (w * h) * 0.8

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area or area > max_area:
                    continue

                x, y, bw, bh = cv2.boundingRect(contour)
                aspect_ratio = bw / max(bh, 1)

                if not (0.1 < aspect_ratio < 15.0):
                    continue

                confidence = 0.4 + min(area / (w * h), 0.1) * 3
                confidence = min(confidence, 0.85)

                results.append({
                    "bbox": [x, y, bw, bh],
                    "confidence": float(confidence),
                    "label": "construction_component",
                    "detector": "traditional_cv",
                })
        except Exception as e:
            logger.warning(f"传统CV检测异常: {e}")

        return results

    def _apply_nms(self, detections):
        if len(detections) == 0:
            return detections

        boxes = []
        scores = []
        for d in detections:
            x, y, w, h = d["bbox"]
            boxes.append([x, y, x + w, y + h])
            scores.append(d["confidence"])

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        try:
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), scores.tolist(),
                score_threshold=0.3,
                nms_threshold=IOU_THRESHOLD,
            )
        except Exception:
            indices = list(range(len(detections)))

        if len(indices) > 0:
            indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            return [detections[i] for i in indices]
        return detections

    def match_component(self, frame: np.ndarray, bbox):
        x, y, w, h = bbox
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0 or self.clip_model is None or self.clip_processor is None:
            return None

        try:
            import torch
            from PIL import Image as PILImage

            roi_image = PILImage.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

            texts = [c["name"] for c in COMPONENT_LIBRARY]
            inputs = self.clip_processor(
                text=texts,
                images=roi_image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image[0]
                probs = logits_per_image.softmax(dim=0).cpu().numpy()

            best_idx = np.argmax(probs)
            best_score = float(probs[best_idx])

            if best_score < CONFIDENCE_THRESHOLD:
                return None

            component = COMPONENT_LIBRARY[best_idx]
            return {
                "component_id": component["id"],
                "component_name": component["name"],
                "confidence": best_score,
            }
        except Exception as e:
            logger.warning(f"CLIP 匹配异常: {e}")
            return None