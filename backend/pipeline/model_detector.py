import time
import uuid
from typing import Optional

import cv2
import numpy as np

from backend.config import (
    CLIP_MODEL_ID,
    CONFIDENCE_THRESHOLD,
    INFERENCE_MAX_SIDE,
    MAX_SIDE,
    MODEL_ID,
    IOU_THRESHOLD,
)
from backend.logger import get_logger

logger = get_logger(__name__)

COMPONENT_LIBRARY = [
    {"id": "comp_001", "name": "预制柱-标准柱", "shape": "rect_vertical", "color": "gray",
     "clip_text": "precast concrete column, vertical pillar, tall rectangular concrete structure", "threshold": 0.45},
    {"id": "comp_005", "name": "预制梁-主梁", "shape": "rect_horizontal", "color": "gray",
     "clip_text": "precast concrete beam, horizontal girder, long horizontal structural member", "threshold": 0.45},
    {"id": "comp_012", "name": "预制板-楼板", "shape": "flat_large", "color": "light_gray",
     "clip_text": "precast concrete slab, floor slab, flat large horizontal concrete surface, wide thin panel", "threshold": 0.45},
    {"id": "comp_023", "name": "预制墙板-外墙", "shape": "rect_vertical", "color": "textured",
     "clip_text": "precast concrete wall panel, exterior wall, textured vertical concrete wall", "threshold": 0.45},
    {"id": "comp_034", "name": "预制楼梯", "shape": "stepped", "color": "gray",
     "clip_text": "precast concrete staircase, stair steps, stepped structure, zigzag stairs with horizontal treads", "threshold": 0.45},
    {"id": "comp_041", "name": "预制节点-连接", "shape": "small_circle", "color": "metallic",
     "clip_text": "precast connection joint, small metallic connector, bolted connection, steel node", "threshold": 0.45},
    {"id": "comp_056", "name": "预制柱-顶层", "shape": "rect_vertical", "color": "gray",
     "clip_text": "precast concrete top column, vertical pillar top section, upper column", "threshold": 0.45},
    {"id": "comp_067", "name": "预制墙-内墙", "shape": "rect_vertical", "color": "light",
     "clip_text": "precast concrete interior wall, internal partition wall, smooth vertical wall panel", "threshold": 0.45},
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
            processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, use_fast=True)
            model = model.to(self.device)
            model.eval()
            logger.info(f"CLIP 加载成功: {CLIP_MODEL_ID}")
            return model, processor
        except Exception as e:
            logger.warning(f"CLIP 加载失败: {e}")
            return None, None

    def detect(self, frame: np.ndarray, work_region: Optional[np.ndarray] = None,
               step_callback=None):
        detections = []
        h, w = frame.shape[:2]

        if step_callback:
            step_callback("GroundingDINO检测")
        detections.extend(self._detect_grounding_dino(frame, w, h))

        if step_callback:
            step_callback("传统CV检测")
        detections.extend(self._detect_traditional_cv(frame, w, h))

        if len(detections) == 0:
            return detections

        if step_callback:
            step_callback("NMS融合去重")
        detections = self._apply_nms(detections)
        return detections

    def detect_batch(self, frames, work_regions=None, step_callback=None):
        if work_regions is None:
            work_regions = [None] * len(frames)

        if step_callback:
            step_callback("GroundingDINO检测")

        h_list = [f.shape[0] for f in frames]
        w_list = [f.shape[1] for f in frames]

        batch_dino = self._detect_grounding_dino_batch(frames, w_list, h_list)

        all_detections = []
        for i, frame in enumerate(frames):
            detections = list(batch_dino[i]) if i < len(batch_dino) else []

            if step_callback and i == 0:
                step_callback("传统CV检测")
            detections.extend(self._detect_traditional_cv(frame, w_list[i], h_list[i]))

            if len(detections) == 0:
                all_detections.append([])
                continue

            if step_callback and i == 0:
                step_callback("NMS融合去重")
            detections = self._apply_nms(detections)
            all_detections.append(detections)

        return all_detections

    def _detect_grounding_dino(self, frame: np.ndarray, w: int, h: int):
        results = []
        if self.grounding_dino is None:
            return results

        try:
            import torch
            scale = 1.0
            if max(h, w) > INFERENCE_MAX_SIDE:
                scale = INFERENCE_MAX_SIDE / max(h, w)
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                frame_small = frame

            from PIL import Image
            pil_image = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))

            prompts = GROUNDING_PROMPTS
            try:
                outputs = self.grounding_dino(
                    image=pil_image,
                    candidate_labels=prompts,
                )
            except Exception:
                prompt_text = ", ".join(prompts)
                outputs = self.grounding_dino(
                    image=pil_image,
                    text=prompt_text,
                )

            detections = outputs if isinstance(outputs, list) else []

            for det in detections:
                score = float(det.get("score", 0))
                if score < 0.3:
                    continue
                label = det.get("label", "unknown")
                box = det.get("box", {})
                x1 = float(box.get("xmin", 0))
                y1 = float(box.get("ymin", 0))
                x2 = float(box.get("xmax", 0))
                y2 = float(box.get("ymax", 0))
                if scale != 1.0:
                    x1 /= scale
                    y1 /= scale
                    x2 /= scale
                    y2 /= scale
                results.append({
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "confidence": score,
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

    def _detect_grounding_dino_batch(self, frames, w_list, h_list):
        if self.grounding_dino is None:
            return [[] for _ in frames]

        try:
            import torch
            from PIL import Image

            pil_images = []
            scales = []
            for frame, w, h in zip(frames, w_list, h_list):
                scale = 1.0
                if max(h, w) > INFERENCE_MAX_SIDE:
                    scale = INFERENCE_MAX_SIDE / max(h, w)
                    frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    frame_small = frame
                scales.append(scale)
                pil_images.append(Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)))

            prompts = GROUNDING_PROMPTS
            try:
                outputs = self.grounding_dino(
                    images=pil_images,
                    candidate_labels=prompts,
                )
            except Exception:
                try:
                    prompt_text = ", ".join(prompts)
                    outputs = self.grounding_dino(
                        images=pil_images,
                        text=prompt_text,
                    )
                except Exception:
                    outputs = []

            batch_results = []
            for frame_idx in range(len(frames)):
                results = []
                if frame_idx < len(outputs) and isinstance(outputs[frame_idx], list):
                    detections = outputs[frame_idx]
                elif isinstance(outputs, list) and len(outputs) == len(frames):
                    det_item = outputs[frame_idx]
                    detections = det_item if isinstance(det_item, list) else [det_item]
                else:
                    detections = []

                for det in detections:
                    score = float(det.get("score", 0))
                    if score < 0.3:
                        continue
                    label = det.get("label", "unknown")
                    box = det.get("box", {})
                    x1 = float(box.get("xmin", 0))
                    y1 = float(box.get("ymin", 0))
                    x2 = float(box.get("xmax", 0))
                    y2 = float(box.get("ymax", 0))
                    scale = scales[frame_idx]
                    if scale != 1.0:
                        x1 /= scale
                        y1 /= scale
                        x2 /= scale
                        y2 /= scale
                    results.append({
                        "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                        "confidence": score,
                        "label": label,
                        "detector": "grounding_dino",
                    })
                batch_results.append(results)

            del outputs, pil_images
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            return batch_results
        except Exception as e:
            logger.warning(f"GroundingDINO 批量推理异常: {e}")
            return [self._detect_grounding_dino(f, w_list[i], h_list[i]) for i, f in enumerate(frames)]

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

            texts = [c.get("clip_text", c["name"]) for c in COMPONENT_LIBRARY]
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
                clip_probs = logits_per_image.softmax(dim=0).cpu().numpy()

            geometry_features = self._analyze_roi_geometry(roi)

            fused_scores = self._fuse_scores(clip_probs, geometry_features)

            best_idx = np.argmax(fused_scores)
            best_score = float(fused_scores[best_idx])

            if best_score < CONFIDENCE_THRESHOLD:
                return None

            component = COMPONENT_LIBRARY[best_idx]
            return {
                "component_id": component["id"],
                "component_name": component["name"],
                "confidence": best_score,
                "clip_score": float(clip_probs[best_idx]),
                "geometry_score": float(fused_scores[best_idx] - clip_probs[best_idx]),
            }
        except Exception as e:
            logger.warning(f"CLIP 匹配异常: {e}")
            return None

    def _analyze_roi_geometry(self, roi: np.ndarray) -> dict:
        h, w = roi.shape[:2]
        features = {
            "aspect_ratio": w / max(h, 1),
            "area_ratio": 1.0,
            "step_score": 0.0,
            "flat_score": 0.0,
            "vertical_score": 0.0,
            "horizontal_score": 0.0,
            "edge_density": 0.0,
            "contour_count": 0,
            "has_stepped_pattern": False,
            "shape_vertical": 0.0,
            "shape_horizontal": 0.0,
            "shape_flat_large": 0.0,
            "shape_stepped": 0.0,
            "shape_small_circle": 0.0,
            "shape_rect_vertical": 0.0,
        }

        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / max(w * h, 1)
            features["edge_density"] = float(edge_density)

            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features["contour_count"] = len(contours)

            approx_count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < (w * h) * 0.01:
                    continue
                approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
                if len(approx) <= 6:
                    approx_count += 1

            features["aspect_ratio"] = w / max(h, 1)

            if h > 10 and w > 10:
                row_means = np.mean(gray, axis=1)
                row_diffs = np.abs(np.diff(row_means))
                threshold = np.mean(row_diffs) + 2 * np.std(row_diffs)
                significant_changes = np.sum(row_diffs > threshold)
                features["step_score"] = float(min(significant_changes / max(h * 0.1, 1), 1.0))

                step_rows = 0
                for i in range(1, len(row_diffs)):
                    if row_diffs[i] > threshold and abs(i - np.argmax(row_diffs)) > 3:
                        step_rows += 1
                features["has_stepped_pattern"] = step_rows >= 2
                features["shape_stepped"] = float(min(step_rows / max(h * 0.05, 1), 1.0))

            if features["aspect_ratio"] > 2.0:
                features["flat_score"] = float(min((features["aspect_ratio"] - 2.0) / 3.0, 1.0))
                features["shape_flat_large"] = features["flat_score"]

            if features["aspect_ratio"] > 3.0 and edge_density < 0.15:
                features["shape_flat_large"] = float(min(features["shape_flat_large"] + 0.3, 1.0))

            if features["aspect_ratio"] < 0.6:
                features["vertical_score"] = float(min((0.6 - features["aspect_ratio"]) / 0.4, 1.0))
                features["shape_vertical"] = features["vertical_score"]
                features["shape_rect_vertical"] = features["vertical_score"]

            if features["aspect_ratio"] > 1.5 and features["aspect_ratio"] <= 3.0:
                features["horizontal_score"] = float(min((features["aspect_ratio"] - 1.5) / 1.5, 1.0))
                features["shape_rect_vertical"] = features["horizontal_score"] * 0.5

            if w * h < 8000:
                features["shape_small_circle"] = float(min(1.0 - (w * h) / 8000, 1.0))

            if features["has_stepped_pattern"]:
                features["shape_stepped"] = float(min(features["shape_stepped"] + 0.4, 1.0))
                features["shape_flat_large"] *= 0.3

        except Exception as e:
            logger.warning(f"ROI几何分析异常: {e}")

        return features

    def _fuse_scores(self, clip_probs: np.ndarray, geometry_features: dict) -> np.ndarray:
        fused = clip_probs.copy()
        geometry_weight = 0.35

        shape_scores = {
            "rect_vertical": geometry_features.get("shape_rect_vertical", 0.0),
            "rect_horizontal": geometry_features.get("horizontal_score", 0.0),
            "flat_large": geometry_features.get("shape_flat_large", 0.0),
            "stepped": geometry_features.get("shape_stepped", 0.0),
            "small_circle": geometry_features.get("shape_small_circle", 0.0),
        }

        for idx, component in enumerate(COMPONENT_LIBRARY):
            shape = component.get("shape", "")
            geo_score = shape_scores.get(shape, 0.0)
            fused[idx] = (1 - geometry_weight) * clip_probs[idx] + geometry_weight * geo_score

        fused = fused / fused.sum()
        return fused