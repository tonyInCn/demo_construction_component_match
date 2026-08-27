import time
import uuid
from typing import Optional

import cv2
import numpy as np

from backend.config import ANNOTATED_DIR, ROIS_DIR
from backend.logger import get_logger
from backend.pipeline.model_detector import ModelDetector

logger = get_logger(__name__)


class Processor:
    def __init__(self, model_detector: ModelDetector):
        self.model_detector = model_detector
        self._pipelines = {}

    def start_pipeline(self, video_path: str, device: str = "cuda"):
        pipeline_id = str(uuid.uuid4())
        logger.info(f"启动检测流水线: {pipeline_id}, 视频: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        pipeline = {
            "id": pipeline_id,
            "video_path": video_path,
            "cap": cap,
            "status": "running",
            "current_frame": 0,
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "detections_count": 0,
            "start_time": time.time(),
            "last_step": "初始化",
            "unique_components": {},
            "frame_results": [],
        }
        self._pipelines[pipeline_id] = pipeline

        return {
            "pipeline_id": pipeline_id,
            "status": "started",
            "total_frames": total_frames,
            "fps": fps,
        }

    def stop_pipeline(self, pipeline_id: str):
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            return {"error": "Pipeline not found"}

        pipeline["status"] = "stopped"
        if pipeline["cap"].isOpened():
            pipeline["cap"].release()
        logger.info(f"停止流水线: {pipeline_id}")
        return {"status": "stopped"}

    def get_status(self, pipeline_id: str):
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            return {"error": "Pipeline not found"}

        elapsed = time.time() - pipeline["start_time"]
        progress = (pipeline["current_frame"] / max(pipeline["total_frames"], 1)) * 100

        return {
            "status": pipeline["status"],
            "current_frame": pipeline["current_frame"],
            "total_frames": pipeline["total_frames"],
            "progress": round(progress, 1),
            "fps": pipeline["fps"],
            "detections_count": pipeline["detections_count"],
            "elapsed_time": f"{int(elapsed * 1000)}ms",
            "last_step": pipeline["last_step"],
        }

    def get_frame_results(self, pipeline_id: str):
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            return {"error": "Pipeline not found"}

        results = pipeline.get("frame_results", [])
        if results:
            latest = results[-1]
            return {
                "frame": latest["frame"],
                "detections": latest["detections"],
            }
        return {"frame": 0, "detections": []}

    def process_frame(self, pipeline_id: str):
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None or pipeline["status"] != "running":
            return None

        ret, frame = pipeline["cap"].read()
        if not ret:
            pipeline["status"] = "completed"
            return None

        pipeline["current_frame"] += 1
        frame_num = pipeline["current_frame"]

        detections = []

        pipeline["last_step"] = "吊机检测"
        work_region = self._detect_crane(frame)

        pipeline["last_step"] = "确定检测区域"
        if work_region is not None:
            pass

        pipeline["last_step"] = "多方法构件检测"
        detections = self.model_detector.detect(frame, work_region)

        tracked = []
        for det in detections:
            track_id = str(uuid.uuid4())[:8]
            det["track_id"] = track_id

            pipeline["last_step"] = "构件匹配与判定"
            match = self.model_detector.match_component(frame, det["bbox"])
            if match:
                det["class_name"] = match["component_name"]
                det["confidence"] = match["confidence"]
                det["component_id"] = match["component_id"]
                pipeline["unique_components"][match["component_id"]] = match
            else:
                det["class_name"] = det.get("label", "未知构件")

            tracked.append(det)

        pipeline["detections_count"] = len(pipeline["unique_components"])

        annotated_frame = self._annotate_frame(frame.copy(), tracked, frame_num)
        self._save_annotated_frame(annotated_frame, pipeline_id, frame_num)

        pipeline["frame_results"].append({
            "frame": frame_num,
            "detections": tracked,
        })

        if len(pipeline["frame_results"]) > 30:
            pipeline["frame_results"] = pipeline["frame_results"][-30:]

        return annotated_frame

    def _detect_crane(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            yellow_lower = np.array([18, 80, 100])
            yellow_upper = np.array([38, 255, 255])
            orange_lower = np.array([5, 100, 100])
            orange_upper = np.array([18, 255, 255])
            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([5, 255, 255])
            red_lower2 = np.array([160, 100, 100])
            red_upper2 = np.array([180, 255, 255])

            mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
            mask_orange = cv2.inRange(hsv, orange_lower, orange_upper)
            mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
            mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)

            combined_mask = cv2.bitwise_or(mask_yellow, mask_orange)
            combined_mask = cv2.bitwise_or(combined_mask, mask_red1)
            combined_mask = cv2.bitwise_or(combined_mask, mask_red2)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            h, w = frame.shape[:2]
            min_area = (w * h) * 0.01

            for contour in contours:
                if cv2.contourArea(contour) > min_area:
                    x, y, cw, ch = cv2.boundingRect(contour)
                    pad_x = int(cw * 0.3)
                    pad_y = int(ch * 0.3)
                    work_region = np.array([
                        [max(0, x - pad_x), max(0, y - pad_y)],
                        [min(w, x + cw + pad_x), max(0, y - pad_y)],
                        [min(w, x + cw + pad_x), min(h, y + ch + pad_y)],
                        [max(0, x - pad_x), min(h, y + ch + pad_y)],
                    ])
                    return work_region
        except Exception as e:
            logger.warning(f"吊机检测异常: {e}")

        return None

    def _annotate_frame(self, frame: np.ndarray, detections, frame_num: int) -> np.ndarray:
        for det in detections:
            x, y, w, h = det["bbox"]
            label = det.get("class_name", "unknown")
            conf = det.get("confidence", 0)

            color = (0, 255, 0)
            if det.get("detector") == "grounding_dino":
                color = (255, 0, 0)
            elif det.get("detector") == "traditional_cv":
                color = (0, 255, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            text = f"{label} {conf:.2f}"
            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

        cv2.putText(
            frame, f"Frame: {frame_num}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        return frame

    def _save_annotated_frame(self, frame: np.ndarray, pipeline_id: str, frame_num: int):
        try:
            ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"{pipeline_id}_frame_{frame_num:06d}.jpg"
            filepath = ANNOTATED_DIR / filename
            cv2.imwrite(str(filepath), frame)
        except Exception as e:
            logger.warning(f"保存标注帧失败: {e}")

    def cleanup(self):
        for pid, pipeline in self._pipelines.items():
            if pipeline["cap"].isOpened():
                pipeline["cap"].release()
        self._pipelines.clear()