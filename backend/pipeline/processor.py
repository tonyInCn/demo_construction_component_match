import threading
import time
import uuid
from typing import Optional

import cv2
import numpy as np

from backend.config import ANNOTATED_DIR, BATCH_SIZE, FRAME_SAMPLE_RATE, ROIS_DIR
from backend.logger import get_logger
from backend.pipeline.model_detector import ModelDetector

logger = get_logger(__name__)


class Processor:
    def __init__(self, model_detector: ModelDetector):
        self.model_detector = model_detector
        self._pipelines = {}
        self._threads = {}

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

        t = threading.Thread(
            target=self._run_processing_loop,
            args=(pipeline_id,),
            daemon=True,
            name=f"pipeline-{pipeline_id[:8]}",
        )
        t.start()
        self._threads[pipeline_id] = t

        return {
            "pipeline_id": pipeline_id,
            "status": "started",
            "total_frames": total_frames,
            "fps": fps,
        }

    def _run_processing_loop(self, pipeline_id: str):
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            logger.error(f"流水线 {pipeline_id} 不存在")
            return

        logger.info(f"流水线 {pipeline_id} 后台线程启动, 视频: {pipeline['video_path']}, 总帧数: {pipeline['total_frames']}")

        batch_frames = []
        batch_nums = []

        def _flush_batch():
            nonlocal batch_frames, batch_nums
            if not batch_frames:
                return
            self._process_batch(pipeline_id, batch_frames, batch_nums)
            batch_frames = []
            batch_nums = []

        while pipeline["status"] == "running":
            ret, frame = pipeline["cap"].read()
            if not ret:
                break

            pipeline["current_frame"] += 1
            frame_num = pipeline["current_frame"]

            if frame_num % FRAME_SAMPLE_RATE != 0 and frame_num != 1:
                continue

            batch_frames.append(frame)
            batch_nums.append(frame_num)

            if len(batch_frames) >= BATCH_SIZE:
                _flush_batch()

        if batch_frames:
            _flush_batch()

        if pipeline["status"] == "running":
            pipeline["status"] = "completed"
            logger.info(f"流水线 {pipeline_id} 处理完成, 共 {pipeline['current_frame']} 帧")

    def _process_batch(self, pipeline_id: str, frames, frame_nums):
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None or pipeline["status"] != "running":
            return

        logger.info(f"批量处理 {len(frames)} 帧 (帧号: {frame_nums[0]}-{frame_nums[-1]})...")

        pipeline["last_step"] = "吊机检测"
        work_regions = []
        for frame in frames:
            wr = self._detect_crane(frame)
            work_regions.append(wr)

        pipeline["last_step"] = "确定检测区域"

        pipeline["last_step"] = "多方法构件检测"

        def _step_cb(step_name):
            pipeline["last_step"] = step_name

        all_detections = self.model_detector.detect_batch(frames, work_regions, step_callback=_step_cb)
        pipeline["last_step"] = "多方法构件检测完成"

        for i, frame in enumerate(frames):
            detections = all_detections[i] if i < len(all_detections) else []
            frame_num = frame_nums[i]

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
            image_filename = f"{pipeline_id}_frame_{frame_num:06d}.jpg"

            pipeline["frame_results"].append({
                "frame": frame_num,
                "image_filename": image_filename,
                "detections": tracked,
            })

            if len(pipeline["frame_results"]) > 30:
                pipeline["frame_results"] = pipeline["frame_results"][-30:]

        pipeline["last_step"] = "构件匹配与判定完成"
        logger.info(f"批量处理完成, 共处理 {len(frames)} 帧")

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
                "image_filename": latest.get("image_filename"),
                "detections": latest["detections"],
            }
        return {"frame": 0, "image_filename": None, "detections": []}

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