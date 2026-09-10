import threading
import time
import uuid
from typing import Optional

import cv2
import numpy as np

from backend.config import ANNOTATED_DIR, BATCH_SIZE, CANDIDATE_WINDOW_FRAMES, FRAME_SAMPLE_RATE, ROIS_DIR
from backend.logger import get_logger
from backend.pipeline.model_detector import ModelDetector

logger = get_logger(__name__)


class Processor:
    def __init__(self, model_detector: ModelDetector):
        self.model_detector = model_detector
        self._pipelines = {}
        self._threads = {}

    @staticmethod
    def _iou(box_a, box_b):
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: Processor._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [Processor._sanitize(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, bool):
            return obj
        elif obj is None:
            return None
        else:
            try:
                return float(obj)
            except (TypeError, ValueError):
                return str(obj)

    def start_pipeline(self, video_path: str, device: str = "cuda"):
        pipeline_id = str(uuid.uuid4())
        logger.info(f"启动检测流水线: {pipeline_id}, 视频: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
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
            "unique_component_details": {},
            "pending_components": {},
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

        self._flush_all_pending(pipeline)

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

        pending = pipeline["pending_components"]
        commit_window = CANDIDATE_WINDOW_FRAMES * FRAME_SAMPLE_RATE

        for i, frame in enumerate(frames):
            detections = all_detections[i] if i < len(all_detections) else []
            frame_num = frame_nums[i]
            work_region = work_regions[i]

            frame_base = f"{pipeline_id}_frame_{frame_num:06d}"

            crane_img = self._save_crane_visualization(frame.copy(), work_region, frame_num, pipeline_id)
            roi_img = self._save_roi_visualization(frame.copy(), work_region, frame_num, pipeline_id)

            matched_candidates = []
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

                    logger.info(
                        f"构件匹配: {match['component_name']} | "
                        f"置信度={match['confidence']:.3f} | "
                        f"CLIP分={match.get('clip_score', 0):.3f} | "
                        f"几何分={match.get('geometry_score', 0):.3f}"
                    )

                    comp_roi_thumb = self._save_component_roi_crop(
                        frame, det["bbox"], match["component_id"], pipeline_id, frame_num
                    )

                    candidate = {
                        "track_id": track_id,
                        "component_id": match["component_id"],
                        "class_name": match["component_name"],
                        "confidence": match["confidence"],
                        "clip_score": match.get("clip_score"),
                        "geometry_score": match.get("geometry_score"),
                        "detector": det.get("detector", ""),
                        "image_filename": f"{frame_base}.jpg",
                        "crane_image_filename": crane_img,
                        "roi_image_filename": roi_img,
                        "component_roi_filename": comp_roi_thumb,
                        "frame": frame_num,
                        "bbox": det["bbox"],
                    }
                    matched_candidates.append(candidate)
                else:
                    det["class_name"] = det.get("label", "未知构件")

                tracked.append(det)

            deduped = []
            used = [False] * len(matched_candidates)
            for ci, cand in enumerate(matched_candidates):
                if used[ci]:
                    continue
                best_idx = ci
                for cj in range(ci + 1, len(matched_candidates)):
                    if used[cj]:
                        continue
                    if self._iou(matched_candidates[best_idx]["bbox"], matched_candidates[cj]["bbox"]) >= 0.4:
                        used[cj] = True
                        if matched_candidates[cj]["confidence"] > matched_candidates[best_idx]["confidence"]:
                            used[best_idx] = True
                            best_idx = cj
                deduped.append(matched_candidates[best_idx])

            committed_details = pipeline.get("unique_component_details", {})
            for cand in deduped:
                bb = cand["bbox"]

                skip = False
                for _, committed in committed_details.items():
                    if self._iou(bb, committed.get("bbox", [0, 0, 0, 0])) >= 0.4:
                        skip = True
                        break
                if skip:
                    logger.info(f"跳过已提交构件 (IoU>=0.4): {cand['class_name']} 帧={frame_num}")
                    continue

                matched_pending_id = None
                for pid, p in pending.items():
                    if self._iou(bb, p["best_candidate"]["bbox"]) >= 0.4:
                        matched_pending_id = pid
                        break

                if matched_pending_id:
                    p = pending[matched_pending_id]
                    p["candidates"].append(cand)
                    if cand["confidence"] > p["best_candidate"]["confidence"]:
                        p["best_candidate"] = cand
                        p["component_name"] = cand["class_name"]
                        logger.info(
                            f"候选更新最佳: {cand['class_name']} | "
                            f"帧={frame_num} | 置信度={cand['confidence']:.3f}"
                        )
                else:
                    new_id = str(uuid.uuid4())[:8]
                    pending[new_id] = {
                        "component_id": cand["component_id"],
                        "component_name": cand["class_name"],
                        "first_frame": frame_num,
                        "best_candidate": cand,
                        "candidates": [cand],
                    }
                    logger.info(
                        f"构件候选已建立: {cand['class_name']} | "
                        f"起始帧={frame_num} | 初始置信度={cand['confidence']:.3f}"
                    )

            self._commit_ready_candidates(pipeline, frame_num, commit_window)

            total_components = len(pipeline["unique_components"]) + len(pipeline["pending_components"])
            pipeline["detections_count"] = total_components

            annotated_frame = self._annotate_frame(frame.copy(), tracked, frame_num)
            self._save_annotated_frame(annotated_frame, pipeline_id, frame_num)

            pipeline["frame_results"].append({
                "frame": frame_num,
                "image_filename": f"{frame_base}.jpg",
                "detections": tracked,
            })

            if len(pipeline["frame_results"]) > 30:
                pipeline["frame_results"] = pipeline["frame_results"][-30:]

        pipeline["last_step"] = "构件匹配与判定完成"
        logger.info(f"批量处理完成, 共处理 {len(frames)} 帧")

    def _commit_ready_candidates(self, pipeline, current_frame, commit_window):
        pending = pipeline["pending_components"]
        committed_ids = []

        for comp_id, p in pending.items():
            if current_frame >= p["first_frame"] + commit_window:
                best = p["best_candidate"]
                pipeline["unique_component_details"][comp_id] = best
                pipeline["unique_components"][comp_id] = {
                    "component_id": comp_id,
                    "component_name": best["class_name"],
                    "confidence": best["confidence"],
                }
                committed_ids.append(comp_id)
                logger.info(
                    f"构件提交最佳结果: {best['class_name']} | "
                    f"候选数={len(p['candidates'])} | "
                    f"最佳帧={best['frame']} | "
                    f"置信度={best['confidence']:.3f}"
                )

        for cid in committed_ids:
            del pending[cid]

    def _flush_all_pending(self, pipeline):
        pending = pipeline.get("pending_components", {})
        if not pending:
            return

        for comp_id, p in pending.items():
            best = p["best_candidate"]
            pipeline["unique_component_details"][comp_id] = best
            pipeline["unique_components"][comp_id] = {
                "component_id": comp_id,
                "component_name": best["class_name"],
                "confidence": best["confidence"],
            }
            logger.info(
                f"[flush] 构件提交: {best['class_name']} | "
                f"候选数={len(p['candidates'])} | "
                f"最佳帧={best['frame']} | "
                f"置信度={best['confidence']:.3f}"
            )

        pending.clear()

    def stop_pipeline(self, pipeline_id: str):
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            return {"error": "Pipeline not found"}

        pipeline["status"] = "stopped"
        if pipeline["cap"].isOpened():
            pipeline["cap"].release()
        self._flush_all_pending(pipeline)
        logger.info(f"停止流水线: {pipeline_id}")
        return {"status": "stopped"}

    def get_status(self, pipeline_id: str):
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            return {"error": "Pipeline not found"}

        elapsed = time.time() - pipeline["start_time"]
        progress = (pipeline["current_frame"] / max(pipeline["total_frames"], 1)) * 100

        return self._sanitize({
            "status": pipeline["status"],
            "current_frame": int(pipeline["current_frame"]),
            "total_frames": int(pipeline["total_frames"]),
            "progress": round(float(progress), 1),
            "fps": float(pipeline["fps"]),
            "detections_count": int(pipeline["detections_count"]),
            "elapsed_time": f"{int(elapsed * 1000)}ms",
            "last_step": pipeline["last_step"],
        })

    def get_frame_results(self, pipeline_id: str):
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            return {"error": "Pipeline not found"}

        results = pipeline.get("frame_results", [])
        latest_frame = results[-1]["frame"] if results else 0

        components = list(pipeline.get("unique_component_details", {}).values())

        pending = pipeline.get("pending_components", {})
        for p in pending.values():
            best = p["best_candidate"]
            comp_copy = dict(best)
            comp_copy["pending"] = True
            comp_copy["total_candidates"] = len(p["candidates"])
            components.append(comp_copy)

        components.sort(key=lambda c: c.get("frame", 0))

        return self._sanitize({
            "frame": latest_frame,
            "image_filename": results[-1].get("image_filename") if results else None,
            "detections": components,
        })

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

    def _save_crane_visualization(self, frame: np.ndarray, work_region, frame_num: int, pipeline_id: str) -> Optional[str]:
        try:
            H, W = frame.shape[:2]

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

            overlay = frame.copy()

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = (W * H) * 0.005

            crane_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    crane_contours.append(contour)
                    x, y, cw, ch = cv2.boundingRect(contour)
                    cx, cy = x + cw // 2, y + ch // 2
                    radius = max(cw, ch) // 2 + 10

                    cv2.circle(overlay, (cx, cy), radius, (0, 0, 255), 3)
                    cv2.circle(overlay, (cx, cy), radius, (255, 255, 255), 1)

                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        cv2.ellipse(overlay, ellipse, (0, 255, 255), 2)

                    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)

                    label_x = x
                    label_y = max(20, y - 12)
                    label = "CRANE"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(overlay, (label_x - 4, label_y - th - 6), (label_x + tw + 4, label_y + 2), (0, 0, 255), -1)
                    cv2.putText(overlay, label, (label_x, label_y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            if not crane_contours:
                cv2.putText(overlay, "No Crane Detected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if work_region is not None:
                pts = work_region.astype(np.int32)
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 3)
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                cv2.putText(overlay, "Work Region", (cx - 40, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(overlay, f"Frame: {frame_num} | Crane Detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"{pipeline_id}_crane_{frame_num:06d}.jpg"
            filepath = ANNOTATED_DIR / filename
            cv2.imwrite(str(filepath), overlay)
            return filename
        except Exception as e:
            logger.warning(f"保存吊机可视化失败: {e}")
            return None

    def _save_roi_visualization(self, frame: np.ndarray, work_region, frame_num: int, pipeline_id: str) -> Optional[str]:
        try:
            H, W = frame.shape[:2]

            overlay = frame.copy()

            if work_region is not None:
                x_min = max(0, int(work_region[0][0]))
                y_min = max(0, int(work_region[0][1]))
                x_max = min(W, int(work_region[2][0]))
                y_max = min(H, int(work_region[2][1]))

                roi_layer = np.zeros_like(frame)
                roi_layer[y_min:y_max, x_min:x_max] = [255, 200, 0]

                overlay = cv2.addWeighted(overlay, 0.55, roi_layer, 0.45, 0)

                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 200, 0), 3)

                pts = work_region.astype(np.int32)
                cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)

                roi_w = x_max - x_min
                roi_h = y_max - y_min
                label = f"ROI: {roi_w}x{roi_h}"
                cv2.putText(overlay, label, (x_min + 5, y_min - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
            else:
                cv2.putText(overlay, "No Crane Detected | Full Frame Scan", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(overlay, f"Frame: {frame_num} | Detection ROI", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"{pipeline_id}_roi_{frame_num:06d}.jpg"
            filepath = ANNOTATED_DIR / filename
            cv2.imwrite(str(filepath), overlay)
            return filename
        except Exception as e:
            logger.warning(f"保存ROI可视化失败: {e}")
            return None

    def _save_component_roi_crop(self, frame: np.ndarray, bbox, component_id: str, pipeline_id: str, frame_num: int) -> Optional[str]:
        try:
            x, y, w, h = bbox
            H, W = frame.shape[:2]

            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(W, x + w + pad_x)
            y2 = min(H, y + h + pad_y)

            roi = frame[y1:y2, x1:x2].copy()
            if roi.size == 0:
                return None

            roi_h, roi_w = roi.shape[:2]
            max_dim = 400
            scale = min(1.0, max_dim / max(roi_w, roi_h))
            if scale < 1.0:
                roi = cv2.resize(roi, (int(roi_w * scale), int(roi_h * scale)))

            cv2.rectangle(roi, (0, 0), (roi.shape[1] - 1, roi.shape[0] - 1), (0, 255, 255), 2)

            short_id = component_id.split("_")[-1] if "_" in component_id else component_id[:6]
            cv2.putText(roi, f"#{short_id}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            ROIS_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"{pipeline_id}_comp_{frame_num:06d}_{short_id}.jpg"
            filepath = ROIS_DIR / filename
            cv2.imwrite(str(filepath), roi)
            return filename
        except Exception as e:
            logger.warning(f"保存构件ROI裁剪失败: {e}")
            return None

    def cleanup(self):
        for pid, pipeline in self._pipelines.items():
            if pipeline["cap"].isOpened():
                pipeline["cap"].release()
        self._pipelines.clear()