"""Detect, track and render poses; export raw/smoothed keypoints and 2D features."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
from src.posturaai.biomechanics import VIEWS, DIRECTIONS, SIDES, CAMERA_MOTIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--torch-threads', type=int, default=4, help='Limit CPU threads used around GPU inference')
    parser.add_argument('--no-video', action='store_true', help='Export poses/features without rendering an MP4')
    parser.add_argument('--progress-file', type=Path, help='Atomic progress JSON for the web app')
    parser.add_argument('--preview-file', type=Path, help='Latest annotated JPEG for the web app')
    parser.add_argument('--det-threshold', type=float, default=0.5)
    parser.add_argument('--point-threshold', type=float, default=0.3)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--view', choices=VIEWS, default='unknown', help='Optional; all views are accepted')
    parser.add_argument('--running-direction', choices=DIRECTIONS, default='unknown')
    parser.add_argument('--visible-side', choices=SIDES, default='unknown')
    parser.add_argument('--camera-motion', choices=CAMERA_MOTIONS, default='unknown')
    parser.add_argument('--athlete-id', help='Grouping identity, when known; never inferred from track_id')
    parser.add_argument('--source-id', help='Original recording identity for grouping related clips')
    parser.add_argument('--ema-alpha', type=float, default=0.65, help='Current-frame weight; 1 disables smoothing')
    parser.add_argument('--track-iou', type=float, default=0.3)
    parser.add_argument('--person-nms', type=float, default=0.3, help='Suppress overlapping duplicate person boxes; 1 disables')
    parser.add_argument('--max-track-gap', type=int, default=5)
    parser.add_argument('--cut-threshold', type=float, default=0.35, help='Thumbnail difference; 1 disables cut reset')
    args = parser.parse_args()
    if not args.input.is_file() or not args.checkpoint.is_file():
        parser.error('Input video and checkpoint must exist')
    if args.input.resolve() == args.output.resolve() or args.output.suffix.lower() != '.mp4':
        parser.error('Output must be a different .mp4 file')
    if not all(0 <= v <= 1 for v in (args.det_threshold, args.cut_threshold)):
        parser.error('Detection and cut thresholds must be between 0 and 1')
    if not math.isfinite(args.point_threshold) or args.point_threshold < 0:
        parser.error('Point score threshold must be finite and nonnegative')
    if not 0 < args.ema_alpha <= 1 or not 0 < args.track_iou <= 1 or not 0 < args.person_nms <= 1 or args.max_track_gap < 0:
        parser.error('Invalid EMA or tracker parameters')
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error('--max-frames must be positive')
    if args.torch_threads <= 0:
        parser.error('--torch-threads must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(args.torch_threads)
    cv2.setNumThreads(1)
    from mmengine.config import Config
    from mmpose.apis import inference_topdown, init_model
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
    from src.posturaai.keypoints import KEYPOINT_NAMES, SKELETON_EDGES
    from src.posturaai.training import checkpoint_load_context, sha256_file
    from src.posturaai.temporal import PoseTracker, scene_difference, serialise_points, person_nms
    from src.posturaai.video_features import export_features

    root = Path(__file__).resolve().parents[1]
    cfg_path = args.checkpoint.resolve().parent / 'effective_config.py'
    if not cfg_path.is_file():
        parser.error('Checkpoint requires adjacent effective_config.py')
    cfg = Config.fromfile(str(cfg_path))
    cfg.model.backbone.init_cfg = None
    cfg.test_dataloader = deepcopy(cfg.val_dataloader)
    cfg.test_dataloader.dataset.metainfo = dict(from_file=str(root / 'configs/datasets/srkd.py'))
    with checkpoint_load_context():
        pose_model = init_model(cfg, str(args.checkpoint), device=args.device)
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    detector = ssdlite320_mobilenet_v3_large(weights=weights).to(args.device).eval()
    transform = weights.transforms()
    tracker = PoseTracker(iou_threshold=args.track_iou, max_gap=args.max_track_gap, alpha=args.ema_alpha)
    capture = cv2.VideoCapture(str(args.input))
    writer = None
    count = detected_frames = cuts = people_count = valid_points = 0
    confidence_sum = inference_seconds = 0.0
    previous_thumb = None
    track_ids, partial_boxes = set(), 0
    metadata_path = args.checkpoint.resolve().parent / 'run_metadata.json'
    provenance = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    poses_path = args.output.with_suffix('.poses.jsonl')
    input_hash = sha256_file(args.input)
    video_id = input_hash[:16]
    header = dict(type='metadata', schema='posturaai.poses.v1', video_id=video_id,
                  input=str(args.input.resolve()), input_sha256=input_hash,
                  checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256_file(args.checkpoint),
                  config_sha256=sha256_file(cfg_path), detector='torchvision/ssdlite320_mobilenet_v3_large',
                  keypoint_names=list(KEYPOINT_NAMES), coordinates='pixels; x right; y down',
                  point_score_note='Raw SimCC model score, not a calibrated probability; may exceed 1',
                  torch_threads=args.torch_threads, rendered_video=not args.no_video,
                  view=args.view, running_direction=args.running_direction, visible_side=args.visible_side,
                  camera_motion=args.camera_motion, athlete_id=args.athlete_id,
                  source_id=args.source_id or video_id, posture_label=None,
                  det_threshold=args.det_threshold, point_threshold=args.point_threshold,
                  person_nms=args.person_nms,
                  tracking=dict(method='Hungarian IoU', iou_threshold=args.track_iou, max_gap=args.max_track_gap),
                  smoothing=dict(method='EMA', alpha=args.ema_alpha, reset_on_gap=True, fills_missing=False),
                  cut_detection=dict(method='RGB thumbnail mean absolute difference', threshold=args.cut_threshold),
                  experimental=provenance.get('experimental', True), audit_status=provenance.get('audit_status', 'unknown'))
    try:
        if not capture.isOpened():
            raise RuntimeError('Cannot open input video')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError('Invalid video frame rate')
        header.update(fps=fps, width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), timing='frame / nominal fps; constant-rate input expected')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        with poses_path.open('w') as poses_file, torch.inference_mode():
            poses_file.write(json.dumps(header, allow_nan=False)+'\n')
            while args.max_frames is None or count < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                height, width = frame.shape[:2]
                if writer is None and not args.no_video:
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot create output video')
                thumb = cv2.resize(frame, (64, 36))
                scene_cut = previous_thumb is not None and scene_difference(previous_thumb, thumb) > args.cut_threshold
                previous_thumb = thumb
                if scene_cut:
                    cuts += 1
                    tracker.reset()
                frame_start = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(args.device)
                prediction = detector([transform(tensor)])[0]
                keep = (prediction['labels'] == 1) & (prediction['scores'] >= args.det_threshold)
                boxes = prediction['boxes'][keep].cpu().numpy()
                det_scores = prediction['scores'][keep].cpu().numpy()
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
                keep_box = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes, det_scores = boxes[keep_box], det_scores[keep_box]
                keep_box = person_nms(boxes, det_scores, args.person_nms)
                boxes, det_scores = boxes[keep_box], det_scores[keep_box]
                identifiers = tracker.update(boxes, count)
                people = []
                if len(boxes):
                    detected_frames += 1
                    poses = inference_topdown(pose_model, frame, bboxes=boxes, bbox_format='xyxy')
                    if len(poses) != len(boxes):
                        raise RuntimeError('Number of poses differs from detections')
                    for box, det_score, identifier, pose in zip(boxes, det_scores, identifiers, poses):
                        points = np.asarray(pose.pred_instances.keypoints)[0]
                        scores = np.asarray(pose.pred_instances.keypoint_scores)[0]
                        if points.shape != (30, 2) or scores.shape != (30,):
                            raise RuntimeError('Expected 30 keypoints and scores')
                        finite = np.isfinite(points).all(axis=1) & np.isfinite(scores)
                        in_frame = (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
                        valid = finite & in_frame & (scores >= args.point_threshold)
                        smooth = tracker.smooth(identifier, points, valid)
                        track = tracker.tracks[identifier]
                        clipped = bool(box[0] <= 1 or box[1] <= 1 or box[2] >= width-1 or box[3] >= height-1)
                        partial_boxes += clipped
                        people_count += 1
                        valid_points += int(valid.sum())
                        confidence_sum += float(scores[np.isfinite(scores)].sum())
                        track_ids.add(identifier)
                        people.append(dict(track_id=identifier, segment_id=track.segment_id,
                                           bbox_xyxy=box.tolist(), detection_score=float(det_score), bbox_touches_border=clipped,
                                           keypoints_raw=serialise_points(points, scores),
                                           keypoints=serialise_points(smooth, scores), valid=valid.tolist()))
                        for a, b in SKELETON_EDGES:
                            if valid[a] and valid[b]:
                                cv2.line(frame, tuple(smooth[a].astype(int)), tuple(smooth[b].astype(int)), (0, 220, 150), 2)
                        for point, show in zip(smooth, valid):
                            if show:
                                cv2.circle(frame, tuple(point.astype(int)), 3, (0, 160, 255), -1)
                        x0, y0, x1, y1 = box.astype(int)
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (220, 150, 0), 1)
                        cv2.putText(frame, f'Track {identifier} / segment {track.segment_id}', (x0, max(25, y0-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 180), 2)
                elapsed = time.perf_counter() - frame_start
                inference_seconds += elapsed
                poses_file.write(json.dumps(dict(type='frame', frame=count, timestamp_s=count/fps,
                                                scene_cut=scene_cut, processing_ms=elapsed*1000, people=people), allow_nan=False)+'\n')
                if writer is not None:
                    writer.write(frame)
                count += 1
                if count == 1 or count % 5 == 0:
                    if args.preview_file:
                        preview = frame
                        if max(frame.shape[:2]) > 960:
                            scale = 960 / max(frame.shape[:2])
                            preview = cv2.resize(frame, (round(width*scale), round(height*scale)))
                        encoded, jpeg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if encoded:
                            temp = args.preview_file.with_suffix('.tmp')
                            temp.write_bytes(jpeg.tobytes())
                            temp.replace(args.preview_file)
                    if args.progress_file:
                        temp = args.progress_file.with_suffix('.tmp')
                        temp.write_text(json.dumps(dict(frames=count, total_frames=total_frames,
                                                       frames_with_people=detected_frames,
                                                       video_seconds=count/fps, elapsed_seconds=time.perf_counter()-start)))
                        temp.replace(args.progress_file)
                if count % 100 == 0:
                    print(f'{count} frames; {detected_frames} with people; {len(track_ids)} tracks', flush=True)
        if count == 0:
            raise RuntimeError('No video frames decoded')
        wall_seconds = time.perf_counter() - start
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    features = export_features(poses_path)
    report = dict(**header, status='completed', output=str(args.output.resolve()) if not args.no_video else None, frames=count,
                  frames_with_people=detected_frames, frames_without_people=count-detected_frames,
                  unique_tracks=len(track_ids), scene_cuts=cuts, person_observations=people_count,
                  bbox_border_observations=partial_boxes,
                  missing_keypoint_fraction=1-valid_points/(people_count*30) if people_count else None,
                  mean_model_score=confidence_sum/(people_count*30) if people_count else None,
                  mean_inference_ms=inference_seconds/count*1000, processing_fps=count/wall_seconds,
                  audio=False, poses=str(poses_path.resolve()), features=features)
    report['type'] = 'summary'
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
