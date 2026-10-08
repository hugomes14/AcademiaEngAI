"""Classify an exported pose video causally using the experimental posture model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from src.posturaai.classification import CLASSES, load_windows, predict_scores
from src.posturaai.video_features import export_features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--poses',type=Path,required=True)
    parser.add_argument('--pose-video',type=Path,required=True,help='Existing annotated MP4 from infer_video')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-training-source',action='store_true',help='Allow general inference on known training footage; never report this as held-out evaluation')
    args=parser.parse_args()
    if args.output.resolve() in (args.pose_video.resolve(),args.poses.resolve(),args.model.resolve()):
        parser.error('Output cannot overwrite inputs')
    if args.output.suffix.lower()!='.mp4':parser.error('Output must be .mp4')
    model=json.loads(args.model.read_text())
    with args.poses.open() as source:pose_header=json.loads(next(source))
    if pose_header['checkpoint_sha256']!=model['pose_checkpoint_sha256']:
        raise ValueError('Pose checkpoint differs from classifier training')
    training_source_match = pose_header['input_sha256'] in model['trained_video_sha256']
    if training_source_match and not args.allow_training_source:
        raise ValueError('Final visual test must not reuse a training video')
    features=export_features(args.poses,plot=False)
    header,windows=load_windows(features['jsonl'],args.poses,window_seconds=model['window_seconds'],
                                stride_seconds=model['stride_seconds'],min_valid_fraction=model['min_valid_fraction'])
    scores=predict_scores(model,np.array([w['descriptor'] for w in windows],float)) if windows else []
    predictions=[]
    by_end={}
    for window,score in zip(windows,scores):
        label=int(score>=0.5)
        record=dict(**window,score_boa=float(score),score_ma=float(1-score),prediction=CLASSES[label],
                    display='inconclusivo' if 0.4<float(score)<0.6 else CLASSES[label])
        predictions.append(record)
        by_end.setdefault(record['end_frame'],[]).append(record)
    records={}
    with args.poses.open() as source:
        next(source)
        for line in source:
            row=json.loads(line);records[row['frame']]=row
    import cv2
    cv2.setNumThreads(1)
    capture=cv2.VideoCapture(str(args.pose_video));writer=None
    processed=classified=uncertain=0
    labels_count={name:0 for name in CLASSES};active={}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        if not capture.isOpened():raise RuntimeError('Cannot open pose video')
        fps=capture.get(cv2.CAP_PROP_FPS)
        if abs(fps-pose_header['fps'])>1e-3:raise ValueError('Pose video FPS differs from exported points')
        while True:
            ok,frame=capture.read()
            if not ok:break
            if processed not in records:raise ValueError('Pose video has extra frames')
            row=records[processed]
            for prediction in by_end.get(processed,[]):
                active[(prediction['track_id'],prediction['segment_id'])]=prediction
            dominant=max(row['people'],key=lambda p:(p['bbox_xyxy'][2]-p['bbox_xyxy'][0])*(p['bbox_xyxy'][3]-p['bbox_xyxy'][1])) if row['people'] else None
            keys={(dominant['track_id'],dominant['segment_id'])} if dominant else set()
            active={key:value for key,value in active.items() if key in keys and processed-value['end_frame']<round(fps*0.5)}
            if writer is None:
                h,w=frame.shape[:2]
                writer=cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
                if not writer.isOpened():raise RuntimeError('Cannot write classified video')
            cv2.rectangle(frame,(0,0),(frame.shape[1],80),(15,15,15),-1)
            cv2.putText(frame,'PosturaAI - CLASSIFICADOR EXPERIMENTAL',(8,19),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            if active:
                prediction=max(active.values(),key=lambda r:r['end_frame'])
                classified+=1
                if prediction['display']=='inconclusivo':
                    title='Inconclusivo';color=(0,200,255);uncertain+=1
                else:
                    good=prediction['display']=='boa_postura'
                    title='Boa postura' if good else 'Ma postura';color=(80,220,80) if good else (80,80,255)
                    labels_count[prediction['display']]+=1
                text=f"Score boa: {prediction['score_boa']:.2f} / ma: {prediction['score_ma']:.2f}"
            else:
                title='A recolher sequencia' if row['people'] else 'Sem deteccao de pessoa'
                text='Sem previsao disponivel';color=(180,180,180)
            cv2.putText(frame,title,(8,44),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
            cv2.putText(frame,text,(8,66),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1)
            writer.write(frame);processed+=1
        if processed!=len(records):raise ValueError('Pose video is incomplete')
    finally:
        capture.release()
        if writer is not None:writer.release()
    prediction_path=args.output.with_suffix('.predictions.jsonl')
    with prediction_path.open('w') as target:
        target.write(json.dumps(dict(type='metadata',schema='posturaai.predictions.v1',model=str(args.model.resolve()),
                                    video_id=header['video_id'],causal=True,score_note=model['score_note'],labelled_ground_truth=False))+'\n')
        for record in predictions:target.write(json.dumps(dict(type='prediction',**record),allow_nan=False)+'\n')
    summary=dict(status='completed',model=str(args.model.resolve()),input=pose_header['input'],output=str(args.output.resolve()),
                 classifier_algorithm=model.get('algorithm','logistic'),
                 training_source_match=training_source_match,
                 video_sha256=pose_header['input_sha256'],frames=processed,windows=len(predictions),frames_with_prediction=classified,
                 frames_without_prediction=processed-classified,inconclusive_frames=uncertain,predicted_frames=labels_count,
                 predictions=str(prediction_path.resolve()),ground_truth=None,accuracy=None,experimental=True,
                 note='Visual inference only. Grouped validation is weak; outputs are uncalibrated experimental predictions.')
    args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
