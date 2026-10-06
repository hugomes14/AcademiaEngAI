"""Experimental geometry classifier with recording-group validation and JSON weights."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

CLASSES = ('ma_postura', 'boa_postura')
AGGREGATION = 'posturaai.geometry_summary.v1'
FEATURE_FAMILIES = ('knee', 'hip', 'ankle', 'elbow', 'shoulder', 'trunk_abs', 'head_trunk')
STATISTICS = ('mean', 'std', 'p10', 'median', 'p90')
DESCRIPTOR_NAMES = tuple(f'{family}_{stat}' for family in FEATURE_FAMILIES for stat in STATISTICS)


def window_descriptor(observations):
    """Pool left/right projected geometry; no speeds, IDs, pixels or video text."""
    vector = []
    for family in FEATURE_FAMILIES:
        values = []
        for observation in observations:
            features = observation['features']
            if family == 'trunk_abs':
                candidates = [features.get('trunk_image_lean_deg')]
            elif family == 'head_trunk':
                candidates = [features.get('head_trunk_projected_angle_deg')]
            else:
                candidates = [features.get(side+'_'+family+'_projected_angle_deg') for side in ('left','right')]
            for value in candidates:
                if value is not None and np.isfinite(value):
                    values.append(abs(value) if family == 'trunk_abs' else value)
        if values:
            vector.extend([np.mean(values), np.std(values), *np.percentile(values, [10,50,90])])
        else:
            vector.extend([np.nan]*len(STATISTICS))
    return np.asarray(vector, dtype=float)


def load_windows(features_path, poses_path, *, intervals=None, window_seconds=1.0,
                 stride_seconds=0.25, min_valid_fraction=0.6):
    if window_seconds <= 0 or stride_seconds <= 0 or not 0 <= min_valid_fraction <= 1:
        raise ValueError('Invalid window settings')
    features_path, poses_path = Path(features_path), Path(poses_path)
    with poses_path.open() as source:
        pose_header = json.loads(next(source))
        dominant = {}
        for line in source:
            frame = json.loads(line)
            if frame.get('people'):
                # A background person must not inherit a foreground runner label.
                dominant[frame['frame']] = max(frame['people'], key=lambda p:
                    (p['bbox_xyxy'][2]-p['bbox_xyxy'][0])*(p['bbox_xyxy'][3]-p['bbox_xyxy'][1]))['track_id']
    segments = {}
    with features_path.open() as source:
        header = json.loads(next(source))
        if header.get('schema') != 'posturaai.features.v1' or pose_header.get('schema') != 'posturaai.poses.v1':
            raise ValueError('Expected exported poses and features schemas')
        if header['video_id'] != pose_header['video_id']:
            raise ValueError('Poses/features video mismatch')
        for line in source:
            frame = json.loads(line)
            for person in frame.get('people', []):
                if person['track_id'] != dominant.get(frame['frame']):
                    continue
                key = (person['track_id'],person['segment_id'])
                segments.setdefault(key, []).append(dict(frame=frame['frame'],timestamp_s=frame['timestamp_s'],**person))
    fps = header['fps']
    length, stride = max(2,round(window_seconds*fps)), max(1,round(stride_seconds*fps))
    windows = []
    for (track,segment), observations in sorted(segments.items()):
        for offset in range(0,len(observations)-length+1,stride):
            window = observations[offset:offset+length]
            if any(b['frame'] != a['frame']+1 or abs(b['timestamp_s']-a['timestamp_s']-1/fps)>1e-5
                   for a,b in zip(window,window[1:])):
                continue
            fraction = sum(sum(p['valid']) for p in window)/(30*length)
            vector = window_descriptor(window)
            if fraction < min_valid_fraction or np.isfinite(vector).sum() < 20:
                continue
            start, end = window[0]['timestamp_s'], (window[-1]['frame']+1)/fps
            label, evidence = None, None
            if intervals is not None:
                matching = [interval for interval in intervals if start >= interval['start_s']-1e-8 and end <= interval['end_s']+1e-8]
                if len(matching) != 1:
                    continue
                label, evidence = matching[0]['label'], matching[0]['evidence']
            windows.append(dict(video_id=header['video_id'], track_id=track, segment_id=segment,
                                start_frame=window[0]['frame'],end_frame=window[-1]['frame'],
                                start_s=start,end_s=end,valid_fraction=fraction,label=label,evidence=evidence,
                                descriptor=[float(x) if np.isfinite(x) else None for x in vector]))
    return header, windows


def fit_classifier(x, y, groups, *, regularization=0.1):
    x,y,groups = np.asarray(x,dtype=float),np.asarray(y,dtype=int),np.asarray(groups)
    if x.ndim != 2 or x.shape[1] != len(DESCRIPTOR_NAMES) or len(x) != len(y) or len(y) != len(groups):
        raise ValueError('Invalid training arrays')
    if set(y.tolist()) != {0,1}:
        raise ValueError('Training requires both posture classes')
    if regularization <= 0:
        raise ValueError('Regularization must be positive')
    fill=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0 for col in x.T])
    imputed=np.where(np.isfinite(x),x,fill)
    mean,scale=imputed.mean(axis=0),imputed.std(axis=0)
    scale[scale<1e-8]=1
    z=(imputed-mean)/scale
    class_groups={label:set(groups[y==label]) for label in (0,1)}
    counts=Counter(zip(y.tolist(),groups.tolist()))
    weights=np.array([1/(2*len(class_groups[label])*counts[(label,group)]) for label,group in zip(y,groups)])
    def objective(parameters):
        logits=z@parameters[:-1]+parameters[-1]
        loss=np.sum(weights*(np.logaddexp(0,logits)-y*logits))+regularization*np.sum(parameters[:-1]**2)/2
        residual=weights*(expit(logits)-y)
        gradient=np.r_[z.T@residual+regularization*parameters[:-1],residual.sum()]
        return float(loss),gradient
    result=minimize(objective,np.zeros(z.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Classifier optimizer failed: '+str(result.message))
    return dict(schema='posturaai.classifier.v1',aggregation=AGGREGATION,classes=list(CLASSES),
                descriptor_names=list(DESCRIPTOR_NAMES),impute=fill.tolist(),mean=mean.tolist(),scale=scale.tolist(),
                weights=result.x[:-1].tolist(),bias=float(result.x[-1]),regularization=regularization,
                optimizer=dict(method='L-BFGS-B',iterations=int(result.nit),objective=float(result.fun)),
                training_groups=sorted(set(groups.tolist())),experimental=True,
                score_note='Uncalibrated sigmoid model score; not clinical probability or validated posture assessment')


def predict_scores(model,x):
    if model.get('aggregation') != AGGREGATION or model.get('descriptor_names') != list(DESCRIPTOR_NAMES):
        raise ValueError('Classifier feature contract mismatch')
    x=np.asarray(x,dtype=float)
    if x.ndim != 2 or x.shape[1] != len(DESCRIPTOR_NAMES):
        raise ValueError('Invalid descriptor shape')
    imputed=np.where(np.isfinite(x),x,np.array(model['impute']))
    return expit(((imputed-np.array(model['mean']))/np.array(model['scale']))@np.array(model['weights'])+model['bias'])


def classification_metrics(y,scores):
    y,scores=np.asarray(y,int),np.asarray(scores,float)
    predicted=(scores>=0.5).astype(int)
    confusion=np.array([[np.sum((y==a)&(predicted==b)) for b in (0,1)] for a in (0,1)])
    per_class={}
    for label,name in enumerate(CLASSES):
        tp=int(confusion[label,label]);fp=int(confusion[:,label].sum()-tp);fn=int(confusion[label,:].sum()-tp)
        precision=tp/(tp+fp) if tp+fp else 0
        recall=tp/(tp+fn) if tp+fn else 0
        # Step-wise area under the precision-recall curve (average precision).
        class_scores=scores if label else 1-scores
        order=np.argsort(-class_scores,kind='stable')
        positive=(y[order]==label).astype(int)
        cumulative=np.cumsum(positive)
        boundaries=np.r_[np.flatnonzero(np.diff(class_scores[order])),len(order)-1]
        if positive.sum():
            r=cumulative[boundaries]/positive.sum()
            p=cumulative[boundaries]/(boundaries+1)
            average_precision=float(np.sum(np.diff(np.r_[0,r])*p))
        else:
            average_precision=None
        per_class[name]=dict(precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0,
                             average_precision=average_precision,support=int(confusion[label,:].sum()))
    return dict(samples=len(y),accuracy=float(np.mean(y==predicted)),
                balanced_accuracy=float(np.mean([p['recall'] for p in per_class.values() if p['support']])),
                macro_f1=float(np.mean([p['f1'] for p in per_class.values()])),
                confusion_matrix=confusion.tolist(),matrix_order=list(CLASSES),per_class=per_class)


def grouped_validation(x,y,groups):
    x,y,groups=np.asarray(x,float),np.asarray(y,int),np.asarray(groups)
    unique=sorted(set(groups.tolist()))
    for label in (0,1):
        if len(set(groups[y==label])) < 2:
            raise ValueError('Grouped validation requires each class in at least two independent groups')
    out_of_fold=np.full(len(y),np.nan)
    folds=[]
    for group in unique:
        test=groups==group;train=~test
        model=fit_classifier(x[train],y[train],groups[train])
        scores=predict_scores(model,x[test]);out_of_fold[test]=scores
        folds.append(dict(held_out_group=group,training_groups=sorted(set(groups[train].tolist())),
                          train_samples=int(train.sum()),validation_samples=int(test.sum()),
                          metrics=classification_metrics(y[test],scores)))
    return dict(protocol='leave-one-recording/athlete-group-out; no overlapping windows cross folds',
                folds=folds,metrics=classification_metrics(y,out_of_fold),scores=out_of_fold.tolist())
