"""Train a first posture classifier from reviewed video intervals and pose features."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from src.posturaai.classification import CLASSES, DESCRIPTOR_NAMES, fit_classifier, grouped_validation, load_windows
from src.posturaai.video_features import export_features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,default=Path('data/running_posture/annotations.json'))
    args=parser.parse_args()
    run=args.run_dir.resolve();annotation_path=args.annotations.resolve()
    annotations=json.loads(annotation_path.read_text())
    if annotations.get('schema') != 'posturaai.annotations.v1':raise ValueError('Invalid annotation schema')
    catalog=json.loads((run/'video_catalog.json').read_text())
    by_id={row['source_id']:row for row in catalog}
    samples=[];video_reports=[];pose_hashes=set()
    for annotation in annotations['videos']:
        row=by_id.get(annotation['video_id'])
        if row is None:raise ValueError('Reviewed video not extracted: '+annotation['video_id'])
        if row['sha256'] != annotation['video_sha256']:raise ValueError('Annotation video hash mismatch')
        output=Path(row['output']);poses=output.with_suffix('.poses.jsonl')
        summary=json.loads(output.with_suffix('.json').read_text())
        if summary.get('status') != 'completed':raise ValueError('Incomplete pose extraction')
        export_features(poses,context_overrides={'view':annotation.get('view','unknown')},plot=False)
        header,windows=load_windows(output.with_suffix('.features.jsonl'),poses,intervals=annotation['intervals'])
        if header.get('model_checkpoint_sha256'):pose_hashes.add(header['model_checkpoint_sha256'])
        group=annotation.get('athlete_id') or annotation['source_id']
        for window in windows:
            window.update(group=group,source_id=annotation['source_id'],video_sha256=row['sha256'])
        samples.extend(windows)
        counts={name:sum(w['label']==i for w in windows) for i,name in enumerate(CLASSES)}
        video_reports.append(dict(path=row['path'],group=group,source_id=annotation['source_id'],windows=len(windows),classes=counts,intervals=annotation['intervals']))
        print(f'{Path(row["path"]).name}: {len(windows)} windows, {counts}',flush=True)
    if len(pose_hashes) != 1:raise ValueError('All videos must use the same RTMPose checkpoint')
    if not samples:raise ValueError('No usable labelled windows')
    x=np.asarray([s['descriptor'] for s in samples],dtype=float)
    y=np.asarray([s['label'] for s in samples],dtype=int)
    groups=np.asarray([s['group'] for s in samples])
    validation=grouped_validation(x,y,groups)
    model=fit_classifier(x,y,groups)
    model.update(pose_checkpoint_sha256=next(iter(pose_hashes)),window_seconds=1.0,stride_seconds=0.25,min_valid_fraction=0.6,
                 annotations_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                 trained_video_sha256=sorted({s['video_sha256'] for s in samples}),
                 annotation_file=str(annotation_path),label_source='user folders for original good videos; reviewed explicit author red/green intervals for mixed videos')
    run.mkdir(parents=True,exist_ok=True)
    with (run/'labelled_windows.jsonl').open('w') as target:
        target.write(json.dumps(dict(type='metadata',schema='posturaai.labelled_windows.v1',descriptor_names=list(DESCRIPTOR_NAMES),annotations_sha256=model['annotations_sha256']))+'\n')
        for sample,score in zip(samples,validation['scores']):
            sample['out_of_fold_score_boa']=score
            target.write(json.dumps(sample,ensure_ascii=False,allow_nan=False)+'\n')
    (run/'classifier.json').write_text(json.dumps(model,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    report=dict(schema='posturaai.classifier_training_report.v1',status='completed',model=str(run/'classifier.json'),
                algorithm='L2 regularized logistic regression on 35 geometry summaries, class/group balanced',
                samples=len(samples),groups=len(set(groups)),classes={name:int(np.sum(y==i)) for i,name in enumerate(CLASSES)},
                videos=video_reports,validation={k:v for k,v in validation.items() if k!='scores'},
                final_fit_uses_all_labelled_sources=True,test_video_used_for_training=False,
                limitations=['Only four source recordings; overlapping windows are correlated.',
                             'Labels are tutorial/user examples, not independent clinical assessment.',
                             'Views and slow motion vary; geometry-only inputs omit velocities but remain 2D projections.',
                             'Dominant person selected by box area; crowd tracking may still change identity.',
                             'RTMPose checkpoint uses experimental unaudited synthetic splits.',
                             'No labelled final-test ground truth; inference is a visual test.'])
    (run/'training_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    metrics=report['validation']['metrics']
    (run/'training_report.md').write_text(f'''# Primeiro classificador de postura

Treino concluído: {len(samples)} janelas de 1 segundo, {len(set(groups))} grupos de origem.
Classes: {report['classes']}.

Modelo: regressão logística com regularização L2, 35 descritores geométricos.
Validação: uma gravação/atleta de fora de cada vez; imputação e normalização
ajustadas apenas aos dados de treino de cada fold. Sem seleção de parâmetros
usando o vídeo de teste. O modelo final usa todas as fontes rotuladas.

Resultados fora de grupo: accuracy {metrics['accuracy']:.3f}, balanced accuracy
{metrics['balanced_accuracy']:.3f}, macro F1 {metrics['macro_f1']:.3f}.
As janelas sobrepostas não são observações independentes; quatro gravações
não permitem estimar generalização com confiança.

Rótulos corrigidos conforme annotations.json. Vídeos mistos têm trechos bons,
maus e excluídos; o nome da pasta original não substitui esses rótulos.
Os rótulos refletem as demonstrações dos autores, sem validação clínica.
As features dependem da projeção da câmara. O modelo não recebe imagem, texto,
identidade, velocidade de reprodução nem diretório como entrada.

Artefactos: classifier.json, labelled_windows.jsonl e training_report.json.
''',encoding='utf-8')
    print(json.dumps({'model':str(run/'classifier.json'),'samples':len(samples),'validation':metrics},indent=2))


if __name__=='__main__':main()
