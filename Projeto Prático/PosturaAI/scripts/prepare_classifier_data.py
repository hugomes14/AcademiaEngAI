"""Extract labelled source videos and an isolated visual test into one run directory."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    checksum=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,default=Path('data/running_posture/annotations.json'))
    parser.add_argument('--checkpoint',type=Path,default=Path('outputs/stage1_experimental/best.pth'))
    parser.add_argument('--test-video',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    annotations=json.loads(args.annotations.read_text())
    rows=[]
    for item in annotations['videos']:
        path=root/item['relative_path']
        checksum=digest(path)
        if checksum!=item['video_sha256']:raise ValueError('Annotation file refers to modified video: '+str(path))
        rows.append(dict(path=str(path),sha256=checksum,source_id=item['video_id'],role='labelled_source',
                         label_source=item['label_source'],annotation_file=str(args.annotations.resolve())))
    checksum=digest(args.test_video)
    if checksum in {r['sha256'] for r in rows}:raise ValueError('Visual test is duplicated in training sources')
    rows.append(dict(path=str(args.test_video.resolve()),sha256=checksum,source_id=checksum[:16],role='visual_test',label_source=None))
    run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    checkpoint_hash=digest(args.checkpoint)
    for row in rows:
        output=run/('test' if row['role']=='visual_test' else 'dataset')/row['source_id']/'pose.mp4'
        row['output']=str(output)
    (run/'video_catalog.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    for row in rows:
        output=Path(row['output']);output.parent.mkdir(parents=True,exist_ok=True)
        report_path=output.with_suffix('.json')
        if report_path.exists():
            report=json.loads(report_path.read_text())
            expected=dict(status='completed',input_sha256=row['sha256'],checkpoint_sha256=checkpoint_hash,
                          det_threshold=0.4,point_threshold=0.3,person_nms=0.3)
            if all(report.get(k)==v for k,v in expected.items()) and output.with_suffix('.poses.jsonl').is_file() and (row['role']!='visual_test' or output.is_file()):
                print('Reusing extraction: '+Path(row['path']).name,flush=True)
                continue
            raise ValueError('Existing extraction has incompatible settings; use a new --run-dir')
        command=[sys.executable,'-m','scripts.infer_video','--input',row['path'],'--checkpoint',str(args.checkpoint),
                 '--output',str(output),'--source-id',row['source_id'],'--det-threshold','0.4']
        if row['role']=='labelled_source':command.append('--no-video')
        print('Extracting: '+Path(row['path']).name,flush=True)
        with output.with_suffix('.console.log').open('w') as log:
            completed=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:raise RuntimeError('Pose extraction failed; see '+str(output.with_suffix('.console.log')))
    print('Dataset extraction completed; test video isolated from labels.')


if __name__=='__main__':main()
