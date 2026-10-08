"""Local web jobs. GPU dependencies remain in the dedicated PosturaAI environment."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
from threading import Lock
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
UPLOAD_MB = 512
TERMINAL = {'completed', 'failed'}
LOGGER = logging.getLogger(__name__)


class QueueFull(ValueError):
    pass


class VideoJobs:
    """One inference at a time, with durable results and a bounded local queue.

    Run with a single Flask process. Threads serve HTTP while the worker launches
    the pose interpreter; status and previews are replaced atomically on disk.
    """
    def __init__(self, root=ROOT, jobs_dir=None):
        self.root = Path(root)
        self.jobs_dir = Path(jobs_dir) if jobs_dir else self.root / 'outputs/web/jobs'
        self.python = self.root / '.venv/bin/python'
        self.checkpoint = self.root / 'outputs/stage1_experimental/best.pth'
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='postura')
        self.lock = Lock()
        self.pending = 0
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        for status_file in self.jobs_dir.glob('*/status.json'):
            status = json.loads(status_file.read_text())
            if status['status'] not in TERMINAL:
                self._update(status_file.parent, status='failed', stage='failed', message='A análise foi interrompida pelo reinício da app. Importa o vídeo novamente.')

    @property
    def classifier(self):
        manifest = self.root / 'outputs/classification/active_classifier.json'
        if not manifest.is_file():
            return self.root / 'outputs/classification/posture_baseline_01/classifier.json'
        selected = (self.root / json.loads(manifest.read_text())['relative_path']).resolve()
        if not selected.is_relative_to((self.root / 'outputs/classification').resolve()):
            raise ValueError('Active classifier must belong to the local classification directory')
        return selected

    def available(self):
        return all(path.is_file() for path in (self.python, self.checkpoint,
                   self.checkpoint.parent / 'effective_config.py', self.classifier))

    def directory(self, job_id):
        if not re.fullmatch(r'[0-9a-f]{32}', job_id):
            raise FileNotFoundError('Análise inexistente.')
        directory = self.jobs_dir / job_id
        if not (directory / 'status.json').is_file():
            raise FileNotFoundError('Análise inexistente.')
        return directory

    def _update(self, directory, **changes):
        path = directory / 'status.json'
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update(changes)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(current, ensure_ascii=False, allow_nan=False))
        temp.replace(path)

    def create(self, upload):
        if not self.available():
            raise RuntimeError('O serviço de vídeo ainda não está configurado neste servidor.')
        suffix = Path(upload.filename or '').suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise ValueError('Escolhe um vídeo MP4, MOV, AVI, MKV ou WebM.')
        with self.lock:
            if self.pending >= 4:
                raise QueueFull('Há várias análises em espera. Tenta novamente quando uma terminar.')
            self.pending += 1
        directory = self.jobs_dir / uuid4().hex
        try:
            directory.mkdir()
            upload.save(directory / ('input' + suffix))
            if not (directory / ('input' + suffix)).stat().st_size:
                raise ValueError('O vídeo está vazio.')
            self._update(directory, id=directory.name, name=Path(upload.filename).name,
                         status='queued', stage='queued', percent=0,
                         message='Vídeo recebido. A aguardar análise.',
                         created_at=datetime.now(timezone.utc).isoformat())
            self.executor.submit(self._run, directory, suffix)
            return self.status(directory.name)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            with self.lock:
                self.pending -= 1
            raise

    def status(self, job_id):
        directory = self.directory(job_id)
        state = json.loads((directory / 'status.json').read_text())
        progress_path = directory / 'progress.json'
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            state['progress'] = progress
            if state['stage'] == 'pose':
                total = progress['total_frames']
                state['percent'] = min(84, round(5 + 79 * progress['frames'] / total)) if total > 0 else 5
        state['has_preview'] = (directory / 'preview.jpg').is_file()
        return state

    def _command(self, directory, command, timeout=21600):
        with (directory / 'worker.log').open('a') as log:
            subprocess.run(command, cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout, env={**os.environ, 'PYTHONUNBUFFERED': '1'})

    def _run(self, directory, suffix):
        try:
            classifier = self.classifier
            classifier_bytes = classifier.read_bytes()
            self._update(directory, status='running', stage='pose', percent=5,
                         classifier_run=classifier.parent.name,
                         classifier_algorithm=json.loads(classifier_bytes).get('algorithm', 'logistic'),
                         classifier_sha256=hashlib.sha256(classifier_bytes).hexdigest(),
                         message='A detetar pessoas e acompanhar os pontos do corpo…')
            self._command(directory, [str(self.python), '-m', 'scripts.infer_video',
                '--input', str(directory / ('input' + suffix)), '--checkpoint', str(self.checkpoint),
                '--output', str(directory / 'pose.mp4'), '--det-threshold', '0.4',
                '--device', os.environ.get('POSTURA_DEVICE', 'cuda:0'),
                '--progress-file', str(directory / 'progress.json'),
                '--preview-file', str(directory / 'preview.jpg')])
            self._update(directory, stage='classification', percent=85,
                         message='A classificar as sequências de movimento…')
            self._command(directory, [str(self.python), '-m', 'scripts.classify_video',
                '--model', str(classifier), '--poses', str(directory / 'pose.poses.jsonl'),
                '--pose-video', str(directory / 'pose.mp4'), '--output', str(directory / 'classified.mp4'),
                '--allow-training-source'])
            self._update(directory, stage='encoding', percent=95,
                         message='A preparar o vídeo para reprodução…')
            self._command(directory, [str(self.python), '-m', 'scripts.web_encode',
                '--input', str(directory / 'classified.mp4'), '--output', str(directory / 'result.mp4')], timeout=3600)
            pose = json.loads((directory / 'pose.json').read_text())
            classification = json.loads((directory / 'classified.json').read_text())
            self._update(directory, status='completed', stage='completed', percent=100,
                         message='Análise concluída.', result=dict(frames=pose['frames'],
                         frames_with_people=pose['frames_with_people'],
                         frames_with_prediction=classification['frames_with_prediction'],
                         inconclusive_frames=classification['inconclusive_frames'],
                         predicted_frames=classification['predicted_frames']))
        except Exception:
            LOGGER.exception('PosturaAI job failed: %s (see worker.log)', directory.name)
            self._update(directory, status='failed', stage='failed',
                         message='Não foi possível analisar este vídeo. Verifica se abre normalmente e tenta novamente.')
        finally:
            with self.lock:
                self.pending -= 1
