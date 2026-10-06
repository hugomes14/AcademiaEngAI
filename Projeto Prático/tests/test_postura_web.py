"""HTTP contract and background-job safety without loading GPU models."""
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import app as web
from PosturaAI.web_service import VideoJobs


class ControlledJobs(VideoJobs):
    def __init__(self, root):
        super().__init__(root)
        self.release = threading.Event()

    def available(self):
        return True

    def _run(self, directory, suffix):
        self.release.wait(10)
        with self.lock:
            self.pending -= 1


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.jobs = ControlledJobs(Path(self.temp.name))
        self.manager = patch.object(web, '_postura_jobs', self.jobs)
        self.manager.start()
        self.client = web.app.test_client()

    def tearDown(self):
        self.jobs.release.set()
        self.jobs.executor.shutdown(wait=True)
        self.manager.stop()
        self.temp.cleanup()

    def upload(self, name='runner.mp4', content=b'video'):
        return self.client.post('/api/postura/jobs', data={'video': (io.BytesIO(content), name)})

    def test_catalog_and_service_page(self):
        self.assertIn(b'href="/postura-ai"', self.client.get('/').data)
        page = self.client.get('/postura-ai')
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'name="video"', page.data)
        self.assertIn(b'postura.js', page.data)

    def test_upload_returns_immediately_and_progress_can_be_resumed(self):
        response = self.upload('../../runner.mp4')
        self.assertEqual(response.status_code, 202)
        state = response.json
        directory = self.jobs.directory(state['id'])
        self.assertEqual((directory / 'input.mp4').read_bytes(), b'video')
        (directory / 'preview.jpg').write_bytes(b'jpeg-preview')
        (directory / 'progress.json').write_text(json.dumps(dict(frames=10, total_frames=100)))
        self.jobs._update(directory, status='running', stage='pose')
        resumed = self.client.get(state['status_url'])
        self.assertEqual(resumed.json['progress']['frames'], 10)
        self.assertTrue(resumed.json['has_preview'])
        self.assertIsNone(resumed.json['video_url'])
        self.assertEqual(resumed.headers['Cache-Control'], 'no-store')
        with self.client.get(resumed.json['preview_url']) as preview:
            self.assertEqual(preview.data, b'jpeg-preview')

    def test_completed_video_supports_browser_range_requests(self):
        state = self.upload().json
        directory = self.jobs.directory(state['id'])
        (directory / 'result.mp4').write_bytes(b'0123456789')
        url = f"/api/postura/jobs/{state['id']}/video"
        self.assertEqual(self.client.get(url).status_code, 404)
        self.jobs._update(directory, status='completed', stage='completed', percent=100)
        with self.client.get(url, headers={'Range': 'bytes=2-5'}) as response:
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.data, b'2345')
            self.assertEqual(response.content_type, 'video/mp4')

    def test_invalid_upload_and_paths_are_rejected(self):
        self.assertEqual(self.client.post('/api/postura/jobs').status_code, 400)
        self.assertEqual(self.upload('runner.txt').status_code, 400)
        self.assertEqual(self.upload(content=b'').status_code, 400)
        self.assertEqual(self.client.get('/api/postura/jobs/not-an-id').status_code, 404)
        self.assertEqual(self.client.get('/api/postura/jobs/' + 'f'*32).status_code, 404)
        self.assertEqual(list(self.jobs.jobs_dir.iterdir()), [])

    def test_queue_is_bounded(self):
        for _ in range(4):
            self.assertEqual(self.upload().status_code, 202)
        self.assertEqual(self.upload().status_code, 429)

    def test_oversize_video_returns_json_and_ritmo_keeps_smaller_limit(self):
        video = self.client.post('/api/postura/jobs', environ_overrides={'CONTENT_LENGTH': str(513*1024*1024)})
        self.assertEqual(video.status_code, 413)
        self.assertIn('512', video.json['erro'])
        ritmo = self.client.post('/api/prever', environ_overrides={'CONTENT_LENGTH': str(33*1024*1024)})
        self.assertEqual(ritmo.status_code, 413)
        self.assertIn('32', ritmo.json['erro'])

    def test_restart_marks_unfinished_job_as_interrupted(self):
        state = self.upload().json
        restarted = VideoJobs(self.jobs.root)
        try:
            status = restarted.status(state['id'])
            self.assertEqual(status['status'], 'failed')
            self.assertIn('reinício', status['message'])
        finally:
            restarted.executor.shutdown()

    def test_worker_failure_releases_queue_slot_and_exposes_safe_message(self):
        jobs = VideoJobs(Path(self.temp.name) / 'failure')
        directory = jobs.jobs_dir / ('1'*32)
        directory.mkdir()
        jobs.pending = 1
        jobs._update(directory, id=directory.name, status='queued', stage='queued')
        with patch.object(jobs, '_command', side_effect=RuntimeError('internal path')), self.assertLogs('PosturaAI.web_service', level='ERROR'):
            jobs._run(directory, '.mp4')
        self.assertEqual(jobs.pending, 0)
        self.assertEqual(jobs.status(directory.name)['status'], 'failed')
        self.assertNotIn('internal path', jobs.status(directory.name)['message'])
        jobs.executor.shutdown()


if __name__ == '__main__':
    unittest.main()
