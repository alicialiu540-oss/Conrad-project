import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import web_app


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        web_app.app.config.update(TESTING=True, JOB_ROOT=Path(self.temp.name))
        self.client = web_app.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def upload(self, name, content=b'video', mode='body'):
        return self.client.post('/api/jobs', data={'mode': mode, 'video': (io.BytesIO(content), name)}, headers={'X-Conrad-Upload': '1'})

    def test_home_and_cross_origin_protection(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.post('/api/jobs').status_code, 403)
        self.assertEqual(self.client.get('/', headers={'Host': 'evil.example'}).status_code, 403)

    def test_invalid_uploads(self):
        self.assertEqual(self.upload('bad.txt').status_code, 400)
        self.assertEqual(self.upload('empty.mp4', b'').status_code, 400)
        self.assertEqual(self.upload('video.mp4', mode='unknown').status_code, 400)

    def test_queue_is_bounded(self):
        with patch.object(web_app, 'slots') as slots:
            slots.acquire.return_value = False
            self.assertEqual(self.upload('video.mp4').status_code, 429)

    def test_isolated_job_and_download(self):
        with patch.object(web_app.executor, 'submit') as submit:
            response = self.upload('../../video.mp4')
            self.assertEqual(response.status_code, 202)
            job = response.json['id']
            path = Path(self.temp.name) / job
            self.assertTrue((path / 'input.mp4').exists())
            self.assertEqual(self.client.get(f'/api/jobs/{job}').json['state'], 'queued')
            output = path / 'output'
            output.mkdir()
            (output / 'summary.json').write_text('{}')
            web_app.save_status(path, state='complete')
            with self.client.get(f'/api/jobs/{job}/files/summary.json') as download:
                self.assertEqual(download.status_code, 200)
            self.assertEqual(self.client.get(f'/api/jobs/{job}/files/../status.json').status_code, 404)
            submit.assert_called_once()
            web_app.slots.release()

    def test_lstm_requires_reference(self):
        self.assertEqual(self.upload('movement.mp4', mode='lstm').status_code, 400)

    def test_two_uploads_with_identical_names_are_kept_separate(self):
        with patch.object(web_app.executor, 'submit') as submit:
            response = self.client.post('/api/jobs', data={
                'mode': 'lstm', 'video': (io.BytesIO(b'movement'), '../../same.mp4'),
                'reference': (io.BytesIO(b'reference'), '../../same.mp4'),
            }, headers={'X-Conrad-Upload': '1'})
            self.assertEqual(response.status_code, 202)
            path = Path(self.temp.name) / response.json['id']
            self.assertEqual((path / 'input.mp4').read_bytes(), b'movement')
            self.assertEqual((path / 'reference.mp4').read_bytes(), b'reference')
            self.assertEqual(submit.call_args.args[3:], ('same.mp4', 'same.mp4'))
            web_app.slots.release()

    def test_empty_reference_rejected_and_slot_released(self):
        with patch.object(web_app, 'slots') as slots:
            response = self.client.post('/api/jobs', data={
                'mode': 'lstm', 'video': (io.BytesIO(b'movement'), 'move.mp4'),
                'reference': (io.BytesIO(b''), 'reference.mp4'),
            }, headers={'X-Conrad-Upload': '1'})
            self.assertEqual(response.status_code, 400)
            slots.release.assert_called_once()
            self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_bad_ids_and_size_limit(self):
        self.assertEqual(self.client.get('/api/jobs/not-a-job').status_code, 404)
        with patch.dict(web_app.app.config, MAX_CONTENT_LENGTH=10):
            self.assertEqual(self.upload('video.mp4', b'12345678901234567890').status_code, 413)


if __name__ == '__main__':
    unittest.main()
