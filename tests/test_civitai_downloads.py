import tempfile
import time
import unittest
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch_lib import api, downloader


class CivitaiDownloadUrlTests(unittest.TestCase):
    def test_with_api_token_appends_token_to_download_url_with_existing_query(self):
        url = "https://civitai.red/api/download/models/3045803?type=Model&format=SafeTensor"
        result = api.with_api_token(url, "secret-token")
        self.assertEqual(
            result,
            "https://civitai.red/api/download/models/3045803?type=Model&format=SafeTensor&token=secret-token",
        )

    def test_with_api_token_does_not_change_non_download_or_non_civitai_urls(self):
        self.assertEqual(
            api.with_api_token("https://civitai.com/models/123", "secret-token"),
            "https://civitai.com/models/123",
        )
        self.assertEqual(
            api.with_api_token("https://example.com/api/download/models/123", "secret-token"),
            "https://example.com/api/download/models/123",
        )


class BatchQueueStatusTests(unittest.TestCase):
    def test_batch_queue_processes_pending_items_and_marks_completed(self):
        bq = downloader.BatchQueue()
        with tempfile.TemporaryDirectory() as tmp:
            item = downloader.BatchItem(
                url="https://civitai.com/models/1",
                filename="model.safetensors",
                status="pending",
                _dl_url="https://civitai.com/api/download/models/2",
                _dest_dir=Path(tmp),
            )
            bq.add_item(item)

            def fake_download(self, task, *args, **kwargs):
                task.done = True
                task.dest.write_bytes(b"ok")

            with mock.patch.object(downloader.DownloadQueue, "download", fake_download):
                bq.start(api_key="token")
                deadline = time.time() + 2
                while bq.running and time.time() < deadline:
                    time.sleep(0.01)

            self.assertFalse(bq.running)
            self.assertEqual(item.status, "completed")
            self.assertEqual(item.progress, 1.0)
            self.assertEqual(bq.summary, "1/1 terminé(s), 0 erreur(s)")


if __name__ == "__main__":
    unittest.main()
