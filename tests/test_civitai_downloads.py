import tempfile
import time
import unittest
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch_lib import api, downloader, utils


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

    def test_parse_model_url_handles_all_variants(self):
        # Standard model URL
        self.assertEqual(api.parse_model_url("https://civitai.com/models/12345"), ("12345", None))
        # With slug
        self.assertEqual(api.parse_model_url("https://civitai.com/models/12345/my-model-slug"), ("12345", None))
        # With modelVersionId query param
        self.assertEqual(api.parse_model_url("https://civitai.com/models/12345?modelVersionId=67890"), ("12345", "67890"))
        # With civitai.red
        self.assertEqual(api.parse_model_url("https://civitai.red/models/12345?modelVersionId=67890"), ("12345", "67890"))
        # Direct version URL
        self.assertEqual(api.parse_model_url("https://civitai.com/model-versions/67890"), (None, "67890"))
        # Direct download URL
        self.assertEqual(api.parse_model_url("https://civitai.red/api/download/models/67890?type=Model"), (None, "67890"))
        # Pure numeric ID
        self.assertEqual(api.parse_model_url("12345"), ("12345", None))


class ModelUtilsTests(unittest.TestCase):
    def test_info_file_path_preserves_dots_in_model_name(self):
        p1 = Path("/models/Lora/flux.1-dev.safetensors")
        self.assertEqual(utils.info_file_path(p1).name, "flux.1-dev.civitai.info")

        p2 = Path("/models/Stable-diffusion/v1.5_pruned_emaonly.safetensors")
        self.assertEqual(utils.info_file_path(p2).name, "v1.5_pruned_emaonly.civitai.info")

    def test_preview_file_path_preserves_dots_in_model_name(self):
        p = Path("/models/Lora/flux.1-dev.safetensors")
        self.assertEqual(utils.preview_file_path(p, "png").name, "flux.1-dev.preview.png")

    def test_iter_model_files_does_not_duplicate_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lora_dir = root / "models" / "Lora"
            lora_dir.mkdir(parents=True)
            (lora_dir / "test.safetensors").write_bytes(b"model")

            found = utils.iter_model_files(root)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].name, "test.safetensors")


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
            self.assertEqual(bq.summary, "1/1 completed, 0 error(s)")


if __name__ == "__main__":
    unittest.main()
