import tempfile
import unittest
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch_lib import downloader, utils
from scripts.sdfch_main import get_card_info_data


class FakeResponse:
    def __init__(self, chunks, status_code=200, headers=None):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = headers or {"Content-Length": str(sum(len(c) for c in chunks))}

    def iter_content(self, chunk_size=1024):
        for c in self._chunks:
            yield c

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def close(self):
        pass


class DownloadEngineTests(unittest.TestCase):
    def test_download_uses_part_file_and_renames_on_success(self):
        q = downloader.DownloadQueue()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "model.safetensors"
            content = b"safetensors content here"
            fake_resp = FakeResponse([content], status_code=200)

            task = downloader.DownloadTask(
                url="https://civitai.com/api/download/models/123",
                dest=dest,
                filename="model.safetensors",
            )

            with mock.patch("requests.get", return_value=fake_resp):
                q.download(task=task, download_preview=False)

            self.assertTrue(task.done)
            self.assertFalse(dest.with_name("model.safetensors.part").exists())
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_bytes(), content)

    def test_download_fails_and_cleans_part_if_sha256_mismatch(self):
        q = downloader.DownloadQueue()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "corrupt.safetensors"
            fake_resp = FakeResponse([b"corrupt data"], status_code=200)

            task = downloader.DownloadTask(
                url="https://civitai.com/api/download/models/123",
                dest=dest,
                filename="corrupt.safetensors",
                sha256_expected="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

            with mock.patch("requests.get", return_value=fake_resp):
                q.download(task=task, download_preview=False)

            self.assertFalse(task.done)
            self.assertIn("mismatch", task.error.lower())
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_name("corrupt.safetensors.part").exists())

    def test_download_handles_200_fallback_when_resuming(self):
        q = downloader.DownloadQueue()
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "partial.safetensors"
            part = dest.with_name("partial.safetensors.part")
            # Existing part with 5 bytes
            part.write_bytes(b"hello")

            # Server returns full content (200 OK) with 10 bytes instead of 206
            full_data = b"0123456789"
            fake_resp = FakeResponse([full_data], status_code=200)

            task = downloader.DownloadTask(
                url="https://civitai.com/api/download/models/123",
                dest=dest,
                filename="partial.safetensors",
            )

            with mock.patch("requests.get", return_value=fake_resp):
                q.download(task=task, download_preview=False)

            self.assertTrue(dest.exists())
            # Content should not be corrupted by appending to existing 5 bytes
            self.assertEqual(dest.read_bytes(), full_data)


class CardInfoEndpointTests(unittest.TestCase):
    def test_card_info_data_returns_civitai_url_and_trigger_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lora_dir = root / "models" / "Lora"
            lora_dir.mkdir(parents=True)
            model_file = lora_dir / "my_style.safetensors"
            model_file.write_bytes(b"dummy")

            # Create info file and txt file
            info_file = lora_dir / "my_style.civitai.info"
            info_file.write_text('{"id": 456, "modelId": 123, "trainedWords": ["keyword1", "keyword2"]}', encoding="utf-8")

            with mock.patch("ch_lib.utils.get_sd_root", return_value=root):
                data = get_card_info_data("my_style")
                self.assertIn("civitai.com/models/123?modelVersionId=456", data["civitai_url"])
                self.assertIn("keyword1, keyword2", data["trigger_words"])


if __name__ == "__main__":
    unittest.main()
