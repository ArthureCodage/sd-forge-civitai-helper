import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch_lib import api, card_index, downloader, model_manager, utils
from scripts import sdfch_main

SECRET = "s3cr3t-token-value"


class FakeJsonResponse:
    def __init__(self, status_code=200, payload=None, headers=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = text
        self.ok = status_code < 400

    def json(self):
        return self._payload


class FakeStream:
    """Streaming download response; `fail_after` chunks raises a network error."""

    def __init__(self, chunks, status_code=200, headers=None, fail_after=None):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = headers if headers is not None else {
            "Content-Length": str(sum(len(c) for c in chunks))
        }
        self._fail_after = fail_after

    def iter_content(self, chunk_size=1024):
        for i, c in enumerate(self._chunks):
            if self._fail_after is not None and i >= self._fail_after:
                raise requests.exceptions.ConnectionError("connection reset")
            yield c

    def close(self):
        pass


class FakeImage:
    def __init__(self, content=b"img", content_type="image/jpeg"):
        self.content = content
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        pass


def fake_gr():
    return types.SimpleNamespace(update=lambda **kw: kw)


def make_root(tmp):
    root = Path(tmp)
    lora = root / "models" / "Lora"
    lora.mkdir(parents=True)
    return root, lora


# ── Lot 1 : bugs visibles ────────────────────────────────────────────────────

class SearchTests(unittest.TestCase):
    def test_search_never_sends_page_with_query(self):
        with mock.patch("ch_lib.api._get", return_value={}) as get:
            api.search_models("portrait")
        params = get.call_args.kwargs["params"]
        self.assertNotIn("page", params)
        self.assertNotIn("cursor", params)
        self.assertEqual(params["query"], "portrait")

    def test_search_passes_cursor_for_next_page(self):
        with mock.patch("ch_lib.api._get", return_value={}) as get:
            api.search_models("portrait", cursor="12|34")
        self.assertEqual(get.call_args.kwargs["params"]["cursor"], "12|34")

    def test_cb_search_returns_next_cursor(self):
        data = {
            "items": [{"id": 7, "name": "M", "type": "LORA", "stats": {"downloadCount": 3}}],
            "metadata": {"nextCursor": "abc"},
        }
        with mock.patch.object(sdfch_main, "gr", fake_gr()), \
             mock.patch("ch_lib.api.search_models", return_value=data):
            rows, status, cursor = sdfch_main.cb_search("m", "All", "", False)
        self.assertEqual(cursor, "abc")
        self.assertEqual(rows["value"][0][4], "https://civitai.com/models/7")
        self.assertIn("Next page", status)


class PreviewTests(unittest.TestCase):
    def test_videos_are_not_used_as_thumbnails(self):
        images = [
            {"url": "https://image.civitai.com/k/a/original=true/clip.mp4", "type": "video"},
            {"url": "https://image.civitai.com/k/b/original=true/clip.mp4"},
            {"url": "https://image.civitai.com/k/c/original=true/pic.jpeg", "type": "image"},
        ]
        self.assertEqual(
            api.preview_image_urls(images),
            ["https://image.civitai.com/k/c/original=true/pic.jpeg"],
        )

    def test_preview_urls_fall_back_to_second_list(self):
        only_video = [{"url": "https://x/v.mp4", "type": "video"}]
        other = [{"url": "https://x/p.png"}]
        self.assertEqual(api.preview_image_urls(only_video, other), ["https://x/p.png"])
        self.assertEqual(api.preview_image_urls(None, []), [])

    def test_sized_image_url_requests_resized_copy(self):
        self.assertEqual(
            api.sized_image_url("https://image.civitai.com/k/uuid/original=true/1.jpeg"),
            "https://image.civitai.com/k/uuid/width=640/1.jpeg",
        )
        self.assertEqual(
            api.sized_image_url("https://image.civitai.com/k/uuid/width=450/1.jpeg", 512),
            "https://image.civitai.com/k/uuid/width=512/1.jpeg",
        )
        self.assertEqual(api.sized_image_url("https://example.com/original=true/1.jpeg"),
                         "https://example.com/original=true/1.jpeg")

    def test_video_response_is_skipped_and_next_image_becomes_main_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.safetensors"
            model.write_bytes(b"x")
            responses = [FakeImage(b"vid", "video/mp4"), FakeImage(b"pic", "image/png")]
            with mock.patch("requests.get", side_effect=responses):
                saved = model_manager.save_preview_images(
                    model, ["https://x/a", "https://x/b"], max_count=1
                )
            self.assertEqual(saved, Path(tmp) / "m.preview.png")
            self.assertEqual(saved.read_bytes(), b"pic")
            self.assertFalse((Path(tmp) / "m.preview.jpg").exists())

    def test_resized_preview_falls_back_to_original_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.safetensors"
            model.write_bytes(b"x")
            url = "https://image.civitai.com/k/u/original=true/1.jpeg"
            calls = []

            def fake_get(u, **kw):
                calls.append(u)
                if "width=" in u:
                    raise requests.exceptions.ConnectionError("nope")
                return FakeImage(b"orig")

            with mock.patch("requests.get", side_effect=fake_get):
                saved = model_manager.save_preview_images(model, [url])
            self.assertEqual(len(calls), 2)
            self.assertEqual(saved.read_bytes(), b"orig")


class UpdateCheckTests(unittest.TestCase):
    VERSIONS = [
        {"id": 30, "name": "v3", "baseModel": "Flux.1 D",
         "files": [{"name": "f.safetensors", "downloadUrl": "https://civitai.com/api/download/models/30"}]},
        {"id": 20, "name": "v2", "baseModel": "SDXL 1.0",
         "files": [{"name": "x2.safetensors", "downloadUrl": "https://civitai.com/api/download/models/20"}]},
        {"id": 10, "name": "v1", "baseModel": "SDXL 1.0",
         "files": [{"name": "x1.safetensors", "downloadUrl": "https://civitai.com/api/download/models/10"}]},
    ]

    def test_latest_version_respects_base_model(self):
        versions = api.extract_versions({"modelVersions": self.VERSIONS})
        self.assertEqual(api.latest_version_for(versions, "sdxl 1.0")["id"], 20)
        self.assertEqual(api.latest_version_for(versions, "")["id"], 30)
        self.assertEqual(api.latest_version_for(versions, "Pony")["id"], 30)
        self.assertIsNone(api.latest_version_for([], "SDXL 1.0"))

    def _run_check(self, infos):
        with tempfile.TemporaryDirectory() as tmp:
            root, lora = make_root(tmp)
            for name, info in infos.items():
                (lora / f"{name}.safetensors").write_bytes(b"x")
                (lora / f"{name}.civitai.info").write_text(json.dumps(info), encoding="utf-8")
            remote = {"modelVersions": self.VERSIONS}
            with mock.patch("ch_lib.utils.get_sd_root", return_value=root), \
                 mock.patch("ch_lib.api.fetch_model_info", return_value=remote) as fetch:
                model_manager.check_for_updates()
            state = model_manager.get_update_state()
            return list(state.results), fetch.call_count

    def test_newer_version_of_other_base_model_is_not_an_update(self):
        results, _ = self._run_check({"a": {"id": 20, "modelId": 5, "baseModel": "SDXL 1.0"}})
        self.assertEqual(results, [])

    def test_real_update_is_reported_and_model_is_queried_once(self):
        results, calls = self._run_check({
            "a": {"id": 10, "modelId": 5, "baseModel": "SDXL 1.0"},
            "b": {"id": 10, "modelId": 5, "base_model": "SDXL 1.0"},
        })
        self.assertEqual([r["latest_version_id"] for r in results], [20, 20])
        self.assertEqual(calls, 1)

    def test_selected_update_labels_map_back_to_rows(self):
        self.assertEqual(
            sdfch_main._selected_update_indexes(["2. B → v2", "1. A → v9", "9. gone", "junk"], 2),
            [1, 0],
        )

    def test_poll_keeps_ticked_updates(self):
        state = model_manager.get_update_state()
        state.reset()
        state.results = [{
            "model_name": "A", "local_version_id": 1, "latest_label": "v2",
            "model_type": "LORA", "model_path": "p",
        }]
        with mock.patch.object(sdfch_main, "gr", fake_gr()):
            _, _, table, picker = sdfch_main.cb_poll_updates(["1. A → v2", "5. old"])
        self.assertEqual(picker["choices"], ["1. A → v2"])
        self.assertEqual(picker["value"], ["1. A → v2"])
        self.assertEqual(len(table["value"]), 1)
        state.reset()


class DownloadedInfoTests(unittest.TestCase):
    def test_info_built_from_download_has_api_field_names(self):
        version = api.extract_versions({"modelVersions": [{
            "id": 456, "name": "v1", "baseModel": "SDXL 1.0", "trainedWords": ["kw"],
            "files": [{"name": "m.safetensors", "downloadUrl": "https://civitai.com/api/download/models/456"}],
        }]})[0]
        info = model_manager.build_info(version, {"id": 123, "name": "My model", "type": "LORA"})
        self.assertEqual(info["modelId"], 123)
        self.assertEqual(info["baseModel"], "SDXL 1.0")
        self.assertEqual(info["trainedWords"], ["kw"])
        self.assertEqual(info["model"]["name"], "My model")

    def test_info_built_from_api_version_keeps_embedded_model_name(self):
        info = model_manager.build_info(
            {"id": 9, "modelId": 4, "model": {"name": "Embedded", "type": "LORA"}}, {}, "abc"
        )
        self.assertEqual(info["model"]["name"], "Embedded")
        self.assertEqual(info["sha256"], "abc")


class HiddenFolderTests(unittest.TestCase):
    def test_models_are_found_when_install_path_has_a_dot_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lora = make_root(Path(tmp) / ".local" / "forge")
            (lora / "ok.safetensors").write_bytes(b"x")
            (lora / ".trash").mkdir()
            (lora / ".trash" / "old.safetensors").write_bytes(b"x")
            names = [p.name for p in utils.iter_model_files(root)]
            self.assertEqual(names, ["ok.safetensors"])


# ── Lot 2 : sécurité et fiabilité ────────────────────────────────────────────

class RedactionTests(unittest.TestCase):
    def test_redact_hides_tokens(self):
        msg = f"GET https://civitai.com/api/download/models/1?type=Model&token={SECRET} failed (Bearer {SECRET})"
        out = utils.redact(msg)
        self.assertNotIn(SECRET, out)
        self.assertIn("token=***", out)
        self.assertIn("type=Model", out)

    def test_network_error_never_shows_token_in_ui(self):
        q = downloader.DownloadQueue()
        with tempfile.TemporaryDirectory() as tmp:
            task = downloader.DownloadTask(
                url="https://civitai.com/api/download/models/1",
                dest=Path(tmp) / "m.safetensors", filename="m.safetensors",
            )

            def boom(url, **kw):
                raise requests.exceptions.ConnectionError(f"Max retries exceeded with url: {url}")

            with mock.patch("requests.get", side_effect=boom), \
                 mock.patch.object(downloader, "_sleep"):
                q.download(task=task, api_key=SECRET, download_preview=False)

            self.assertTrue(task.error)
            self.assertNotIn(SECRET, task.error)
            self.assertNotIn(SECRET, "\n".join(q.log))


class SafeFilenameTests(unittest.TestCase):
    def test_safe_filename_strips_paths_and_forbidden_characters(self):
        self.assertEqual(utils.safe_filename("../../evil.safetensors"), "evil.safetensors")
        self.assertEqual(utils.safe_filename("C:\\Windows\\x.ckpt"), "x.ckpt")
        self.assertEqual(utils.safe_filename('a<b>:"c|?*.pt'), "a_b___c___.pt")
        self.assertEqual(utils.safe_filename("name. "), "name")
        self.assertEqual(utils.safe_filename(".."), "download.bin")
        self.assertEqual(utils.safe_filename(None), "download.bin")
        self.assertEqual(utils.safe_filename("flux.1-dev.safetensors"), "flux.1-dev.safetensors")

    def test_extract_versions_sanitizes_file_names(self):
        versions = api.extract_versions({"modelVersions": [{
            "id": 1, "files": [{"name": "../up.safetensors", "downloadUrl": "https://x/y"}],
        }]})
        self.assertEqual(versions[0]["files"][0]["name"], "up.safetensors")


class ApiRetryTests(unittest.TestCase):
    def test_retries_on_429_then_succeeds(self):
        responses = [
            FakeJsonResponse(429, headers={"Retry-After": "7"}),
            FakeJsonResponse(503),
            FakeJsonResponse(200, {"ok": True}),
        ]
        with mock.patch("requests.get", side_effect=responses), \
             mock.patch.object(api, "_sleep") as sleep:
            self.assertEqual(api._get("https://civitai.com/api/v1/models/1"), {"ok": True})
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [7.0, 2.0])

    def test_gives_up_after_max_attempts(self):
        with mock.patch("requests.get", return_value=FakeJsonResponse(429)) as get, \
             mock.patch.object(api, "_sleep"):
            with self.assertRaises(api.CivitaiAPIError) as ctx:
                api._get("https://civitai.com/api/v1/models/1")
        self.assertEqual(get.call_count, api._MAX_ATTEMPTS)
        self.assertEqual(ctx.exception.status, 429)

    def test_404_is_not_retried(self):
        with mock.patch("requests.get", return_value=FakeJsonResponse(404)) as get, \
             mock.patch.object(api, "_sleep"):
            self.assertIsNone(api.fetch_version_by_hash("abc"))
        self.assertEqual(get.call_count, 1)

    def test_server_error_mentioning_404_is_not_mistaken_for_not_found(self):
        resp = FakeJsonResponse(400, text="upstream said 404")
        with mock.patch("requests.get", return_value=resp):
            with self.assertRaises(api.CivitaiAPIError):
                api.fetch_version_by_hash("abc")


class DownloadRobustnessTests(unittest.TestCase):
    def _task(self, tmp, sha=""):
        return downloader.DownloadTask(
            url="https://civitai.com/api/download/models/1",
            dest=Path(tmp) / "m.safetensors", filename="m.safetensors", sha256_expected=sha,
        )

    def test_html_page_is_never_saved_as_a_model(self):
        q = downloader.DownloadQueue()
        with tempfile.TemporaryDirectory() as tmp:
            task = self._task(tmp)
            page = FakeStream([b"<html>login</html>"], headers={"Content-Type": "text/html; charset=utf-8"})
            with mock.patch("requests.get", return_value=page) as get:
                q.download(task=task, download_preview=False)
            self.assertFalse(task.done)
            self.assertIn("API key", task.error)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_access_denied_is_reported_without_retry(self):
        q = downloader.DownloadQueue()
        with tempfile.TemporaryDirectory() as tmp:
            task = self._task(tmp)
            with mock.patch("requests.get", return_value=FakeStream([], status_code=401)) as get:
                q.download(task=task, download_preview=False)
            self.assertIn("401", task.error)
            self.assertEqual(get.call_count, 1)

    def test_interrupted_download_resumes_automatically(self):
        q = downloader.DownloadQueue()
        with tempfile.TemporaryDirectory() as tmp:
            task = self._task(tmp)
            first = FakeStream([b"hello", b"world"], headers={"Content-Length": "10"}, fail_after=1)
            second = FakeStream([b"world"], status_code=206,
                                headers={"Content-Range": "bytes 5-9/10", "Content-Length": "5"})
            seen_headers = []

            def fake_get(url, headers=None, **kw):
                seen_headers.append(dict(headers or {}))
                return [first, second][len(seen_headers) - 1]

            with mock.patch("requests.get", side_effect=fake_get), \
                 mock.patch.object(downloader, "_sleep") as sleep:
                q.download(task=task, download_preview=False)

            self.assertTrue(task.done)
            self.assertEqual(task.error, "")
            self.assertEqual(task.dest.read_bytes(), b"helloworld")
            self.assertEqual(seen_headers[1].get("Range"), "bytes=5-")
            sleep.assert_called_once()

    def test_truncated_download_is_not_accepted(self):
        q = downloader.DownloadQueue()
        with tempfile.TemporaryDirectory() as tmp:
            task = self._task(tmp)

            def short(url, **kw):
                return FakeStream([b"abc"], headers={"Content-Length": "10"})

            with mock.patch("requests.get", side_effect=short), \
                 mock.patch.object(downloader, "MAX_ATTEMPTS", 1), \
                 mock.patch.object(downloader, "_sleep"):
                q.download(task=task, download_preview=False)
            self.assertFalse(task.done)
            self.assertFalse(task.dest.exists())
            self.assertIn("early", task.error)

    def test_second_start_is_refused_while_running(self):
        q = downloader.DownloadQueue()
        release = threading.Event()
        with tempfile.TemporaryDirectory() as tmp:
            task = self._task(tmp)

            def slow(*a, **kw):
                release.wait(2)
                q.running = False

            with mock.patch.object(downloader.DownloadQueue, "download", side_effect=slow):
                self.assertTrue(q.start_async(task))
                self.assertFalse(q.start_async(task))
                release.set()


class JobStateTests(unittest.TestCase):
    def test_try_begin_is_exclusive(self):
        state = model_manager.ScanState()
        self.assertTrue(state.try_begin())
        self.assertFalse(state.try_begin())
        state.running = False
        self.assertTrue(state.try_begin())

    def test_log_is_capped_and_redacted(self):
        state = model_manager.ScanState()
        with mock.patch("ch_lib.utils.safe_print"):
            for i in range(model_manager._MAX_LOG_LINES + 50):
                state.append_log(f"line {i} token={SECRET}")
        self.assertEqual(len(state.log), model_manager._MAX_LOG_LINES)
        self.assertNotIn(SECRET, state.log[-1])


# ── Lot 3 : performance ──────────────────────────────────────────────────────

class CardIndexTests(unittest.TestCase):
    def setUp(self):
        card_index.invalidate()

    def test_lookup_requires_exact_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lora = make_root(tmp)
            for name in ("abc_v2", "flux", "flux.1-dev"):
                (lora / f"{name}.safetensors").write_bytes(b"x")
            with mock.patch("ch_lib.utils.get_sd_root", return_value=root):
                self.assertIsNone(card_index.find_model("abc"))
                self.assertEqual(card_index.find_model("flux.1-dev").name, "flux.1-dev.safetensors")
                self.assertEqual(card_index.find_model("FLUX").name, "flux.safetensors")
                self.assertEqual(card_index.find_model("flux.safetensors").name, "flux.safetensors")

    def test_subfolder_disambiguates_same_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lora = make_root(tmp)
            for sub in ("a", "b"):
                (lora / sub).mkdir()
                (lora / sub / "same.safetensors").write_bytes(b"x")
            with mock.patch("ch_lib.utils.get_sd_root", return_value=root):
                self.assertEqual(card_index.find_model("b\\same.safetensors").parent.name, "b")
                self.assertEqual(card_index.find_model("b/same").parent.name, "b")

    def test_folders_are_walked_once_for_many_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lora = make_root(tmp)
            (lora / "one.safetensors").write_bytes(b"x")
            with mock.patch("ch_lib.utils.get_sd_root", return_value=root), \
                 mock.patch("ch_lib.utils.iter_model_files", wraps=utils.iter_model_files) as walk:
                for _ in range(50):
                    sdfch_main.get_card_info_data("one")
                    sdfch_main.get_card_info_data("missing")
            self.assertEqual(walk.call_count, 1)

    def test_card_info_appears_after_scan_without_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lora = make_root(tmp)
            (lora / "late.safetensors").write_bytes(b"x")
            with mock.patch("ch_lib.utils.get_sd_root", return_value=root):
                self.assertEqual(sdfch_main.get_card_info_data("late")["civitai_url"], "")
                (lora / "late.civitai.info").write_text('{"id": 2, "modelId": 1}', encoding="utf-8")
                self.assertIn("models/1?modelVersionId=2",
                              sdfch_main.get_card_info_data("late")["civitai_url"])


if __name__ == "__main__":
    unittest.main()
