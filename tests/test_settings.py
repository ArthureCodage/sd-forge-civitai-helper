import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch_lib import settings


class FakeOpts:
    def __init__(self):
        self.sd_forge_civitai_helper_api_key = "saved-token"
        self.added = []

    def add_option(self, key, option):
        self.added.append((key, option))


class FakeShared:
    def __init__(self):
        self.opts = FakeOpts()

    class OptionInfo:
        def __init__(self, default, label, section=None):
            self.default = default
            self.label = label
            self.section = section


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.modules = types.ModuleType("modules")
        self.fake_shared = FakeShared()
        self.modules.shared = self.fake_shared
        self.patcher = mock.patch.dict(sys.modules, {"modules": self.modules, "modules.shared": self.fake_shared})
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_resolve_api_key_prefers_explicit_value(self):
        self.assertEqual(settings.resolve_api_key(" typed-token "), "typed-token")

    def test_resolve_api_key_falls_back_to_saved_webui_setting(self):
        self.assertEqual(settings.resolve_api_key(""), "saved-token")

    def test_resolve_api_key_falls_back_to_environment_without_webui(self):
        self.patcher.stop()
        with mock.patch.dict(os.environ, {settings.ENV_API_KEY: "env-token"}):
            self.assertEqual(settings.resolve_api_key(""), "env-token")
        self.patcher.start()

    def test_register_options_adds_webui_setting(self):
        settings.register_options()
        self.assertEqual(len(self.fake_shared.opts.added), 1)
        key, option = self.fake_shared.opts.added[0]
        self.assertEqual(key, settings.API_KEY_OPTION)
        self.assertEqual(option.section, ("sd_forge_civitai_helper", "CivitAI Helper"))


if __name__ == "__main__":
    unittest.main()
