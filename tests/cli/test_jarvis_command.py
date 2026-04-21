"""Tests for the /jarvis slash command."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _import_cli():
    import hermes_cli.config as config_mod

    if not hasattr(config_mod, "save_env_value_secure"):
        config_mod.save_env_value_secure = lambda key, value: {
            "success": True,
            "stored_as": key,
            "validated": False,
        }

    import cli as cli_mod

    return cli_mod


class TestJarvisRegistered(unittest.TestCase):
    def test_jarvis_is_in_command_registry(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("jarvis")
        self.assertIsNotNone(cmd, "/jarvis must be registered in COMMAND_REGISTRY")
        self.assertEqual(cmd.name, "jarvis")
        self.assertTrue(cmd.cli_only, "/jarvis is interactive only; gateways cannot deliver claps")


class TestHandleJarvisCommand(unittest.TestCase):
    """Unit tests for HermesCLI._handle_jarvis_command orchestration."""

    def _make_stub(self, voice_mode=False):
        import queue
        import threading

        stub = SimpleNamespace(
            _voice_mode=voice_mode,
            _voice_tts=False,
            _voice_lock=threading.Lock(),
            _pending_input=queue.Queue(),
            _enable_voice_mode=MagicMock(),
        )
        # When _enable_voice_mode is called, simulate it setting _voice_mode True
        def _fake_enable():
            stub._voice_mode = True
        stub._enable_voice_mode.side_effect = _fake_enable
        return stub

    def test_audio_unavailable_aborts_without_enabling_voice(self):
        cli_mod = _import_cli()
        stub = self._make_stub()

        with patch.object(cli_mod, "_cprint"), patch(
            "tools.voice_mode.detect_audio_environment",
            return_value={"available": False, "warnings": ["no microphone"], "notices": []},
        ):
            cli_mod.HermesCLI._handle_jarvis_command(stub, "/jarvis")

        stub._enable_voice_mode.assert_not_called()
        self.assertTrue(stub._pending_input.empty())

    def test_happy_path_enables_voice_tts_and_queues_briefing(self):
        cli_mod = _import_cli()
        stub = self._make_stub(voice_mode=False)

        with patch.object(cli_mod, "_cprint"), patch(
            "tools.voice_mode.detect_audio_environment",
            return_value={"available": True, "warnings": [], "notices": []},
        ), patch("tools.voice_mode.play_beep") as mock_beep, patch(
            "tools.clap_detector.ClapDetector"
        ) as MockDetector:
            MockDetector.return_value.listen.return_value = True
            cli_mod.HermesCLI._handle_jarvis_command(stub, "/jarvis")

        stub._enable_voice_mode.assert_called_once()
        MockDetector.return_value.listen.assert_called_once_with(timeout_seconds=30.0)
        self.assertTrue(stub._voice_tts, "TTS must be enabled so the briefing is spoken")
        self.assertGreaterEqual(mock_beep.call_count, 1)
        self.assertFalse(stub._pending_input.empty())
        queued = stub._pending_input.get_nowait()
        self.assertIn("weather", queued.lower())
        self.assertIn("할 일", queued)
        self.assertIn("3문장", queued)

    def test_clap_timeout_does_not_queue_briefing(self):
        cli_mod = _import_cli()
        stub = self._make_stub(voice_mode=False)

        with patch.object(cli_mod, "_cprint"), patch(
            "tools.voice_mode.detect_audio_environment",
            return_value={"available": True, "warnings": [], "notices": []},
        ), patch("tools.voice_mode.play_beep"), patch(
            "tools.clap_detector.ClapDetector"
        ) as MockDetector:
            MockDetector.return_value.listen.return_value = False  # timeout
            MockDetector.return_value.peak_rms = 500.0  # concrete value; handler formats this
            cli_mod.HermesCLI._handle_jarvis_command(stub, "/jarvis")

        self.assertTrue(stub._pending_input.empty(), "timeout must not queue a briefing prompt")

    def test_voice_mode_already_on_does_not_double_enable(self):
        cli_mod = _import_cli()
        stub = self._make_stub(voice_mode=True)  # already on

        with patch.object(cli_mod, "_cprint"), patch(
            "tools.voice_mode.detect_audio_environment",
            return_value={"available": True, "warnings": [], "notices": []},
        ), patch("tools.voice_mode.play_beep"), patch(
            "tools.clap_detector.ClapDetector"
        ) as MockDetector:
            MockDetector.return_value.listen.return_value = True
            cli_mod.HermesCLI._handle_jarvis_command(stub, "/jarvis")

        stub._enable_voice_mode.assert_not_called()
        self.assertTrue(stub._voice_tts)
        self.assertFalse(stub._pending_input.empty())


if __name__ == "__main__":
    unittest.main()
