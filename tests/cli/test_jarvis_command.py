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


if __name__ == "__main__":
    unittest.main()
