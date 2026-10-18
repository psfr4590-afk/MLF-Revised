import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def test_launcher_targets_single_browser_command_center(self):
        source = (ROOT / "launch.py").read_text(encoding="utf-8")
        self.assertIn('COMMAND_CENTER = ROOT / "run_command_center.py"', source)
        self.assertIn('subprocess.call([sys.executable, str(COMMAND_CENTER)], cwd=ROOT)', source)
        self.assertNotIn('UI_ENTRYPOINT = ROOT / "ui" / "app.py"', source)
        self.assertNotIn('tkinter', source)


if __name__ == "__main__":
    unittest.main()
