import os
from pathlib import Path
import subprocess
import tempfile
import unittest

CODE = Path(__file__).resolve().parents[2]


class LauncherTests(unittest.TestCase):
    def environment(self):
        env = os.environ.copy()
        env.update(DRY_RUN="1", NGPUS="1", EVAL_NGPUS="1",
                   BAGEL_MODEL_DIR="/example/base", STAGE1_CKPT="/example/stage1",
                   SSL_MODEL_DIR="/example/stage2", DATA_ROOT="/example/data")
        for key in ("OPENAI_API_KEY",):
            env.pop(key, None)
        return env

    def test_training_config_resolves_from_callers_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.yaml"
            path.write_text((CODE / "configs/train/stage1_pretext.yaml").read_text())
            result = subprocess.run(["bash", str(CODE / "scripts/train.sh"), "local.yaml"],
                                    cwd=directory, env=self.environment(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("training was not started", result.stdout)

    def test_training_does_not_default_to_eight_gpus(self):
        env = self.environment()
        env.pop("NGPUS")
        result = subprocess.run(["bash", str(CODE / "scripts/train.sh"),
                                 str(CODE / "configs/train/stage1_pretext.yaml")],
                                env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Set NGPUS", result.stderr)

    def test_empty_path_variable_is_rejected(self):
        env = self.environment()
        env["DATA_ROOT"] = ""
        result = subprocess.run(["bash", str(CODE / "scripts/train.sh"),
                                 str(CODE / "configs/train/stage1_pretext.yaml")],
                                env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DATA_ROOT", result.stderr)

    def test_eval_dry_run_needs_no_model_or_judge(self):
        result = subprocess.run(["bash", str(CODE / "eval/run_eval.sh"),
                                 str(CODE / "configs/eval/ssl.yaml")],
                                env=self.environment(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1 allocated GPU(s)", result.stdout)
        self.assertIn("No checkpoint merge", result.stdout)


if __name__ == "__main__":
    unittest.main()
