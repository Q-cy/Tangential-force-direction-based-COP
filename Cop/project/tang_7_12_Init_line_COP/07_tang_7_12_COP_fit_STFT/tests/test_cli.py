from __future__ import annotations

import csv
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tangential import cli
from tangential.tools.training import TrainingResult


class CliTests(unittest.TestCase):
    def test_help_and_version(self):
        help_result = subprocess.run(
            [sys.executable, "-m", "tangential.cli", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("example", help_result.stdout)
        self.assertNotIn("calconsistence", help_result.stdout)
        self.assertNotIn("consistence", help_result.stdout.lower())
        version_result = subprocess.run(
            [sys.executable, "-m", "tangential.cli", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(version_result.returncode, 0)
        self.assertEqual(version_result.stdout.strip(), "0.6.0")

    def test_cli_module_import_isolated_from_optional_libraries(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import tangential.cli; "
                "assert 'pyqtgraph' not in sys.modules; "
                "assert 'matplotlib' not in sys.modules",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_app_arguments_map_to_config(self):
        observed = {}

        def fake_handler(args):
            observed.update(vars(args))
            return 0

        with mock.patch.object(cli, "_handle_app", fake_handler):
            self.assertEqual(
                cli.main([
                    "app", "--pressure-port", "p", "--force-port", "f",
                    "--save-dir", "out", "--model", "m",
                    "--max-time-diff-ms", "12.5",
                ]),
                0,
            )
        self.assertEqual(observed["pressure_port"], "p")
        self.assertEqual(observed["force_port"], "f")
        self.assertEqual(observed["save_dir"], "out")
        self.assertEqual(observed["max_time_diff_ms"], 12.5)

    def test_cli_has_five_commands_and_no_maintainer_options(self):
        parser = cli._build_parser()
        commands = parser._subparsers._group_actions[0].choices
        self.assertEqual(set(commands), {"example", "app", "dual", "plot", "fit"})

        rejected = (
            ["example", "--disable-consistence"],
            ["app", "--consistence-coefficients", "coefficients.npz"],
            [
                "dual", "--port-a", "pa", "--port-b", "pb",
                "--consistence-coefficients-a", "a.npz",
            ],
        )
        for arguments in rejected:
            with self.subTest(arguments=arguments), mock.patch(
                "sys.stderr", new_callable=io.StringIO
            ):
                with self.assertRaises(SystemExit) as raised:
                    cli.main(arguments)
            self.assertEqual(raised.exception.code, 2)

    def test_dual_arguments_are_exposed_and_forwarded_to_example(self):
        """统一 dual 子命令复用双路示例入口，不复制配置/采集逻辑。"""
        observed = {}

        def fake_dual(args):
            observed.update(vars(args))
            return 0

        with mock.patch(
            "tangential.examples.dual_sensor.run_from_namespace",
            side_effect=fake_dual,
        ):
            self.assertEqual(
                cli.main([
                    "dual", "--port-a", "pa", "--port-b", "pb",
                    "--force-port-a", "fa", "--save-dir-a", "out-a",
                    "--save-dir-b", "out-b", "--model", "model.bin",
                ]),
                0,
            )
        self.assertEqual(observed["port_a"], "pa")
        self.assertEqual(observed["port_b"], "pb")
        self.assertEqual(observed["force_port_a"], "fa")
        self.assertIsNone(observed["force_port_b"])
        self.assertEqual(observed["save_dir_a"], "out-a")
        self.assertEqual(observed["save_dir_b"], "out-b")

    def test_app_uses_acquisition_loop_runner(self):
        with mock.patch("tangential.runtime.session.FullApplicationRunner") as runner:
            self.assertEqual(
                cli.main([
                    "app", "--pressure-port", "p", "--force-port", "f",
                    "--save-dir", "out", "--max-time-diff-ms", "12.5",
                ]),
                0,
            )
        runner.assert_called_once()
        target = runner.call_args.args[0]
        config = runner.call_args.kwargs["config"]
        self.assertEqual(target.__name__, "acquisition_loop")
        self.assertEqual(config.pressure_port, "p")
        self.assertEqual(config.force_port, "f")
        self.assertEqual(config.max_time_diff_s, 0.0125)
        runner.return_value.run.assert_called_once_with()

    def test_plot_generates_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "sample.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["rel_ms", "adc_sum"])
                writer.writerow([0, 10])
                writer.writerow([1, 11])
            output = root / "plot.png"
            self.assertEqual(
                cli.main([
                    "plot", "--dir", str(root), "--files", "sample.csv",
                    "--columns", "adc_sum", "--save", str(output),
                ]),
                0,
            )
            self.assertTrue(output.is_file())

    def test_fit_arguments_map_and_default_does_not_write_back(self):
        observed = {}
        fake_result = TrainingResult(
            model_path=Path("model.bin"), plot_path=None, fit_results=[],
            sample_counts={}, written_path=None,
        )

        def fake_train(config):
            observed.update(vars(config))
            return fake_result

        with mock.patch("tangential.tools.training.train_model", side_effect=fake_train):
            self.assertEqual(
                cli.main([
                    "fit", "--xy-csv", "xy.csv", "--z-csv", "z.csv",
                    "--dim", "2", "--poly-order", "2", "--no-valid-only",
                    "--no-split-sign", "--no-one-on-one",
                ]),
                0,
            )
        self.assertEqual(observed["xy_csv"], "xy.csv")
        self.assertEqual(observed["dim"], 2)
        self.assertFalse(observed["valid_only"])
        self.assertFalse(observed["split_sign"])
        self.assertFalse(observed["one_on_one"])
        self.assertIsNone(observed["write_back"])

    def test_argparse_errors_return_two(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["fit", "--xy-csv", "only-one.csv"])
        self.assertEqual(raised.exception.code, 2)

    def test_runtime_exception_returns_one_and_writes_stderr(self):
        stderr = io.StringIO()
        with mock.patch("tangential.tools.training.train_model", side_effect=RuntimeError("bad data")):
            with mock.patch("sys.stderr", stderr):
                result = cli.main([
                    "fit", "--xy-csv", "xy.csv", "--z-csv", "z.csv",
                ])
        self.assertEqual(result, 1)
        self.assertIn("错误: bad data", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
