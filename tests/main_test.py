import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from funsearch.__main__ import main, parse_input

ROOT_DIR = Path(__file__).parent.parent

runner = CliRunner()


class TestMain(unittest.TestCase):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    self.default_args = ["--output_path", self.temp_dir, "--samplers", "1", "--iterations", "1",
                         str(ROOT_DIR / "examples" / "cap_set_spec.py"), "5"]

  def tearDown(self):
    shutil.rmtree(self.temp_dir)

  def test_main(self):
    result = runner.invoke(main, [])
    assert result.exit_code != 0
    assert 'Usage:' in result.output
    with patch('funsearch.core.run', return_value=None) as mock_run:
      result = runner.invoke(main, self.default_args)
      assert result.exit_code == 0
      assert mock_run.call_count == 1

  def test_main_sample(self):
    with patch('funsearch.sampler.LLM._draw_sample', return_value="return 0.5") as mock_run:
      result = runner.invoke(main, self.default_args)
      assert result.exit_code == 0
      assert mock_run.call_count == 2, "There should be 2 sampler per sampler by default"


def test_parse_input():
  assert parse_input("1") == [1]
  assert parse_input("1,2,3") == [1, 2, 3]
  assert parse_input(str(ROOT_DIR / "examples" / "cap_set_input_data.json")) == [8, 9, 10]
