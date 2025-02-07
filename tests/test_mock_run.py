import pytest
import os
import tempfile
from click.testing import CliRunner
from funsearch.__main__ import mock_test

def test_mock_run():
    """Test the mock_test CLI command using cap_set_spec."""
    # Skip if cap_set_spec.py doesn't exist
    if not os.path.exists("examples/cap_set_spec.py"):
        pytest.skip("cap_set_spec.py not found in examples directory")
    
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create required directories
        os.makedirs(os.path.join(temp_dir, "data/scores"), exist_ok=True)
        
        result = runner.invoke(mock_test, [
            '--duration', '5',
            '--samplers', '1',
            '--evaluators', '1',
            '--output_path', temp_dir
        ])
        
        # Print output for debugging
        print(f"Mock test output: {result.output}")
        
        assert result.exit_code == 0, f"Mock test failed with error: {result.output}"
        
        # Check if score files were created
        scores_dir = os.path.join(temp_dir, "cap_set_spec")
        assert os.path.exists(scores_dir), "Test directory should exist"
        
        # Should have at least one timestamp directory
        timestamp_dirs = [d for d in os.listdir(scores_dir) if os.path.isdir(os.path.join(scores_dir, d))]
        assert len(timestamp_dirs) > 0, "Should have created a timestamp directory"
        
        # Check for scores file
        data_scores = os.path.join(temp_dir, "data/scores")
        assert os.path.exists(data_scores), "Scores directory should exist"
        score_files = [f for f in os.listdir(data_scores) if f.endswith('.csv')]
        assert len(score_files) > 0, "Should have created at least one score file"