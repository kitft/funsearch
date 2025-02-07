import pytest
import asyncio
from funsearch import async_agents, config, programs_database, sandbox, models
import tempfile
import os
from funsearch.__main__ import runAsync

def test_mock_model_run():
    # Test the mock_test CLI command
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_test(
            duration=2,
            samplers=1, 
            evaluators=1,
            output_path=temp_dir
        )
        
        # Verify outputs
        scores_dir = os.path.join(temp_dir, "data/scores")
        assert os.path.exists(scores_dir), "Score directory should exist"
        
        score_files = [f for f in os.listdir(scores_dir) if f.endswith('.csv')]
        assert len(score_files) > 0, "Should have created at least one score file"