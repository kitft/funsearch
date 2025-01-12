from dataclasses import dataclass
import datetime
from typing import Optional
import json
from pathlib import Path
import atexit

@dataclass
class UsageStats:
    id: str
    model: str
    provider: str
    tokens_prompt: Optional[int] = None
    tokens_completion: Optional[int] = None
    total_tokens: Optional[int] = None
    total_cost: Optional[float] = None
    generation_time: Optional[float] = None
    instance_id: Optional[str] = None
    timestamp: str = datetime.datetime.now(datetime.UTC).isoformat()
    island_id: Optional[str] = None
    island_version: Optional[str] = None
    version_generated: Optional[str] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    eval_state: Optional[str] = None
    sandbox_current_call_count: Optional[int] = None
    prompt_count: Optional[int] = None
    sampler_id: Optional[int] = None
    scores_per_test: Optional[list] = None
    std_err: Optional[str] = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


class UsageLogger:
    def __init__(self,id = 0,log_dir: str = "./data/usage"):
        """
        Args:
            sampler_id: ID of the LLMModel instance
            log_dir: Base directory for all usage logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.id = id
        self.current_date = datetime.datetime.now(datetime.UTC).date()
        self.buffer = []
        #self.model_name = model_name
        self.total_requests = 0
        #self.total_tokens_in_prompts = 0
        #self.total_tokens_in_completions = 0
        atexit.register(self.flush)

    # def get_total_data(self):
    #     return {
    #         "total_requests": self.total_requests,
    #         "total_tokens_in_prompts": self.total_tokens_in_prompts,
    #         "total_tokens_in_completions": self.total_tokens_in_completions,
    #     }
        
    @property
    def log_file(self) -> Path:
        """Get the current day's log file path, including ID"""
        return self.log_dir / "usage" / f"usage_stats_{self.id}.jsonl"
    
    def flush(self):
        """Write buffered logs to file"""
        if self.buffer:
            # Ensure parent directories exist
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open('a') as f:
                for entry in self.buffer:
                    json.dump(entry, f)
                    f.write('\n')
            self.buffer = []
    def log_usage(self, stats: UsageStats):
        """Buffer usage statistics"""
        # Check if we need to roll over to a new day's file
        # current_date = datetime.utcnow().date()
        # if current_date != self.current_date:
        #     self.flush()  # Flush any remaining entries from previous day
        #     self.current_date = current_date
        provider = stats.provider
        self.total_requests += 1
        log_entry = stats.to_dict()
        log_entry.pop('prompt')
        log_entry.pop('response')

        self.buffer.append(log_entry)
        # if stats.tokens_prompt:
        #     self.total_tokens_in_prompts += stats.tokens_prompt
        # if stats.tokens_completion:
        #     self.total_tokens_in_completions += stats.tokens_completion
        
        # # Flush if buffer gets too large
        if len(self.buffer) >= 100:
            self.flush()

def get_usage_summary(self, log_dir, provider=None, model_name=None):
    """Get usage summary for specified provider"""
    stats = []
    log_files = sorted(self.log_dir.glob("usage/usage_stats_*.jsonl"))
    
    for log_file in log_files:
        with log_file.open('r') as f:
            for line in f:
                entry = json.loads(line)
                if provider and entry.get('provider') != provider:
                    continue
                if model_name and entry.get('model') != model_name:
                    continue
                stats.append(entry)
    
    return stats
