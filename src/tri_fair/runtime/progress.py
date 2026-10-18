"""Common pilot watchdog: flat scores are allowed, endless unevaluated loops are not."""

from promptolution.utils.callbacks import BaseCallback
import logging


class EvaluationProgressCallback(BaseCallback):
    def __init__(self, max_idle_steps=3):
        self.limit = int(max_idle_steps)
        if self.limit < 1:
            raise ValueError("max_idle_steps must be positive")
        self.last = None
        self.idle = 0

    def on_step_end(self, optimizer):
        progress = (
            len(optimizer.task.pred_cache),
            int(optimizer.predictor.llm.get_token_count()["total_tokens"]),
        )
        self.idle = self.idle + 1 if self.last == progress else 0
        self.last = progress
        if self.idle >= self.limit:
            optimizer.runtime_stop_reason = "no_new_downstream_evaluations"
            logging.getLogger(__name__).warning(
                "Stopping after %d consecutive steps without new downstream evaluations",
                self.idle,
            )
            return False
        return True
