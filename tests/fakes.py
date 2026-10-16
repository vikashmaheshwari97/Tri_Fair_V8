"""Small deterministic fake models; never download data or call an LLM service."""

from types import SimpleNamespace
import hashlib
import pandas as pd
from tri_fair.data.task import FairnessTask
from tri_fair.fairness.statistical_metrics import compute_bios_statistical_tpr_gap


class FakeTokenizer:
    name_or_path = "fake-tokenizer"

    def encode(self, text):
        return str(text).split()


class FakeLLM:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.input_token_count = 0
        self.output_token_count = 0
        self.sampling_params = SimpleNamespace(
            max_tokens=16, temperature=0.0, top_p=1.0, seed=42
        )

    def get_token_count(self):
        return {
            "input_tokens": self.input_token_count,
            "output_tokens": self.output_token_count,
            "total_tokens": self.input_token_count + self.output_token_count,
        }

    def get_response(self, prompts, system_prompts=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        outputs = []
        for prompt in prompts:
            text = (
                f"<prompt>Use explicit professional evidence and ignore gender. Revision "
                f"{self.output_token_count}. Return only one final answer label.</prompt>"
            )
            self.input_token_count += len(self.tokenizer.encode(prompt))
            self.output_token_count += len(self.tokenizer.encode(text))
            outputs.append(text)
        return outputs


class FakePredictor:
    begin_marker = "<final_answer>"
    end_marker = "</final_answer>"
    extraction_description = "tagged lower-case class"

    def __init__(self):
        self.llm = FakeLLM()
        self.classes = ["nurse", "teacher"]

    def predict(self, prompts, xs, system_prompts=None):
        predictions, sequences = [], []
        for prompt, text in zip(prompts, xs):
            # Toy input intentionally names its synthetic profession. This fake
            # verifies control flow and accounting, not real predictive quality.
            label = "nurse" if "nurse" in text else "teacher"
            digest = hashlib.sha256((prompt + text).encode()).digest()[0]
            if digest % 7 == 0:
                label = "teacher" if label == "nurse" else "nurse"
            raw = f"<final_answer>{label}</final_answer>"
            self.llm.input_token_count += len(
                self.llm.tokenizer.encode(prompt + "\n" + text)
            )
            self.llm.output_token_count += len(self.llm.tokenizer.encode(raw))
            predictions.append(label)
            sequences.append(text + "\n" + raw)
        return predictions, sequences


def task_and_pool(n_blocks=8, eval_strategy="sequential_block"):
    rows, blocks = [], []
    for b in range(n_blocks):
        ids = []
        for label in ["nurse", "teacher"]:
            for gender in [0, 1]:
                for repetition in range(2):
                    key = f"{b}-{label}-{gender}-{repetition}"
                    ids.append(key)
                    rows.append(
                        {
                            "_manifest_id": key,
                            "input": f"{label} synthetic person {key}",
                            "target": label,
                            "gender": gender,
                            "profession_id": int(label == "teacher"),
                        }
                    )
        blocks.append(ids)
    frame = pd.DataFrame(rows)
    task = FairnessTask(
        df=frame,
        task_description="Choose nurse or teacher.",
        classes=["nurse", "teacher"],
        quality_metric="macro_f1",
        fairness_metric=compute_bios_statistical_tpr_gap,
        fairness_metric_name="bios_tpr_gap_statistical",
        protected_columns=["gender"],
        metadata_columns=["gender", "profession_id"],
        min_group_count=2,
        fairness_kwargs={
            "min_valid_professions": 2,
            "max_rate_ci_width": 1.0,
            "min_macro_recall": 0.0,
            "normalization_cost_upper_bound": 120.0,
        },
        block_ids=blocks,
        eval_strategy=eval_strategy,
        seed=42,
    )
    # Separate synthetic few-shot IDs, even though text distribution is similar.
    pool = frame.iloc[:16].copy()
    pool["_manifest_id"] = "fs-" + pool["_manifest_id"]
    return task, pool


def optimizer(cls, **kwargs):
    task, pool = task_and_pool()
    return cls(
        predictor=FakePredictor(),
        task=task,
        meta_llm=FakeLLM(),
        initial_prompts=[
            f"Classify occupation using duties. Seed {i}." for i in range(6)
        ],
        df_few_shots=pool,
        create_fs_reasoning=False,
        upper_shots=2,
        crossovers_per_iter=2,
        cost_per_input_token=0.11,
        cost_per_output_token=0.41,
        **kwargs,
    )
