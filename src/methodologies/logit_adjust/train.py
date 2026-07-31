"""logit_adjust: fine-tune the shared CE warmup with the logit-adjusted softmax
loss, then Shifman-LP clip."""

from src.methodologies.imbalanced_common import run_imbalanced
from src.pipeline.contracts import TrainInputs, TrainOutputs


def train(inputs: TrainInputs) -> TrainOutputs:
    return run_imbalanced(inputs, "logit_adjust")
