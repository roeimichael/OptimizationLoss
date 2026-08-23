"""class_balanced: fine-tune the shared CE warmup with class-balanced CE,
then Shifman-LP clip."""

from src.methodologies.imbalanced_common import run_imbalanced
from src.pipeline.contracts import TrainInputs, TrainOutputs


def train(inputs: TrainInputs) -> TrainOutputs:
    return run_imbalanced(inputs, "class_balanced")
