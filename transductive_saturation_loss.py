import warnings
from loss import TransductivePortfolioLoss
from test_validation import run_validation


warnings.warn(
    "transductive_saturation_loss.py is deprecated. "
    "Please import TransductivePortfolioLoss from 'loss' module instead: "
    "from loss import TransductivePortfolioLoss",
    DeprecationWarning,
    stacklevel=2
)


if __name__ == "__main__":
    run_validation()
