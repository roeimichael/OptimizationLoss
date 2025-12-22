# Neural Network Experiments

Clean Python script environment for running neural network experiments with transductive loss.

## Structure

```
nn_experiments/
├── config.py              - All configuration parameters
├── data_loader.py         - Data loading and preprocessing
├── constraints.py         - Constraint computation functions
├── transductive_loss.py   - Your multiclass transductive loss
├── model.py               - Neural network architecture
├── dataset.py             - PyTorch Dataset class
├── trainer.py             - Training and evaluation functions
├── run_experiments.py     - Main experiment script
└── results/               - Output directory (created automatically)
```

## Usage

Run all experiments:
```bash
cd nn_experiments
python run_experiments.py
```

## Configuration

Edit `config.py` to change:
- Constraint configurations
- NN hyperparameters (lambda values, hidden dims)
- Training parameters (epochs, batch size, learning rate)

## Output

Results are saved to:
- `results/students__train__nn_config{1,2,3}__transductive.csv` - Detailed results per fold
- `results/nn_results.json` - Aggregated results with mean and std

## Device

Automatically uses CUDA if available, otherwise CPU.
