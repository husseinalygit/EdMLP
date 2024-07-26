# EdMLP: Boosted Multilayer Feedforward Neural Network with Multiple Output Layers

This repository contains the code for the EdMLP architecture proposed in the paper "Boosted Multilayer Feedforward Neural Network with Multiple Output Layers."


## Installation
To use this repository, you'll need to install the following dependencies:

```
matplotlib
numpy
pandas
scikit-learn
scipy
torch
smac
easydict
configspace
```

## Dataset
You can get the dataset from the following sources:

- Download the data from http://www.bioinf.jku.at/people/klambauer/data_py.zip
- Check the GitHub repository at https://github.com/bioinf-jku/SNNs/tree/master/UCI

After downloading, unzip the data and move it under the `DLoader/UCIdata` folder. The folder should have the following structure:

```
UCIdata/
    - abalone
    - acute-inflammation
    - ...
```

## Hyperparameter Tuning Parameters

**Box Models (Box_HBO):**
- `--dataset`: Name of the UCI dataset(s) to use (can be multiple)
- `--model`: Model to use (choices: `EdMLP`, `EdSNN`, `EdMLP_DLHO`, `EdSNN_DLHO`, `MLP`, `SNN`, `EdMLP_Boost`, `EdSNN_Boost`, `EdMLP_DLHO_Boost`, `EdSNN_DLHO_Boost`, `EdRVFL`)
- `--device`: Device to use ('cpu' or 'cuda')
- `--n_trials`: Number of trials to run
- `--tunning_reps`: Number of repetitions for Bayesian optimization
- `--eval_reps`: Number of repetitions for evaluation
- `--random_state`: Random state
- `--n_jobs`: Number of jobs to run in parallel
- `--run_id`: Run ID (set a value if you want to run multiple optimization rounds with the same ID, otherwise leave blank)

**Layer-Wise Tuning Models (LW_HPO):**
- `--dataset`: Name of the UCI dataset(s) to use (can be multiple)
- `--model`: Model to use (choices: `LEdMLP_DLHO_Boost`, `BLEdMLP_DLHO_Boost`)
- `--device`: Device to use ('cpu' or 'cuda')
- `--n_trials`: Number of trials to run
- `--tunning_reps`: Number of repetitions for Bayesian optimization
- `--eval_reps`: Number of repetitions for evaluation
- `--random_state`: Random state
- `--n_jobs`: Number of jobs to run in parallel
- `--layer_patience`: Number of layers to wait before stopping layer-wise optimization
- `--epochs`: Number of epochs to train
- `--max_layers`: Maximum number of layers to train
- `--run_id`: Run ID (set a value if you want to run multiple optimization rounds with the same ID, otherwise leave blank)

The results of the hyperparameter tuning will be stored in a new folder named `results` in the following format:
- `"results/results_r{run_id}_s{session_id}.csv"`
- The `run_id` is the hypertuning operation ID, which can be used to retrieve the model later after the best hyperparameters are saved.
- The `session_id` is used to record the timestamp at which the session is run, which is important when you want to run the same optimization operation with the same `run_id` on multiple sessions.

## Evaluation
You can use the `evaluators.py` script to run the evaluation of the model:

```
python evaluators.py --model LEdMLP_DLHO_Boost --dataset abalone --epochs 1 --eval_reps 5 --run_id 26_07_2024T_05_59
```

The `evaluators.py` script takes the following options:
- `--run_id`: Run ID of the hyperparameter tuning (leave as "auto" to use the most recent ID)
- `--device`: Device to run the model on ('cpu' or 'cuda')
- `--model`: Model to use (can be multiple)
- `--dataset`: Dataset to use (can be multiple)
- `--random_seed`: Random seed for the evaluation
- `--n_jobs`: Number of parallel jobs to run
- `--epochs`: Training epochs (used for layer-wise algorithms only, otherwise ignored)
- `--eval_reps`: Number of repetitions for evaluation

## Citation
If you use this code in your research, please cite the following paper:

```bibtex
@article{aly2024boosted,
  title={Boosted multilayer feedforward neural network with multiple output layers},
  author={Aly, Hussein and Al-Ali, Abdulaziz K and Suganthan, Ponnuthurai Nagaratnam},
  journal={Pattern Recognition},
  pages={110740},
  year={2024},
  publisher={Elsevier}
}
```
