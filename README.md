# ContraDTI
Implementation of ContraDTI via TensorFlow

## Requirements:
python 3.7

tensorflow-gpu >= 2.3.0

rdkit >= 2020.03.2.0

pandas >= 1.0.3

numpy >= 1.19.2


## Data
Single-target and multi-target datasets can be found at TDC Datasets' homepage: https://tdcommons.ai/overview/

Run `data/single_target_data_preprocess.py --input [input_csv_file]` for single-target data preprocessing. 

Run `data/multi_target_data_preprocess.py --input [input_csv_file]` for multi-target data preprocessing. 

You can also preprocess your customized data with a format like `data/examples.csv`. 




## Training
Run `training.py` to train a model with default hyper-parameters.

More hyper-parameter setting can be found and changed in code file `training.py`.

