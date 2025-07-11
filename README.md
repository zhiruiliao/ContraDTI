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

## Citation
Our article has been published in Artificial Intelligence in Medicine.

If you find this study is useful, please cite our article.

>@article{ContraDTI2025Liao,
>
>title = {ContraDTI: Improved drug–target interaction prediction via multi-view contrastive learning},
>
>journal = {Artificial Intelligence in Medicine},
>
>volume = {168},
>
>pages = {103195},
>
>year = {2025},
>
>issn = {0933-3657},
>
>doi = {https://doi.org/10.1016/j.artmed.2025.103195},
>
>url = {https://www.sciencedirect.com/science/article/pii/S0933365725001307},

>author = {Zhirui Liao and Lei Xie and Shanfeng Zhu}
>
>}
## Declaration
It is free for non-commercial use. For commercial use, please contact with Zhirui Liao and Prof. Shanfeng Zhu (zhusf@fudan.edu.cn).

