# Predicting Protein Function

Using Graph Neural Networks to predict whether a protien is an enzymes or non-enzymes using pytorch_geometric and the [TU Dortmund University Proteins Benchmark Dataset](https://chrsmrrs.github.io/datasets/docs/home/)

## Dataset

The PROTEINS dataset is a dataset of proteins classified as enzymes or non-enzymes. Nodes in the graph represent the amino acids and nodes are connected if they are less than 6 Angstroms apart. See [Borgwardt et al.](https://academic.oup.com/bioinformatics/article/21/suppl_1/i47/202991?login=true) for more information

## Installation Instruction

1. `git clone https://github.com/Ibrahimsuf/protein_function_prediction.git`
2. `cd protein_function_prediction`
3. create conda environment `conda create --name protien_prediction`
4. activate the environment `conda activate protien_prediction`
5. `conda install pip`
6. `pip install .`
7. Take a look at the Juypter Notebooks in the Examples folder
