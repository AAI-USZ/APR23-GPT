Each directory contains the source code for the adequate model.
The dataset and the preprocessed dataset is available in our previous work.


Example run:
1. python3 main.py --task predict --input_path [input_path] --experiment_dir [experiment_dir] --gpu 0
2. python3 main.py --task results --input_path [input_path] --experiment_dir [experiment_dir] --gpu 0


--input_path: path to APR_Data folder (this folder contains the dataset)
--experiment_dir = "paper_exp_dir" a folder in APR_Data (results and candidates are saved in this folder)

For additional parameters check argument_parser.py
