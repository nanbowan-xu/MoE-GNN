# EMAIL
EMAIL: Equivariant Message Attention Interaction Learning on Graphs for Drug Discovery

# Dependency
The details can be seen ``` email.yaml```
# Datasets
The PDBbind dataset can be downloaded [here](http://www.pdbbind.org.cn/).
The CSAR-HiQ dataset can be downloaded [here](http://www.csardock.org/).
The molecule dataset can be downloaded [here](https://moleculenet.org/datasets-1).
You may need to use the UCSF Chimera tool to convert the PDB-format files into MOL2-format files for feature extraction at first.
# How to run
The downloaded dataset should be preprocessed to obtain features and spatial coordinates:


``` python preprocess_pdbbind.py --data_path_core YOUR_DATASET_PATH --data_path_refined YOUR_DATASET_PATH --dataset_name pdbbind2016 --output_path YOUR_OUTPUT_PATH --cutoff 5 ```


You can also try to use the processed data from [this link](https://www.dropbox.com/sh/68vc7j5cvqo4p39/AAB_96TpzJWXw6N0zxHdsppEa).
Before training the model, please put the downloaded files into the directory (./data/).


To train the model, you can run this command:

```python train.py --cuda YOUR_DEVICE --model_dir MODEL_PATH_TO_SAVE --dataset pdbbind2016 --cut_dist 5 ```
The molecular properties prediction task can be seen in the file [molecular properties prediction](https://github.com/nanbowan-xu/email/tree/main/molecular%20properties%20prediction)
# Contact
If you have any comments or questions, feel free to contact wanxu2021@email.szu.edu.cn
