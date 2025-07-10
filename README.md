# SynBridge: Bridging Reaction States via Discrete Flow for Bidirectional Reaction Prediction
<p align="center">
    <img src="temp/schematic.png" width="800" class="center" alt="schematic"/>
    <br/>
</p>

## Installation

#### Create the conda environment and activate it.
```bash
conda create -n synbridge python==3.10
conda activate synbridge
```
#### Install basic packages
```bash
# install requirements
pip install -r requirements.txt
```

## Dataset 
We provide the processed dataset of `USPTO-50K` and `USPTO-MIT` through [hugging face](https://huggingface.co/datasets/Delcher/synbridge-uspto).
The raw datasets are `.txt` and `.csv`  file, while the parsed ones are `.pickle` file.

Please download `data.zip` and unzip it, leading to the data file directoory as 
```
- data
    - uspto50k
        raw_test.csv
        raw_train.csv
        raw_val.csv
        test_data.pickle
        test.txt
        train_data.pickle
        ...
    - usptomit
        raw_test.csv
        raw_train.csv
        raw_val.csv
        test_data.pickle
        test.txt
        train_data.pickle
        ...
```
If you want the raw datasets for preprocessing, please refer to 
`./preprocess/preprocess_txt.py`
## Training and Generating
> **Note** This branch is mainly based on single separate task reproduction. If you want to run SynBridge on separate tasks, please turn to [Master Branch](https://github.com/EDAPINENUT/synbridge/).
### Training from scratch
Run the following command for training the multi-task version of synbridge:

```bash
python train.py --config-name={config_name} \
train.devices={available_world_size}
```
where `config_name` should be in `difm_forward_uspto50k`, `difm_forward_usptomit`, `difm_retro_uspto50k` or `difm_retro_usptomit`. 


#### After training, you can choose an epoch for generating the peptides through:

```
python test.py --ckpt_path {/path/to/your/checkpoint.ckpt} \
--config_path {/path/to/your/config.yaml} \
--save_samples \
--eval_mode {eval_mode}
```
where `eval_mode` should be in `forward` or `retro`.

### Testing from pretrained checkpoints
We provide pretrained models on [Hugging Face](https://huggingface.co/Delcher/synbridge/tree/main). After downloading, store them in `./ckpts/{dataset}_{task}/`, which results in the following ckpts folder structure:
```
- ckpts
    - uspto50k_forward
        epoch=313-step=98000.ckpt
        config.yaml
        
    - uspto50k_retro
        epoch=341-step=107000.ckpt
        config.yaml
    
    - usptomit_forward
        ...
    
    - usptomit_retro
        ...
```
Use the following command to test on the test set:
```bash
# test the multitask model on uspto50k on forward task
python test.py --ckpt_path ./ckpts/{dataset}_{task}/{ckpt_name} \
--config_path ./ckpts/{dataset}_{task}/config.yaml \
--eval_mode {task} \
--save_samples 
```

For example,
```bash
# test the multitask model on uspto50k on forward task
python test.py --ckpt_path ./ckpts/uspto50k_forward/epoch=313-step=98000.ckpt \
--config_path ./ckpts/uspto50k_forward/config.yaml \
--eval_mode forward \
--save_samples 
```

## Citation
If our paper or the code in the repository is helpful to you, please cite the following:
```
IN PROCESS
```

