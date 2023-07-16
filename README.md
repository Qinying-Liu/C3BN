# C3BN
## Introduction
This repository is used for Unleashing the Potential of Adjacent Snippets for Weakly-supervised Temporal Action Localization. 
It can reproduce the main results on THUMOS14 with the baseline of BaS-Net. We choose this baseline for illustration due to its simplicity.

Accepted by ICME 2023, oral.

## Install & Requirements
We conduct experiments on the following environment: <br>
 * python == 3.9.13 <br>
 * pytorch == 1.10.1 <br>
 * CUDA == 11.3.1 <br>

Please refer to “requirements.txt” for detailed information.
## Data Preparation
1. Download the preprocessed THUMOS14 features and annotations from [https://github.com/Pilhyeon/BaSNet-pytorch](https://github.com/Pilhyeon/BaSNet-pytorch).
2. Place the features and annotations inside the `data/THUMOS14/` folder.
## Core Files
1. Configuration:
   
   `./config/config.py`
2. Data loading: 
   
   `./dataset/thumos_feature.py`  
3. Model definition:

   `./model/model.py`
4. Training process: 

   `./main.py`
## Train and Evaluate
1. Train the baseline model by run 
   ```
   python main_base.py --exp_name baseline
   ```
   The model would be automatically evaluated during training, and the log file and checkpoints would be saved in the folder `outputs/baseline`.
2. Train the our model by run 
   ```
   python main.py --exp_name c3bn
   ```
   The model would be automatically evaluated during training, and the log file and checkpoints would be saved in the folder `outputs/c3bn`.
   
## References
We referenced the repos below for the code.

 * [https://github.com/Pilhyeon/BaSNet-pytorch](https://github.com/Pilhyeon/BaSNet-pytorch)
 * [https://github.com/layer6ai-labs/ASL](https://github.com/layer6ai-labs/ASL) 
 * [https://github.com/LeonHLJ/RSKP](https://github.com/LeonHLJ/RSKP). 


