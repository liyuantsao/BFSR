# LINF-LP
Here is the training/evaluation code when integrating our latent module with [LINF](https://arxiv.org/abs/2303.05156).

## Setup
### 1. Environment
```bash
pip install -r requirements.txt
pip install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. Datasets
a. [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)<br>
* For training and evaluating edsr-baseline-linf-LP and rrdb-linf-LP, you have to download `DIV2K/DIV2K_train_HR/`, `DIV2K/DIV2K_valid_HR/`, and `DIV2K/DIV2K_valid_LR_bicubic/X4/` in this website.
b. [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar)<br>
* For training rrdb-linf-LP, we first transform Flickr2K dataset to `.pkl` files.


## Running your experiments
* Please modify the `root_path` in the config files to the path of your datasets.

### 1. Training
To train your latent module, please use the following command:
```bash
# for edsr-baseline-linf-LP
python3 train.py --config config/train/train_edsr-baseline-linf-LP.yaml --gpu 0 --name edsr-baseline-linf-LP --patch 3
# for rrdb-linf-LP
python3 train.py --config config/train/train_rrdb-linf-LP.yaml --gpu 0 --name rrdb-linf-LP --patch 3
```
* The checkpoints of the latent module will be saved in `./save/EXP_NAME`. 

### 2. Evaluation
To evaluate the results after integrating LINF with our latent module, please use the following command:
```bash
# for edsr-baseline-linf-LP
python3 test.py --config configs/test/test-fast-div2k-4.yaml --model edsr-baseline-linf.pth --prior_model edsr-baseline-linf-LP.pth --gpu 0 --detail --patch --sample 100 --name edsr-baseline-linf-LP-div2k-4xSR
# for rrdb-linf-LP
python3 test.py --config configs/test/test-fast-div2k-4.yaml --model rrdb-linf.pth --prior_model rrdb-linf-LP.pth --gpu 0 --detail --patch --sample 100 --name rrdb-linf-LP-div2k-4xSR
```

* The generated images are saved in `./sample/edsr-baseline-linf-LP-div2k-4xSR` for edsr-baseline-linf-LP, and in `./sample/rrdb-linf-LP-div2k-4xSR` for rrdb-linf-LP.
