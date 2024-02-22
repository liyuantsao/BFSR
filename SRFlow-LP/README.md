# SRFlow-LP
Here is the evaluation code when integrating our latent module with [SRFlow](https://arxiv.org/abs/2006.14200).

## Setup
### 1. Environment and Datasets
You can create an environment, install the packages, and download the datasets and pretrained models of SRFlow using the following command:
```bash
bash ./setup.sh
```
## Running your experiments
### 1. Training (Coming Soon)

### 2. Evaluation
Please download and put our [pre-trained model](https://drive.google.com/file/d/19w9dDQdgOGRbmP-x2eAb-BM81G2adf4_/view?usp=drive_link) into `./pretrained_models`

```bash
cd code
python3 test.py confs/SRFlow-LP_DF2K_4X.yml
```

* The generated images are saved in `./results/SRFlow-LP`
