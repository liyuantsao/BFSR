# Boosting Flow-based Generative Super-Resolution Models via Learned Prior

This is the official repository of "Boosting Flow-based Generative Super-Resolution Models via Learned Prior".

[Li-Yuan Tsao](https://liyuantsao.github.io/), [Yi-Chen Lo](https://scholar.google.com/citations?user=EPYQ48sAAAAJ&hl=zh-TW), [Chia-Che Chang](https://scholar.google.com/citations?user=FK1RcpoAAAAJ&hl=zh-TW), [Hao-Wei Chen](https://scholar.google.com/citations?user=cpOf3qMAAAAJ&hl=en), [Roy Tseng](https://scholar.google.com/citations?user=uKgYlYYAAAAJ&hl=zh-TW), Chien Feng, [Chun-Yi Lee](https://scholar.google.com/citations?user=5mYNdo0AAAAJ&hl=zh-TW)


<img width="70%" height="70%" alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/9fb598bd-2c1c-47e7-b0a4-35385027c8c2">

## Launch your experiments
This repository includes the training/evaluation code for LINF-LP, along with the evaluation code for SRFlow-LP (the training code will be released soon), which are the implementations after integrating the proposed latent module with LINF and SRFlow. 

To run your experiments on LINF-LP and SRFlow-LP, please refer to `LINF-LP` and `SRFlow-LP`, respectively.

## Results
### Arbitrary-scale SR results 
* The arbitrary-scale SR results on SR benchmark datasets. “In-scales” and “OOD-scales” refer to in- and out-of-training-distribution scales. LPIPS scores are reported (lower is better), with the best and second-best highlighted in red and blue, respectively.
<img width="926" alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/b3afc44b-8932-4e00-83f5-61dab1d7fec8">

### Generative SR results
* The 4× SR results on the DIV2K validation set. The best results are highlighted in red.

<img width="359" alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/07237dd3-4444-4f7b-af1a-2e4640e9a9fe">

### Qualitative Results
* Our method tackles the grid artifacts and exploding inverse issue.

<img width="75%" height="75%" alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/cd6e3a1f-6236-4808-b20b-4b9a04339f79">

---

* A qualitative comparison between the 4× SR results of LINF and our LINF-LP.

<img width="75%" height="75%" alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/c318778d-1c23-4e41-a306-b795d218dd41">

---

* A qualitative comparison between the 4× SR results of SRFlow and our SRFlow-LP.

<img alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/cca13c15-490c-4fb9-945b-6098887ad769">
<img width="75%" height="75%" alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/51c97d4f-0aa3-4698-b698-540b213ec7ff">



## Acknowledgements
Our codes are built on LINF ([Paper](https://arxiv.org/abs/2303.05156), [code](https://github.com/JNNNNYao/LINF)) and SRFlow ([Paper](https://arxiv.org/abs/2006.14200), [Code](https://github.com/andreas128/SRFlow)), we appriciate their amazing works that advance this community.
