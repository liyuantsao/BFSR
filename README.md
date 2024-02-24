# Boosting Flow-based Generative Super-Resolution Models via Learned Prior

**This is the official repository of "Boosting Flow-based Generative Super-Resolution Models via Learned Prior".**

[Li-Yuan Tsao](https://liyuantsao.github.io/), [Yi-Chen Lo](https://scholar.google.com/citations?user=EPYQ48sAAAAJ&hl=zh-TW), [Chia-Che Chang](https://scholar.google.com/citations?user=FK1RcpoAAAAJ&hl=zh-TW), [Hao-Wei Chen](https://scholar.google.com/citations?user=cpOf3qMAAAAJ&hl=en), [Roy Tseng](https://scholar.google.com/citations?user=uKgYlYYAAAAJ&hl=zh-TW), [Chien Feng](https://www.linkedin.com/in/chien-feng-528393293/), [Chun-Yi Lee](https://scholar.google.com/citations?user=5mYNdo0AAAAJ&hl=zh-TW)

[Arxiv (soon)] [[PDF](https://drive.google.com/file/d/1DGjE5cQHohMfZf-Aw170NEiCHj_NXu3t/view?usp=drive_link)]

> In this work, we identify several challenges in flow-based SR methods, including grid artifacts, exploding inverses, and suboptimal results due to a fixed sampling temperature. To tackle these issues, we introduce a learned prior, which is predicted by the proposed latent module, to the inference phase of flow-based SR models. This framework not only addresses the inherent issues in flow-based SR models but also enhances the quality of synthesized images without modifying the original design or pre-trained weights of these models. Our proposed framework is effective, flexible in design, and able to generalize to both fixed-scale and arbitrary-scale SR frameworks without requiring customized components.


<img width="65%" height="65%" alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/e0900fdf-b11b-44d0-912d-ca502c2628b3">

## Launch your experiments
This repository includes the training/evaluation code for LINF-LP, along with the evaluation code for SRFlow-LP (the training code will be released soon), which are the implementations after integrating the proposed latent module with LINF and SRFlow. 

To run your experiments on LINF-LP and SRFlow-LP, please refer to `LINF-LP` and `SRFlow-LP`, respectively.

## Results
### Arbitrary-scale SR results 
* The arbitrary-scale SR results on SR benchmark datasets. “In-scales” and “OOD-scales” refer to in- and out-of-training-distribution scales. LPIPS scores are reported (lower is better), with the best and second-best highlighted in red and blue, respectively.
<img width="926" alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/43b9761a-188a-4d70-964e-d556912471a0">



### Generative SR results
* The 4× SR results on the DIV2K validation set. The best results are highlighted in red.

<img width="500" alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/34a9cd9b-bd0e-45a8-b50b-c78358714336">


### Qualitative Results
* Our method tackles the grid artifacts and exploding inverse issue.

<img width="60%" height="60%" alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/229ff47e-86fa-4490-a573-1533a38f0f94">

---

* A qualitative comparison between the 4× SR results of SRFlow and our SRFlow-LP.

<img alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/9d876f03-3ecc-4b0d-bda0-eee8b55e94f9">
<br>
<img width="50%" height="50%" alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/47810536-4ecc-431e-8911-bac983fd62b0">

---

* A qualitative comparison between the 4× SR results of LINF and our LINF-LP.

<img width="70%" height="70%" alt="image" src="https://github.com/liyuantsao/FlowSR-LP/assets/73187544/4bee463a-ca1c-494e-9e7b-af5b0167b8f0">


## Acknowledgements
Our codes are built on LINF ([Paper](https://arxiv.org/abs/2303.05156), [code](https://github.com/JNNNNYao/LINF)) and SRFlow ([Paper](https://arxiv.org/abs/2006.14200), [Code](https://github.com/andreas128/SRFlow)), we appriciate their amazing works that advance this community.
