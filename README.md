# MLFF: Multi-Level Feature Fusion for Continual Learning in Visual Quality Inspection 

This repo. contains code for the method propsed in our work on *Multi-Level Feature Fusion for Continual Learning in Visual Quality Inspection* (see reference at the bottom). We give a short explanation below. For more details please have a look at our paper. 

## About 🔍

**Aim:** Allowing for frequent adaptation of deep neural networks (DNN) in high-mix, low-volume manufacturing applications, by improving the computational efficiency of re-training processes and robustness to catastrophic forgetting.  

**Problem:** End-to-end training of DNN can be computationally costly and facilitate overfitting. Using a frozen feature extractor often leads to subpar performance in manufacturing applications, like, e.g., visual quality inspection. 

**Approach:** The idea is simple. We extract multiple latent representations from a pretrained, frozen feature extractor DNN, fuse them via concatenation and feed them to a classifier. During training only the classifier is updated. 

**Results:**

⚖️ The reduced number of trained model parameters increases training speed and reduces required GPU memory, while a predictive performance comparable to end-to-end training is maintained.   

🧠 The robustness to catastrophic forgetting in rehearsal-based continual learning settings is improved. 

**What's included:**

We provide implementations for 4 model architectures:
- ResNet-50 [1](https://arxiv.org/abs/1512.03385)
- DINOv2 with registers [2](https://arxiv.org/html/2309.16588v2) 
- SwinV2-B [3](https://arxiv.org/abs/2111.09883) 
- MobileNetV2-S [4](https://arxiv.org/abs/1801.04381) 

Currently only classification is supported. 


## Setup 💻

**Requirements**

You will need:

- Python environment with Python>=3.10
- Installation of `torch` and `torchvision`, see PyTorch's [Getting Started](https://pytorch.org/get-started/locally/#linux-pip). Although most PyTorch versions should work, this code was developed using `torch>=2.6.0`, see also section *Test* below. 
- Installation of Huggingface's `transformers`, see their [GitHub](https://github.com/huggingface/transformers)
 
**Installation**

- Clone this repository to your desired directory.
- From the local project directory run `pip install -e .`. The `-e` flag installs the package in editable mode. 

**Test**

The proejct was tested with the following configurations.

| OS            | torch | torchvision | numpy | transformers | cuda  | nvidia driver | 
|---------------|-------|-------------|-------|--------------|------|--------|
| Ubuntu 22.04  | 2.6.0 | 0.21.0      | 2.2.4 | 4.50.3       | 12.4 | 535    |
| *Ubuntu 22.04  | 2.9.0 | 0.24.0      | 2.3.4 | 4.57.1       | 12.8 | 570    |

*corresponds to the version used to export `requirements.txt`

**Examples**

A minimum working example is included in `./examples/main.py`.


## Cite 📝
If you use this package in your work, please cite out paper:

Bauer, J.C., Geng, P., Trattnig, S., Dokládal, P., Daub, R. (2025). Multi-Level Feature Fusion for Continual Learning in Visual Quality Inspection. Accpeted at *The 13th International Conference on Control, Mechatronics and Automation (ICCMA)*, Paris, France 
