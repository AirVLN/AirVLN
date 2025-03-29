# 🚁 **AerialVLN: Vision-and-Language Navigation for UAVs**

[![GitHub stars](https://img.shields.io/github/stars/AirVLN/AirVLN?style=social)](https://github.com/AirVLN/AirVLN) 
[![License](https://img.shields.io/github/license/AirVLN/AirVLN)](LICENSE) 
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/AirVLN/AirVLN/actions)

---

## 📖 **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Getting Started](#getting-started)
4. [Simulator & Dataset](#dataset--simulator)
5. [Example Usage](#example-usage)
6. [Citation](#citation)
7. [Contact](#contact)

此外，你也可以参阅[本页面的中文版本](https://github.com/AirVLN/AirVLN/blob/main/README-ZH.md)

---

## 🌟 **Introduction**

Recently emerged Vision-and-Language Navigation (VLN) tasks have drawn significant attention in both computer vision and natural language processing communities. Existing VLN tasks are built for agents that navigate on the ground, either indoors or outdoors. However, many tasks require intelligent agents to carry out in the sky, such as UAV-based goods delivery, traffic/security patrol, and scenery tour, to name a few. Navigating in the sky is more complicated than on the ground because agents need to consider the flying height and more complex spatial relationship reasoning. To fill this gap and facilitate research in this field, we propose a new task named AerialVLN, which is UAV-based and towards outdoor environments. We develop a 3D simulator rendered by near-realistic pictures of 25 city-level scenarios. Our simulator supports continuous navigation, environment extension and configuration. We also proposed an extended baseline model based on the widely-used cross-modal-alignment (CMA) navigation methods. We find that there is still a significant gap between the baseline model and human performance, which suggests AerialVLN is a new challenging task.


---

## 🚀 **Features**

- **Realistic 3D Simulator**: 25 city-level scenarios with lifelike imagery.
- **Cross-Modal Alignment Model**: Advanced navigation using vision and language.
- **Extensible Framework**: Add new environments and configurations easily.
- **Comprehensive Dataset**: Includes AerialVLN and AerialVLN-S for training and evaluation.

![AerialVLN Demo](./files/instruction_graph.jpg)
Instruction: Take off, fly through the tower of cable bridge and down to the end of the road. Turn left, fly over the five-floor building with a yellow shop sign and down to the intersection on the left. Head to the park and turn right, fly along the edge of the park. March forward, at the intersection turn right, and finally land in front of the building with a red billboard on its rooftop.

---

## 🛠️ **Getting Started**

### Prerequisites
- Ubuntu Operating System
- Some NVidia GPUs
- Python 3.8+
- Conda for environment management


### Installation
```bash
# Enter the workspace folder
mkdir AirVLN_ws
cd AirVLN_ws

# Clone the repository
git clone https://github.com/AirVLN/AirVLN.git
cd AirVLN

# Create and activate a virtual environment
conda create -n AirVLN python=3.8
conda activate AirVLN

pip install pip==24.0 setuptools==63.2.0

# Install dependencies
pip install -r requirements.txt
pip install airsim==1.7.0

# Install PyTorch
# We suggest you select the right version on https://pytorch.org/get-started/locally/
pip install torch torchaudio torchvision

pip install pytorch-transformers==1.2.0

cd ..
mkdir -p ENVs\
  DATA/data/aerialvln\
  DATA/data/aerialvln-s\
  DATA/models/ddppo-models

# download models, datasets and simulators and place them into the right folders
```

Finally, your project dir should be like this:

```bash
- Project workspace
    - [AirVLN](https://www.kaggle.com/datasets/shuboliu/aerialvln-simulators)
    - DATA
        - data
            - aerialvln
    - ENVs
      - env_1
      - env_2
      - ...
```

## 📦 **Simulator & Dataset**

下载https://github.com/facebookresearch/habitat-lab/tree/v0.1.7/habitat_baselines/rl/ddppo  gibson-2plus-resnet50.pth

提供百度网盘地址。

For **AerialVLN simulators (~35GB)**, you may download via [Kaggle website](https://www.kaggle.com/datasets/shuboliu/aerialvln-simulators) by simplely click **Download**, or you may download them via cURL:
```bash
#!/bin/bash
curl -L -o ~/Downloads/aerialvln-simulators.zip\
  https://www.kaggle.com/api/v1/datasets/download/shuboliu/aerialvln-simulators
```

Alternatively, you may download it via kagglehub and then place it under your AerialVLN project.
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("shuboliu/aerialvln")

print("Path to dataset files:", path)
```


For **AerialVLN and AerialVLN-S** annotated data (both less than 100M), you may via [Kaggle website for AerialVLN](https://www.kaggle.com/datasets/shuboliu/aerialvln) and [Kaggle website for AerialVLN-S](https://www.kaggle.com/datasets/shuboliu/aerialvln-s) by simplely click **Download**, or you may download them via cURL:
```bash
#!/bin/bash
curl -L -o ~/Downloads/aerialvln.zip\
  https://www.kaggle.com/api/v1/datasets/download/shuboliu/aerialvln
```
and 
```bash
#!/bin/bash
curl -L -o ~/Downloads/aerialvln.zip\
  https://www.kaggle.com/api/v1/datasets/download/shuboliu/aerialvln-s
```

## 🔧 **Example Usage**

Please see the examples in [scripts](https://github.com/AirVLN/AirVLN/tree/main/scripts).



## 📜 **Citing**
If you use AerialVLN in your research, please cite the following paper:

```
@inproceedings{liu_2023_AerialVLN,
  title={AerialVLN: Vision-and-language Navigation for UAVs},
  author={Shubo Liu and Hongsheng Zhang and Yuankai Qi and Peng Wang and Yanning Zhang and Qi Wu},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

In addition, we have noticed that some scholars wish to apply the AerialVLN dataset and simulator to research areas beyond VLN. We fully welcome such endeavors! We also encourage you to contact [us](mailto:shubo.liu@mail.nwpu.edu.cn) and share the intended application areas of your research.

## ✉️ **Contact**
Feel free to contact [Shubo LIU](mailto:shubo.liu@mail.nwpu.edu.cn) via email [shubo.liu@mail.nwpu.edu.cn](mailto:shubo.liu@mail.nwpu.edu.cn) for more support.
