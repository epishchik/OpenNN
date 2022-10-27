<div align="center">

**Open Neural Networks library for image classification.**

[![PyPI](https://img.shields.io/pypi/v/opennn_pytorch?color=blue&style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/opennn-pytorch/) 
[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/Pe4enIks/OpenNN/blob/main/LICENSE)
<br>
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://github.com/Pe4enIks/OpenNN/blob/main/docker/Dockerfile)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org)

</div>

### Table of content
  1. [Quick start](#start)
  2. [Warnings](#warnings)
  3. [Encoders](#encoders)
  4. [Decoders](#decoders)
  5. [Pretrained](#pretrained)
  6. [Pretrained old configs fixes](#pretrained_old)
  7. [Datasets](#datasets)
  8. [Losses](#losses)
  9. [Metrics](#metrics)
  10. [Optimizers](#optimizers)
  11. [Schedulers](#schedulers)
  12. [Examples](#examples)
  13. [Wandb](#wandb)

### Quick start <a name="start"></a>
#### 1. Straight install.
##### 1.1 Install torch with cuda.
```bash
pip install -U torch --extra-index-url https://download.pytorch.org/whl/cu113
```
##### 1.2 Install opennn_pytorch.
```bash
pip install -U opennn_pytorch
```
#### 2. Dockerfile.
```bash
cd docker/
docker build -t opennn:latest .
```

### Warnings <a name="warnings"></a>
1. Cuda is only supported for nvidia graphics cards.
2. Alexnet decoder doesn't support bce losses family.
3. Sometimes combination of dataset/encoder/decoder/loss will give bad results, try to combine others.
4. Custom cross-entropy support only mode when preds have (n, c) shape and labels have (n) shape.
5. Not all options in transform.yaml and config.yaml are required.
6. Mean and std in datasets section must be used in transform.yaml, for example [mean=[0.2859], std=[0.3530]] -> normalize: [[0.2859], [0.3530]]

### Encoders <a name="encoders"></a>
- LeNet [[paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/lenet.py)]
- AlexNet [[paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/alexnet.py)]
- GoogleNet [[paper](https://arxiv.org/pdf/1409.4842.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/googlenet.py)]
- ResNet18 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/resnet.py)]
- ResNet34 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/resnet.py)]
- ResNet50 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/resnet.py)]
- ResNet101 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/resnet.py)]
- ResNet152 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/resnet.py)]
- MobileNet [[paper](https://arxiv.org/pdf/1704.04861.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/mobilenet.py)]
- VGG-11 [[paper](https://arxiv.org/pdf/1409.1556.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/vgg.py)]
- VGG-16 [[paper](https://arxiv.org/pdf/1409.1556.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/vgg.py)]
- VGG-19 [[paper](https://arxiv.org/pdf/1409.1556.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/vgg.py)]
  
### Decoders <a name="decoders"></a>
- LeNet [[paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/decoders/lenet.py)]
- AlexNet [[paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/decoders/alexnet.py)]
- Linear [[docs](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/decoders/linear.py)]

### Pretrained <a name="pretrained"></a>
<details>
<summary style="margin-left: 25px;">LeNet</summary>
<div style="margin-left: 25px;">

|Encoder             |Decoder             |Dataset             |Weights           |Configs           |Logs              |
|--------------------|--------------------|--------------------|------------------|------------------|------------------|
|LeNet               |LeNet               |MNIST               |[BEST](https://cloud.mail.ru/public/Xr5z/42Mx6gC7p), [PLAN](https://cloud.mail.ru/public/eAgN/iPFm2v4UG)          |[CONFIG](https://cloud.mail.ru/public/sCvP/UWXwUJxeb), [TRANSFORM](https://cloud.mail.ru/public/L8tj/kKEf5bKiA)               |[TRAINVAL](https://cloud.mail.ru/public/c2BY/6BnctFNWd)               |
|LeNet               |LeNet               |FASHION-MNIST       |[BEST](https://cloud.mail.ru/public/TDuA/86rXD6yNV), [PLAN](https://cloud.mail.ru/public/RtcL/81CTm8doR)          |[CONFIG](https://cloud.mail.ru/public/fZgh/FMzWYiDGS), [TRANSFORM](https://cloud.mail.ru/public/H2ic/j8gnpMGKV)               |[TRAINVAL](https://cloud.mail.ru/public/5c1r/GsRgkFBsX)               |
|LeNet               |Linear              |MNIST               |[BEST](https://cloud.mail.ru/public/REu7/za1tGB6qb), [PLAN](https://cloud.mail.ru/public/8Kfc/51kRbBdau)          |[CONFIG](https://cloud.mail.ru/public/5Ztd/vFESYuv81), [TRANSFORM](https://cloud.mail.ru/public/xuwr/665RBtYTM)               |[TRAINVAL](https://cloud.mail.ru/public/EHHp/mr4FzmiTX)               |
|LeNet               |Linear              |FASHION-MNIST       |[BEST](https://cloud.mail.ru/public/5oFr/YgrEuCjwY), [PLAN](https://cloud.mail.ru/public/cau1/17FS5Ab8L)          |[CONFIG](https://cloud.mail.ru/public/iNrh/oLbWHydYW), [TRANSFORM](https://cloud.mail.ru/public/5mAj/fPWjDGZjb)               |[TRAINVAL](https://cloud.mail.ru/public/2WUX/VZXsvBs8Z)               |
|LeNet               |AlexNet             |MNIST               |[BEST](https://cloud.mail.ru/public/oX7D/YkvXxgftt), [PLAN](https://cloud.mail.ru/public/wjSw/woE5Zh2NS)          |[CONFIG](https://cloud.mail.ru/public/vo77/xiGPwELZu), [TRANSFORM](https://cloud.mail.ru/public/k12g/UoHHx3iBJ)               |[TRAINVAL](https://cloud.mail.ru/public/5nQ3/FfnmxPZws)               |
|LeNet               |AlexNet             |FASHION-MNIST       |[BEST](https://cloud.mail.ru/public/MUeQ/yFo7Rsrgo), [PLAN](https://cloud.mail.ru/public/BSG8/xtMqCR6mt)          |[CONFIG](https://cloud.mail.ru/public/TCDL/SiXjmaDSV), [TRANSFORM](https://cloud.mail.ru/public/y3iw/5choydKkM)               |[TRAINVAL](https://cloud.mail.ru/public/f9k3/ZMpjAgStT)               |

</div>
</details>

<details>
<summary style="margin-left: 25px;">AlexNet</summary>
<div style="margin-left: 25px;">

|Encoder             |Decoder             |Dataset             |Weights           |Configs           |Logs              |
|--------------------|--------------------|--------------------|------------------|------------------|------------------|
|AlexNet             |LeNet               |MNIST               |[BEST](https://cloud.mail.ru/public/KiVr/TAe8EV3yf), [PLAN](https://cloud.mail.ru/public/PHRA/jZtdVdWPP)          |[CONFIG](https://cloud.mail.ru/public/np4t/A43jgb1oY), [TRANSFORM](https://cloud.mail.ru/public/FJ6x/SniuHZfEK)               |[TRAINVAL](https://cloud.mail.ru/public/nq7F/kBV9NeVEq)               |
|AlexNet             |LeNet               |FASHION-MNIST       |[BEST](https://cloud.mail.ru/public/nieg/s5Nf7Pmnq), [PLAN](https://cloud.mail.ru/public/ZrgH/joJn5xhsn)          |[CONFIG](https://cloud.mail.ru/public/bpFo/Vjmm7Dd4e), [TRANSFORM](https://cloud.mail.ru/public/Uqjw/P4BnF1xeQ)               |[TRAINVAL](https://cloud.mail.ru/public/7DAu/ygnrhn5X7)               |
|AlexNet             |Linear              |MNIST               |[BEST](https://cloud.mail.ru/public/N9nj/VeYTWXoyK), [PLAN](https://cloud.mail.ru/public/FyGQ/oLXxQDhYw)          |[CONFIG](https://cloud.mail.ru/public/ReTm/oMmkKVkmS), [TRANSFORM](https://cloud.mail.ru/public/fksy/B4tHCfQQf)               |[TRAINVAL](https://cloud.mail.ru/public/PvUP/Q3yubZu15)               |
|AlexNet             |Linear              |FASHION-MNIST        |[BEST](https://cloud.mail.ru/public/ZJPa/MtxqWu69p), [PLAN](https://cloud.mail.ru/public/uF4S/78sLapHdZ)          |[CONFIG](https://cloud.mail.ru/public/ReDL/9ya4e56vu), [TRANSFORM](https://cloud.mail.ru/public/qpZm/kM2jfk13a)               |[TRAINVAL](https://cloud.mail.ru/public/3hwi/EuKnoHiUw)               |
|AlexNet             |AlexNet             |MNIST               |[BEST](https://cloud.mail.ru/public/RUsN/dHia5LgKP), [PLAN](https://cloud.mail.ru/public/bKjQ/Jot1gWoLu)          |[CONFIG](https://cloud.mail.ru/public/iPJX/rtcFg7c8n), [TRANSFORM](https://cloud.mail.ru/public/HUxD/jkdVQ8gt4)               |[TRAINVAL](https://cloud.mail.ru/public/7Y5x/92fZTtd1A)               |
|AlexNet             |AlexNet             |FASHION-MNIST       |[BEST](https://cloud.mail.ru/public/q93Z/RRbNEXJrk), [PLAN](https://cloud.mail.ru/public/KXoT/wNxScRaBm)          |[CONFIG](https://cloud.mail.ru/public/vxwb/VivRXsEud), [TRANSFORM](https://cloud.mail.ru/public/Df9p/vd82CP8h1)               |[TRAINVAL](https://cloud.mail.ru/public/sVvz/CDoCkgt8f)               |

</div>
</details>

<details>
<summary style="margin-left: 25px;">GoogleNet</summary>
<div style="margin-left: 25px;">

|Encoder             |Decoder             |Dataset             |Weights           |Configs           |Logs              |
|--------------------|--------------------|--------------------|------------------|------------------|------------------|
|GoogleNet           |Linear              |MNIST               |[BEST](https://cloud.mail.ru/public/oTD1/xRa26UVaG), [PLAN](https://cloud.mail.ru/public/ByvC/y1oZ2icei)          |[CONFIG](https://cloud.mail.ru/public/Lt9M/WxiehNMwL), [TRANSFORM](https://cloud.mail.ru/public/hGb4/BanKp7YrQ)               |[TRAINVAL](https://cloud.mail.ru/public/5W22/SFLgwsVqu)               |
|GoogleNet           |Linear              |FASHION-MNIST       |[BEST](https://cloud.mail.ru/public/rhKv/ATwGUW3bQ), [PLAN](https://cloud.mail.ru/public/TDCf/Txpg8HY3c)          |[CONFIG](https://cloud.mail.ru/public/zkaJ/WB8Zp2c7E), [TRANSFORM](https://cloud.mail.ru/public/Y1GU/5KzEgDdug)               |[TRAINVAL](https://cloud.mail.ru/public/q9PD/7Sk3rHVp8)               |

</div>
</details>
  
<details>
<summary style="margin-left: 25px;">ResNet</summary>
<div style="margin-left: 25px;">

|Encoder             |Decoder             |Dataset             |Weights           |Configs           |Logs              |
|--------------------|--------------------|--------------------|------------------|------------------|------------------|
|ResNet18            |Linear              |MNIST               |[BEST](https://cloud.mail.ru/public/HNQe/GrQTYVR4D), [PLAN](https://cloud.mail.ru/public/np89/1ZKoTVruk)          |[CONFIG](https://cloud.mail.ru/public/bW8W/Xxmpk9r2a), [TRANSFORM](https://cloud.mail.ru/public/WxRb/YgwRkWjkA)               |[TRAINVAL](https://cloud.mail.ru/public/8zXr/6z4gY4Gsp)               |
|ResNet34            |Linear              |MNIST               |[BEST](https://cloud.mail.ru/public/eJZ9/15V8dgGKg), [PLAN](https://cloud.mail.ru/public/SoxA/gcWABVppm)          |[CONFIG](https://cloud.mail.ru/public/9bp6/G6q7ZPamn), [TRANSFORM](https://cloud.mail.ru/public/hG4D/xF9mTFF6P)               |[TRAINVAL](https://cloud.mail.ru/public/A97j/EKN2WKseA)               |
|ResNet50            |Linear              |MNIST               |[BEST](https://cloud.mail.ru/public/m6dS/dEphhQGq7), [PLAN](https://cloud.mail.ru/public/Uu4W/3fEtherM7)          |[CONFIG](https://cloud.mail.ru/public/Bzzg/igi13GBuD), [TRANSFORM](https://cloud.mail.ru/public/LMRU/Tuo1Abc9j)               |[TRAINVAL](https://cloud.mail.ru/public/ednt/o5MTGJQxX)               |
|ResNet101           |Linear              |MNIST               |[BEST](https://cloud.mail.ru/public/GFT9/METVBdbLG), [PLAN](https://cloud.mail.ru/public/o2sL/CvKQw9Dde)          |[CONFIG](https://cloud.mail.ru/public/dgCC/Ead9DPqiq), [TRANSFORM](https://cloud.mail.ru/public/CRzw/1oWVSkjEt)               |[TRAINVAL](https://cloud.mail.ru/public/1drh/eonZo69jx)               |
|ResNet152           |Linear              |MNIST               |[BEST](https://cloud.mail.ru/public/cp6D/YGM7rqFQk), [PLAN](https://cloud.mail.ru/public/BJsM/yi34akeYU)          |[CONFIG](https://cloud.mail.ru/public/c9iD/b2U9TTU54), [TRANSFORM](https://cloud.mail.ru/public/kYvR/y2EscgW3d)               |[TRAINVAL](https://cloud.mail.ru/public/XCNJ/f9Z1F8D8C)               |

</div>
</details>

<details>
<summary style="margin-left: 25px;">MobileNet</summary>
<div style="margin-left: 25px;">

|Encoder             |Decoder             |Dataset             |Weights           |Configs           |Logs              |
|--------------------|--------------------|--------------------|------------------|------------------|------------------|
|MobileNet           |Linear              |MNIST               |[BEST](https://cloud.mail.ru/public/1N8q/a7W615oHq), [PLAN](https://cloud.mail.ru/public/rje5/L4fdJr9BE)          |[CONFIG](https://cloud.mail.ru/public/whgD/ntaS6BebM), [TRANSFORM](https://cloud.mail.ru/public/h2Tm/mLaD4bYeN)               |[TRAINVAL](https://cloud.mail.ru/public/F6X1/jhDRPYoGb)               |

</div>
</details>

<details>
<summary style="margin-left: 25px;">VGG</summary>
<div style="margin-left: 25px;">

|Encoder             |Decoder             |Dataset             |Weights           |Configs           |Logs              |
|--------------------|--------------------|--------------------|------------------|------------------|------------------|
|VGG-11              |Linear              |MNIST               |[BEST](https://cloud.mail.ru/public/6wUm/vKJxLiHtJ), [PLAN](https://cloud.mail.ru/public/3rEd/MXwrSKPRE)          |[CONFIG](https://cloud.mail.ru/public/hTe5/jA8PrYgdN), [TRANSFORM](https://cloud.mail.ru/public/nUjx/PLKuqkNU4)               |[TRAINVAL](https://cloud.mail.ru/public/SceV/Mk4PHzg6n)               |
|VGG-16              |Linear              |MNIST               |[BEST](https://cloud.mail.ru/public/mGsd/SY83ovV1R), [PLAN](https://cloud.mail.ru/public/fStQ/Xn5eRRgXo)          |[CONFIG](https://cloud.mail.ru/public/F4q4/Qc7FWLHeq), [TRANSFORM](https://cloud.mail.ru/public/tDAy/K1E3Gix3B)               |[TRAINVAL](https://cloud.mail.ru/public/w3ed/HLSCjA9NU)               |
|VGG-19              |Linear              |MNIST               |[BEST](https://cloud.mail.ru/public/mMUk/a5VWw8PN1), [PLAN](https://cloud.mail.ru/public/BfZb/ps46dKYgy)          |[CONFIG](https://cloud.mail.ru/public/2ZQJ/6BeeqB1rj), [TRANSFORM](https://cloud.mail.ru/public/Nhya/QFxGiRdvG)               |[TRAINVAL](https://cloud.mail.ru/public/urow/wbKkV6uyv)               |

</div>
</details>

### Pretrained configs issues <a name="pretrained_old"></a>
Config file changed, check configs folder!!!
1. Config must include test_part value, (train_part + valid_part + test_part) value can be < 1.0.
2. You will able to add wandb structure for logging in wandb.
3. Full restructure into branches structure.

### Datasets <a name="datasets"></a>
- MNIST [[files](http://yann.lecun.com/exdb/mnist/)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/datasets/mnist.py)] [classes=10] [mean=[0.1307], std=[0.3801]]
- FASHION-MNIST [[files](https://github.com/zalandoresearch/fashion-mnist)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/datasets/mnist.py)] [classes=10] [mean=[0.2859], std=[0.3530]]
- CIFAR-10 [[files](https://www.cs.toronto.edu/~kriz/cifar.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/datasets/cifar.py)] [classes=10] [mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]]
- CIFAR-100 [[files](https://www.cs.toronto.edu/~kriz/cifar.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/datasets/cifar.py)] [classes=100] [mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]]
- GTSRB [[files](https://benchmark.ini.rub.de/gtsrb_news.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/datasets/gtsrb.py)] [classes=43] [mean=unknown, std=unknown]
- CUSTOM [[docs](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/datasets/custom.py)] [[example](https://cloud.mail.ru/public/PyuH/sNiZrFGK4)] [classes=nc] [mean=unknown, std=unknown]

### Losses <a name="losses"></a>
- Cross-Entropy [[pytorch](https://pytorch.org), [custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn_pytorch/losses)] [[docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/losses/celoss.py)]
- Binary-Cross-Entropy [[pytorch](https://pytorch.org), [custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn_pytorch/losses)] [[docs](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/losses/bceloss.py)]
- Binary-Cross-Entropy-With-Logits [[pytorch](https://pytorch.org), [custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn_pytorch/losses)] [[docs](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/losses/bceloss.py)]
- Mean-Squared-Error [[pytorch](https://pytorch.org), [custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn_pytorch/losses)] [[docs](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/losses/meanloss.py)]
- Mean-Absolute-Error [[pytorch](https://pytorch.org), [custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn_pytorch/losses)] [[docs](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/losses/meanloss.py)]

### Metrics <a name="metrics"></a>
- Accuracy [[custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn_pytorch/metrics)] [[docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/metrics/accuracy.py)]
- Precision [[sklearn](https://scikit-learn.org/stable/)] [[docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/metrics/precision.py)]
- Recall [[sklearn](https://scikit-learn.org/stable/)] [[docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/metrics/recall.py)]
- F1 [[sklearn](https://scikit-learn.org/stable/)] [[docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/metrics/f1_score.py)]

### Optimizers <a name="optimizers"></a>
- Adam [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/optimizers/adam.py)]
- AdamW [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/optimizers/adam.py)]
- Adamax [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/optimizers/adam.py)]
- RAdam [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/optimizers/adam.py)]
- NAdam [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/optimizers/adam.py)]

### Schedulers <a name="schedulers"></a>
- StepLR [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/schedulers/steplr.py)]
- MultiStepLR [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/schedulers/steplr.py)]
- PolynomialLRDecay [[custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn_pytorch/schedulers)] [[docs](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/schedulers/custom.py)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/schedulers/custom.py)]

### Examples <a name="examples"></a>

1. Run from yaml config.
```python
from opennn_pytorch import run

config = 'path to yaml config'  # check configs folder
run(config)
```

2. Get encoder and decoder.
```python
from opennn_pytorch.encoders import get_encoder
from opennn_pytorch.decoders import get_decoder
  
encoder_name = 'resnet18'
decoder_name = 'alexnet'
decoder_mode = 'decoder'
input_channels = 1
number_classes = 10
device = 'cuda'

encoder = get_encoder(encoder_name, 
                       input_channels).to(device)

model = get_decoder(decoder_name, 
                    encoder, 
                    number_classes, 
                    decoder_mode, 
                    device).to(device)
```

3.1 Get dataset.
```python
from opennn_pytorch.datasets import get_dataset
from torchvision import transforms

transform_config = 'path to transform yaml config'
dataset_name = 'mnist'
datafiles = None
train_part = 0.7
valid_part = 0.2
test_part = 0.05

transform_lst = opennn_pytorch.transforms_lst(transform_config)
transform = transforms.Compose(transform_lst)

train_data, valid_data, test_data = get_dataset(dataset_name,
                                                train_part, 
                                                valid_part, 
                                                test_part, 
                                                transform, 
                                                datafiles)
```

3.2 Get custom dataset.
```python
from opennn_pytorch.datasets import get_dataset
from torchvision import transforms

transform_config = 'path to transform yaml config'
dataset_name = 'custom'
images = 'path to folder with images'
annotation = 'path to annotation yaml file with image: class structure'
datafiles = (images, annotation)
train_part = 0.7
valid_part = 0.2
test_part = 0.05

transform_lst = opennn_pytorch.transforms_lst(transform_config)
transform = transforms.Compose(transform_lst)

train_data, valid_data, test_data = get_dataset(dataset_name,
                                                train_part, 
                                                valid_part, 
                                                test_part, 
                                                transform, 
                                                datafiles)
```

4. Get optimizer.
```python
from opennn_pytorch.optimizers import get_optimizer

optim_name = 'adam'
lr = 1e-3
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 1e-6

optimizer = get_optimizer(optim_name, 
                          model, 
                          lr=lr, 
                          betas=betas, 
                          eps=opt_eps, 
                          weight_decay=weight_decay)
```

5. Get scheduler.
```python
from opennn_pytorch.schedulers import get_scheduler

scheduler_name = 'steplr'
step = 10
gamma = 0.5
milestones = None
max_decay_steps = None
end_lr = None
power = None

scheduler = get_scheduler(scheduler_name,
                          optimizer,
                          step=step,
                          gamma=gamma,
                          milestones=milestones,
                          max_decay_steps=mdsteps,
                          end_lr=end_lr,
                          power=power)
```

6. Get loss function.
```python
from opennn_pytorch.losses import get_loss

loss_fn = 'custom_mse'
loss_fn, one_hot = get_loss(loss_fn)
```

7. Get metrics functions.
```python
from opennn_pytorch.metrics import get_metrics

metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
number_classes = 10
metrics_fn = get_metrics(metrics_names, 
                         nc=number_classes)
```

8. Train/Test.
```python
from opennn_pytorch.algo import train, test, prediction
import random

algorithm = 'train'
batch_size = 16
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
number_classes = 10
save_every = 5
epochs = 20
wandb_flag = True
wandb_metrics = ['accuracy', 'f1_score']

train_dataloader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=batch_size, 
                                               shuffle=True)

valid_dataloader = torch.utils.data.DataLoader(valid_data, 
                                               batch_size=batch_size, 
                                               shuffle=False)

test_dataloader = torch.utils.data.DataLoader(test_data, 
                                              batch_size=1, 
                                              shuffle=False)

if algorithm == 'train':
  train(train_dataloader, 
        valid_dataloader, 
        model, 
        optimizer, 
        scheduler, 
        loss_fn, 
        metrics_fn, 
        epochs, 
        checkpoints, 
        logs, 
        device, 
        save_every, 
        one_hot, 
        number_classes,
        wandb_flag,
        wandb_metrics)
elif algorithm == 'test':
  test_logs = test(test_dataloader, 
                   model, 
                   loss_fn, 
                   metrics_fn, 
                   logs, 
                   device, 
                   one_hot, 
                   number_classes,
                   wandb_flag,
                   wandb_metrics)
  if pred:
    indices = random.sample(range(0, len(test_data)), 10)
    os.mkdir(test_logs + '/prediction', 0o777)
    for i in range(10):
      tmp_range = range(number_classes)
      tmp_dct = {i: names[i] for i in tmp_range}
      prediction(test_data,
                 model,
                 device,
                 tmp_dct,
                 test_logs + f'/prediction/{i}',
                 indices[i])
```

### Wandb <a name="wandb"></a>

[Wandb](https://wandb.ai/) is very powerful logging tool, you will able to log metrics, hyperparamets, model hooks etc.

```bash
wandb login
<your api token from wandb.ai>
```

![Workspace](images/wandb1.png)

![Table](images/wandb2.png)

### Citation <a name="citing"></a>
Project [citation](https://github.com/Pe4enIks/OpenNN/blob/main/CITATION.cff).

### License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/Pe4enIks/OpenNN/blob/main/LICENSE).
