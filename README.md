<div align="center">

**Open Neural Networks library for image classification.**

[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/Pe4enIks/OpeNN/LICENSE) 
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.4+-blue?style=for-the-badge&logo=pytorch)](https://pepy.tech/project/segmentation-models-pytorch) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.6+-blue?style=for-the-badge&logo=python&logoColor=white)](https://pepy.tech/project/segmentation-models-pytorch)

</div>

### Table of content
  1. [Quick start](#start)
  2. [Encoders](#encoders)
  3. [Decoders](#decoders)
  4. [Examples](#examples)

### Quick start <a name="start"></a>
#### 1. Straight install.
##### 1.1 Install torch with cuda.
```bash
pip install -U torch --extra-index-url https://download.pytorch.org/whl/cu113
```
##### 1.2 Install opennn.
```bash
pip install -U opennn
```
#### 2. Dockerfile.
```bash
cd docker/
docker build .
```
  
### Encoders <a name="encoders"></a>
- LeNet [[paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)] [[code](opennn/encoders/lenet.py)]
- AlexNet [[paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)] [[code](opennn/encoders/alexnet.py)]
- GoogleNet [[paper](https://arxiv.org/pdf/1409.4842.pdf)] [[code](opennn/encoders/googlenet.py)]
- Resnet18 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](opennn/encoders/resnet.py)]
- Resnet34 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](opennn/encoders/resnet.py)]
- Resnet50 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](opennn/encoders/resnet.py)]
- Resnet101 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](opennn/encoders/resnet.py)]
- Resnet152 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](opennn/encoders/resnet.py)]
  
### Decoders <a name="decoders"></a>
- LeNet [[paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)] [[code](opennn/decoders/lenet.py)]
- AlexNet [[paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)] [[code](opennn/decoders/alexnet.py)]

### Examples <a name="examples"></a>
  
1. Run from yaml configs.
```python
import opennn
  
config = 'path to yaml config'
transform = 'path to transform yaml config'

opennn.run(config, transform)
```

2. Get encoder and decoder.
```python
import opennn
  
encoder_name = 'resnet18'
decoder_name = 'alexnet'
decoder_mode = 'decoder'
input_channels = 1
number_classes = 10
device = 'cuda'

encoder = opennn.get_encoder(encoder_name, input_channels).to(device)
model = opennn.get_decoder(decoder_name, encoder, number_classes, deocder_mode, device).to(device)
```
  
3. Get dataset.
```python
import opennn
from torchvision import transforms

transform_config = 'path to transform yaml config'
dataset_name = 'mnist'
train_part = 0.7
valid_part = 0.2

transform_lst = opennn.transforms_lst(transform_config)
transform = transforms.Compose(transform_lst)
  
train_data, valid_data, test_data = opennn.get_dataset(dataset_name, train_part, valid_part, transform)
```
