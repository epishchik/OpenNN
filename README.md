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
  2. [Encoders](#encoders)
  3. [Decoders](#decoders)
  4. [Datasets](#datasets)
  5. [Losses](#losses)
  6. [Metrics](#metrics)
  7. [Optimizers](#optimizers)
  8. [Schedulers](#schedulers)
  9. [Examples](#examples)

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
  
### Encoders <a name="encoders"></a>
- LeNet [[paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/lenet.py)]
- AlexNet [[paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/alexnet.py)]
- GoogleNet [[paper](https://arxiv.org/pdf/1409.4842.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/googlenet.py)]
- Resnet18 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/resnet.py)]
- Resnet34 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/resnet.py)]
- Resnet50 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/resnet.py)]
- Resnet101 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/resnet.py)]
- Resnet152 [[paper](https://arxiv.org/pdf/1512.03385.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/resnet.py)]
- Mobilent [[paper](https://arxiv.org/pdf/1704.04861.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/mobilenet.py)]
- VGG-11 [[paper](https://arxiv.org/pdf/1409.1556.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/vgg.py)]
- VGG-16 [[paper](https://arxiv.org/pdf/1409.1556.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/vgg.py)]
- VGG-19 [[paper](https://arxiv.org/pdf/1409.1556.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/encoders/vgg.py)]
  
### Decoders <a name="decoders"></a>
- LeNet [[paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/decoders/lenet.py)]
- AlexNet [[paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/decoders/alexnet.py)]

### Datasets <a name="datasets"></a>
- MNIST [[files](http://yann.lecun.com/exdb/mnist/)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/datasets/mnist.py)]
- CIFAR-10 [[files](https://www.cs.toronto.edu/~kriz/cifar.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/datasets/cifar.py)]
- CIFAR-100 [[files](https://www.cs.toronto.edu/~kriz/cifar.html)] [[code](https://github.com/Pe4enIks/OpenNN/blob/main/opennn_pytorch/datasets/cifar.py)]

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

### Examples <a name="examples"></a>
  
1. Run from yaml configs.
```python
import opennn_pytorch
  
config = 'path to yaml config'  # check configs folder
transform = 'path to transform yaml config'  # check configs folder

opennn_pytorch.run(config, transform)
```

2. Get encoder and decoder.
```python
import opennn_pytorch
  
encoder_name = 'resnet18'
decoder_name = 'alexnet'
decoder_mode = 'decoder'
input_channels = 1
number_classes = 10
device = 'cuda'

encoder = opennn_pytorch.encoders.get_encoder(encoder_name, input_channels).to(device)
model = opennn_pytorch.decoders.get_decoder(decoder_name, encoder, number_classes, decoder_mode, device).to(device)
```
  
3. Get dataset.
```python
import opennn_pytorch
from torchvision import transforms

transform_config = 'path to transform yaml config'
dataset_name = 'mnist'
train_part = 0.7
valid_part = 0.2

transform_lst = opennn_pytorch.transforms_lst(transform_config)
transform = transforms.Compose(transform_lst)
  
train_data, valid_data, test_data = opennn_pytorch.datasets.get_dataset(dataset_name, train_part, valid_part, transform)
```

4. Get optimizer.
```python
import opennn_pytorch

optim_name = 'adam'
lr = 1e-3
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 1e-6
optimizer = opennn_pytorch.optimizers.get_optimizer(optim_name, model, lr=lr, betas=betas, eps=opt_eps, weight_decay=weight_decay)
```

5. Get scheduler.
```python
import opennn_pytorch

scheduler_name = 'steplr'
step = 10
gamma = 0.5
scheduler = opennn_pytorch.schedulers.get_scheduler(sched, optimizer, step=step, gamma=gamma, milestones=None)
```

6. Get loss function.
```python
import opennn_pytorch

loss_fn = 'custom_mse'
loss_fn, one_hot = opennn_pytorch.losses.get_loss(loss_fn)
```

7. Get metrics functions.
```python
import opennn_pytorch

metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
number_classes = 10
metrics_fn = opennn_pytorch.metrics.get_metrics(metrics_names, nc=number_classes)
```

8. Train/Test.
```python
import opennn_pytorch

algorithm = 'train'
batch_size = 16
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
number_classes = 10
save_every = 5
epochs = 20

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

if algorithm == 'train':
  opennn_pytorch.algo.train(train_dataloader, valid_dataloader, model, optimizer, scheduler, loss_fn, metrics_fn, epochs, checkpoints, logs, device, save_every, one_hot, number_classes)
elif algorithm == 'test':
  opennn_pytorch.algo.test(test_dataloader, model, loss_fn, metrics_fn, logs, device, one_hot, number_classes)
  if viz:
    for _ in range(10):
      opennn_pytorch.algo.vizualize(valid_data, model, device, {i: class_names[i] for i in range(number_classes)})
```

### Citation <a name="citing"></a>
Project [citation](https://github.com/Pe4enIks/OpenNN/blob/main/CITATION.cff).

### License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/Pe4enIks/OpenNN/blob/main/LICENSE).
