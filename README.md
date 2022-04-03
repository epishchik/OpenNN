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

### Datasets <a name="datasets"></a>
- MNIST [[files](http://yann.lecun.com/exdb/mnist/)] [[code](opennn/datasets/mnist.py)]
- CIFAR-10 [[files](https://www.cs.toronto.edu/~kriz/cifar.html)] [[code](opennn/datasets/cifar.py)]
- CIFAR-100 [[files](https://www.cs.toronto.edu/~kriz/cifar.html)] [[code](opennn/datasets/cifar.py)]

### Losses <a name="losses"></a>
- Cross-Entropy [[pytorch](https://pytorch.org), [custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn/losses)] [[docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)] [[code](opennn/losses/celoss.py)]
- Binary-Cross-Entropy [[pytorch](https://pytorch.org), [custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn/losses)] [[docs](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)] [[code](opennn/losses/bceloss.py)]
- Binary-Cross-Entropy-With-Logits [[pytorch](https://pytorch.org), [custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn/losses)] [[docs](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)] [[code](opennn/losses/bceloss.py)]
- Mean-Squared-Error [[pytorch](https://pytorch.org), [custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn/losses)] [[docs](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)] [[code](opennn/losses/meanloss.py)]
- Mean-Absolute-Error [[pytorch](https://pytorch.org), [custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn/losses)] [[docs](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)] [[code](opennn/losses/meanloss.py)]

### Metrics <a name="metrics"></a>
- Accuracy [[custom](https://github.com/Pe4enIks/OpenNN/tree/main/opennn/metrics)] [[docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)] [[code](opennn/metrics/accuracy.py)]
- Precision [[sklearn](https://scikit-learn.org/stable/)] [[docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)] [[code](opennn/metrics/precision.py)]
- Recall [[sklearn](https://scikit-learn.org/stable/)] [[docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)] [[code](opennn/metrics/recall.py)]
- F1 [[sklearn](https://scikit-learn.org/stable/)] [[docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)] [[code](opennn/metrics/f1_score.py)]

### Optimizers <a name="optimizers"></a>
- Adam [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)] [[code](opennn/optimizers/adam.py)]
- AdamW [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW)] [[code](opennn/optimizers/adam.py)]
- Adamax [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html#torch.optim.Adamax)] [[code](opennn/optimizers/adam.py)]
- RAdam [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam)] [[code](opennn/optimizers/adam.py)]
- NAdam [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam)] [[code](opennn/optimizers/adam.py)]

### Schedulers <a name="schedulers"></a>
- StepLR [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)] [[code](opennn/schedulers/steplr.py)]
- MultiStepLR [[pytorch](https://pytorch.org)] [[docs](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR)] [[code](opennn/schedulers/steplr.py)]

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

encoder = opennn.encoders.get_encoder(encoder_name, input_channels).to(device)
model = opennn.decoders.get_decoder(decoder_name, encoder, number_classes, decoder_mode, device).to(device)
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
  
train_data, valid_data, test_data = opennn.datasets.get_dataset(dataset_name, train_part, valid_part, transform)
```

4. Get optimizer.
```python
import opennn

optim_name = 'adam'
lr = 1e-3
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 1e-6
optimizer = opennn.optimizers.get_optimizer(optim_name, model, lr=lr, betas=betas, eps=opt_eps, weight_decay=weight_decay)
```

5. Get scheduler.
```python
import opennn

scheduler_name = 'steplr'
step = 10
gamma = 0.5
scheduler = opennn.schedulers.get_scheduler(sched, optimizer, step=step, gamma=gamma, milestones=None)
```

6. Get loss function.
```python
import opennn

loss_fn = 'custom_mse'
loss_fn, one_hot = opennn.losses.get_loss(loss_fn)
```

7. Get metrics functions.
```python
import opennn

metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
number_classes = 10
metrics_fn = opennn.metrics.get_metrics(metrics_names, nc=number_classes)
```

8. Train/Test.
```python
import opennn

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
  opennn.algo.train(train_dataloader, valid_dataloader, model, optimizer, scheduler, loss_fn, metrics_fn, epochs, checkpoints, logs, device, save_every, one_hot, number_classes)
elif algorithm == 'test':
  opennn.algo.test(test_dataloader, model, loss_fn, metrics_fn, logs, device, one_hot, number_classes)
  if viz:
    for _ in range(10):
      opennn.algo.vizualize(valid_data, model, device, {i: class_names[i] for i in range(number_classes)})
```

### Citation <a name="citing"></a>
Project [citation](https://github.com/Pe4enIks/OpenNN/blob/main/CITATION.ctf).

### License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/Pe4enIks/OpenNN/blob/main/LICENSE).