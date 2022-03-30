import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def vizualize(dataset, model, device, class_name):
    model.eval()
    ind = random.randint(0, len(dataset) - 1)

    img = dataset[ind][0]
    label = dataset[ind][1]
    img = img.unsqueeze(0).to(device)
    pred = torch.argmax(model(img), dim=1).cpu()[0]

    if dataset[ind][0].shape[0] == 1:
        img = np.moveaxis(torch.cat((img[0], img[0], img[0]), dim=0).cpu().numpy(), [0, 1, 2], [2, 0, 1])
    else:
        img = np.moveaxis(img[0].cpu().numpy(), [0, 1, 2], [2, 0, 1])

    _ = plt.imshow(img)
    plt.show()
    print(f'Предсказанный класс {class_name[pred.item()]}')
    print(f'Правильный класс {class_name[label]}')
