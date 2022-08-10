import torch


def prediction(dataset, model, device, class_name, save_path, ind):
    '''
    Predict result for one image.

    Parameterts
    -----------
    dataset : list[torch.tensor, int]
        dataset.

    device : str
        device ['cpu', 'cuda'].

    class_name : list[str]
        name for all classes.

    save_path : str
        path to save output file

    ind : int
        index of image in dataset
    '''
    model.eval()
    img = dataset[ind][0]
    label = dataset[ind][1]
    img = img.unsqueeze(0).to(device)
    pred = torch.argmax(model(img), dim=1).cpu()[0]

    with open(save_path + '.log', 'w+') as in_f:
        in_f.write(f'predict class {class_name[pred.item()]}\n')
        in_f.write(f'correct class {class_name[label]}\n')
