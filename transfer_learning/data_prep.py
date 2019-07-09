import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import OrderedDict
import matplotlib.pyplot as plt
from pathlib import Path

import helper


def get_data(data_dir=str(Path.home()) + '/data/cat_and_dog'):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader, testloader


if __name__ == '__main__':
    data_dir = str(Path.home()) + '/data/cat_and_dog'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor()])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    data_iter = iter(testloader)

    # images, labels = next(data_iter)
    # fig, axes = plt.subplots(figsize=(10, 4), ncols=4)
    # for ii in range(4):
    #     ax = axes[ii]
    #     helper.imshow(images[ii], ax=ax, normalize=False)
    # plt.show()

    print(len(trainloader))