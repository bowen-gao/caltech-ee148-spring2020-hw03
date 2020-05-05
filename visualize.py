from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import ImageGrid
import os


class Net(nn.Module):
    '''
    Design your model with convolutional layers.
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1)
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout2d(0.1)
        self.bn = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(400, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = self.bn(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        embedding = x
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output, embedding


assert os.path.exists("mnist_model.pt")

if __name__ == '__main__':
    # Set the test model
    device = torch.device("cuda")
    model = Net().to(device)
    model.load_state_dict(torch.load("mnist_model.pt"))
    test_dataset = datasets.MNIST('../data', train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, **kwargs)
    model.eval()
    ims = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == "conv1.weight":
                for i in range(9):
                    img = param.detach().cpu()[i][0]
                    ims.append(img)


    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 3x3 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, ims):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap='gray')
    plt.title("learned kernels")
    plt.show()
    embeds = []
    targets = []
    wrong_preds=[]
    with torch.no_grad():  # For the inference step, gradient is not computed
        count = 0
        conf = [[0] * 10 for _ in range(10)]
        print(len(test_dataset[0]))
        for data, target in test_loader:
            targets.append(target.item())
            print(count)
            data, target = data.to(device), target.to(device)
            output, embedding = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            conf[target][pred] = conf[target][pred] + 1
            embedding = embedding.detach().cpu()[0]
            embeds.append((np.array(embedding)))
            if not pred.eq(target):
                img = data.detach().cpu()[0][0]
                wrong_preds.append(img)
                # plt.savefig("wrong_images/" + str(count))
            count += 1
    print(conf)
    np.savetxt("conf",conf)

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 3x3 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, wrong_preds):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap='gray')
    plt.title("misclassified images")
    plt.show()


    from sklearn.neighbors import NearestNeighbors

    embeds = np.array(embeds)
    nbrs = NearestNeighbors(n_neighbors=8, p=2, algorithm='ball_tree').fit(embeds)
    ims = []
    for i in range(4):
        ls = nbrs.kneighbors(embeds[i].reshape(1, -1), return_distance=False)
        for id in ls[0]:
            data = test_dataset[id][0]
            ims.append(data[0])
    from sklearn.manifold import TSNE

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(4, 8),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, ims):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap='gray')

    plt.show()

    embeds = TSNE(n_components=2).fit_transform(embeds)
    print(embeds.shape)

    X = embeds[:, 0]
    Y = embeds[:, 1]

    plt.clf()
    sc = plt.scatter(X, Y, c=targets, cmap=cm.get_cmap('Paired', 10))
    plt.legend(*sc.legend_elements())
    plt.title("2d visualization of embedding vectors")
    plt.show()
