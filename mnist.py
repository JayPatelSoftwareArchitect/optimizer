
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Testing optimizer on mnist dataset.
# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

from test_m import Net

net = Net()
from test_op import Test_OP
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.9, momentum=0.09)
optimizer = Test_OP(net.parameters() )
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.09)


dataiter = iter(trainloader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)


def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

features = images.view(-1, 28 * 28)

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))

    return fig
running_loss = 0.0
for epoch in range(30):  # loop over the dataset multiple times
    total_correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        predictions = torch.max(outputs, 1)[1]
        train_acc = torch.sum(predictions == labels)
        # if outputs == labels:
        total_correct += train_acc
        total += 4
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step(loss)

        running_loss += loss.item()
        # scheduler.step()
        if i % 1000 == 999:    # every 1000 mini-batches...

            # ...log the running loss
            print('training loss',
                            running_loss / 1000)
            print('accuracy ', total_correct * 100 / total)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            # print('predictions vs. actuals',
            #                 plot_classes_preds(net, inputs, labels),
            #                 global_step=epoch * len(trainloader) + i)
            running_loss = 0.0

    running_loss = 0.0
    accuracy = total_correct * 100 / total
    total_correct = 0
    total = 0
    print(" Current epoch accuracy:",str(accuracy))
    for i, data in enumerate(testloader, 0):

        inputs, labels = data
        outputs = net(inputs)
        predictions = torch.max(outputs, 1)[1]
        train_acc = torch.sum(predictions == labels)
        total_correct += train_acc
        total += 4
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        if i % 1000 == 999:
            print('testing loss',
                            running_loss / 1000)
            running_loss = 0.0
    accuracy = total_correct * 100 / total
    total_correct = 0
    total = 0
    print(" Current epoch accuracy:",str(accuracy))
print('Finished Training')

PATH = 'test_op_mnist_model'
torch.save(net.state_dict(), PATH)