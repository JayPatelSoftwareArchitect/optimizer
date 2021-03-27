
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

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 600, 5)
#         self.pool = nn.MaxPool2d(3, 3)
#         self.conv2 = nn.Conv2d(600, 64*4, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
from test_m import Net

net = Net()
from test_op import Test_OP
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.9, momentum=0.09)
optimizer = Test_OP(net.parameters() )
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.09)

# from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('runs/fashion_mnist_experiment_1')

dataiter = iter(trainloader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
# writer.add_image('four_fashion_mnist_images', img_grid)
# writer.add_graph(net, images)

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

# log embeddings
features = images.view(-1, 28 * 28)
# writer.add_embedding(features,
#                     metadata=class_labels,
#                     label_img=images.unsqueeze(1))
# writer.close()


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
# For epoch 1:
#SGD:
# training loss 2.302158164978027
# training loss 2.2998192558288575
# training loss 2.292562749862671
# training loss 2.287133439540863
# training loss 2.2726018540859223
# training loss 2.234312132358551
# training loss 2.0445919197797777
# training loss 1.431033306211233
# training loss 1.0104906366541981
# training loss 0.9008196279257535
# training loss 0.8370124964583665
# training loss 0.7967864256612956
# training loss 0.7708656269572676
# training loss 0.7684128242842853
# training loss 0.7023762980252505
# testing loss 0.7415284909904003
# testing loss 0.7127044718787074
# Test_OP:
# training loss 1.4146121299117804
# training loss 1.0515470182970166
# training loss 0.8997676745802164
# training loss 0.9210348175950348
# training loss 0.8437173630087637
# training loss 0.796096006847336
# training loss 0.8017516585960984
# training loss 0.7297586165063549
# training loss 0.7149010808631429
# training loss 0.7119893821496517
# training loss 0.6849273890582844
# training loss 0.6501740739011147
# training loss 0.6347336277334252
# training loss 0.6395617391596315
# training loss 0.6303229261846282