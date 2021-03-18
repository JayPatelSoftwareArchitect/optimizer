
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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

    optimizer.reset_after_epoch() #resets state 
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
# training loss 1.6338135780394076
# training loss 1.1190108029693364
# training loss 1.0901973182149232
# training loss 0.9108199938144534
# training loss 0.920561436627584
# training loss 0.8430368340644054
# training loss 0.8371156890373677
# training loss 0.7873017362472019
# training loss 0.7860973827544949
# training loss 0.7633222482840393
# training loss 0.7419457610772224
# training loss 0.6938450459787855
# training loss 0.6886203742609359
# training loss 0.716971114214306
# training loss 0.7098038397308556
#  Current epoch accuracy: tensor(67.7783)
# testing loss 0.8072343270705314
# testing loss 0.7781512646827614
#  Current epoch accuracy: tensor(72.5500)
# training loss 1.0874489933837321
# training loss 0.7517717505764449
# training loss 0.6850757514339202
# training loss 0.6545005107618636
# training loss 0.6652295287948801
# training loss 0.6872926002801396
# training loss 0.6801109503376574
# training loss 0.6617041466127266
# training loss 0.6565554857996467
# training loss 0.6626002474847482
# training loss 0.6660469573085429
# training loss 0.6379003160682624
# training loss 0.638073895997979
# training loss 0.6479522319560056
# training loss 0.6393748551502358
#  Current epoch accuracy: tensor(76.6867)
# testing loss 0.7169805774148553
# testing loss 0.6848432531358849
#  Current epoch accuracy: tensor(75.5500)
# training loss 0.963078390502982
# training loss 0.653149830960232
# training loss 0.6281752541588503
# training loss 0.6476071642671013
# training loss 0.6336052754438715
# training loss 0.6182231155231711
# training loss 0.6140982484028209
# training loss 0.6088366661001201
# training loss 0.6169185366351158
# training loss 0.5964793686287594
# training loss 0.6193688545185432
# training loss 0.6255622619868955
# training loss 0.6204389083062124
# training loss 0.5678655926780776
# training loss 0.629163765217585
#  Current epoch accuracy: tensor(78.9983)
# testing loss 0.5968382239678176
# testing loss 0.5724192007627571
#  Current epoch accuracy: tensor(80.1900)
# training loss 0.9285543742862937
# training loss 0.5601823847278138
# training loss 0.5926299821456196
# training loss 0.5896249209931702
# training loss 0.6109274336606031
# training loss 0.5707006000375259
# training loss 0.5795952428242017
# training loss 0.562569748197362
# training loss 0.5986895466134592
# training loss 0.5894144681801845
# training loss 0.5713553309109412
# training loss 0.577073297986819
# training loss 0.5530529673647834
# training loss 0.5815791349339852
# training loss 0.6175166819450678
#  Current epoch accuracy: tensor(80.4833)
# testing loss 0.6282445477095607
# testing loss 0.6044744485133852
#  Current epoch accuracy: tensor(78.6900)
