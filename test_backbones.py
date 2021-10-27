#Use the Backbones on Cifar-10
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.resnet18 import ResNet18
from models.resnet12 import ResNet12
from models.wrn import WRN
from models.convnet import ConvNet
import time
import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"		 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# set logging
logdir = "./logs/res18"
if not os.path.exists(logdir):
        os.makedirs(logdir)
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(logdir, 'log.txt')), logging.StreamHandler(os.sys.stdout)])
logger = logging.getLogger('main')

#Time calculate
begin = time.time()
localtime = time.asctime( time.localtime(begin) )
logger.info('Train begin at: %s', localtime)

#check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set hyperparameter
EPOCH = 20
pre_epoch = 0
BATCH_SIZE = 1024
LR = 0.01

#prepare dataset and preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# labels in CIFAR10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define ResNet18
net = ResNet18().to(device)

# define ResNet12
# net = ResNet12().to(device)

# define WRN
# net = WRN().to(device)

# define ConvNet
# net = ConvNet().to(device)

#define loss funtion & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

#train
for epoch in range(pre_epoch, EPOCH):
    logger.info('Epoch: %d', epoch + 1)
    # print('\nEpoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(trainloader, 0):
        #prepare dataset
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        #forward & backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #print ac & loss in each batch
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        
    correct = float(correct)
    print(correct, total)
    
    logger.info('Train\'s ac is: %f' , float(correct) / total)

    #get the ac with testdataset in each epoch
    with torch.no_grad():
        correct = 0
        total = 0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        logger.info('Test\'s ac is: %f', float(correct) / total)

logger.info('Train has finished, total epoch is %d' % EPOCH)

# Time calculate
time_cost = time.time() - begin
logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_cost // 60, time_cost % 60))