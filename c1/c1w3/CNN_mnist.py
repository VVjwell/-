import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

"""
缺少现代化 CNN 特性：
未使用 BatchNorm 或 Dropout，可能影响训练稳定性或泛化能力。
激活函数单一（仅 ReLU），可以考虑替代方案。
"""


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        """
        H_in:height输入特征图的高度width是特征图宽度W_in
        CNN的输入和特征图形状(batch_size, channels, height, width)，简称 NCHW 格式
        默认stride = kernel_size例如这里都是2
        池化公式不直接依赖输入是否为奇数，而是通过 H_out = floor( (H_in - kernel_size) / stride + 1) 计算。
        """
        self.conv = nn.Sequential(
            # conv1后从1*28*28变成16*28*28
            # in_ch=RGB=1,out_ch=16,相当于16个卷积的filters.kernel为3*3,padding=1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # maxpool1后变成16*14*14
            nn.MaxPool2d(2),
            # conv2后32*14*14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # maxpool2后是32*7*7
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(32 * 7 * 7, 128),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root='./c1w2/dataset/mnist', transform=None, train=True):
        self.train = train
        self.transform = transform
        self.mnist = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=True
        )
        self.images = self.mnist.data  # (N,28,28)
        self.labels = self.mnist.targets  # (N,)

    def __getitem__(self, index):
        img, label = self.images[index], self.labels[index]
        img = img.unsqueeze(0).float() / 255.0  # 形状(1,28,28)归一化到[0,1]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)


class Callback():
    def __init__(self):
        self.stop_training = False

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.002 and logs.get('acc') > 94.0:
            print('loss so low,stopping training')
            self.stop_training = True


def train(epochs, model, train_loader, optimizer, criterion, device, test_loader, callback):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, pred = torch.max(output, dim=1)
            total += img.size(0)
            correct += torch.sum(pred == label).item()
        acc = 100 * correct / total
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f'Epoch:{epoch + 1}\nTrain :accuracy:{acc:.2f}%, avg_loss:{avg_loss:.4f}')
        if callback:
            callback.on_epoch_end(epoch, logs={'loss': avg_loss, 'acc': acc})
            if callback.stop_training:
                print("Training stopped by callback.")
                break
        model.eval()
        test(test_loader, model, criterion, device)


def test(test_loader, model, criterion, device):
    model.to(device)
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = criterion(output, label)
            total_loss += loss.item() * img.size(0)
            """经过softmax归一化的版本将原始得分（logits）转换为概率分布（每个类别的概率和为 1）"""
            prob = F.softmax(output, dim=1)
            _, predicted = torch.max(prob, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        avg_loss = total_loss / len(test_loader.dataset)
        acc = 100 * correct / total
        print(f'Test  :accuracy:{acc:.2f}%, avg_loss:{avg_loss:.4f}')


def main():
    transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNISTDataset(train=True, transform=transform, root='./c1w2/dataset/mnist')
    test_dataset = MNISTDataset(train=False, transform=transform, root='./c1w2/dataset/mnist')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNmodel()
    callback = Callback()
    epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(epochs=epochs, optimizer=optimizer, train_loader=train_loader, criterion=criterion, device=device,
          model=model, test_loader=test_loader, callback=callback)


if __name__ == '__main__':
    main()
