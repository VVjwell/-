import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.sequential(x)


"""
当 train=True 时，self.mnist 加载 MNIST 的训练集，self.images 的形状是 (60000, 28, 28)，self.labels 的形状是 (60000,)。
当 train=False 时，加载测试集，self.images 的形状是 (10000, 28, 28)，self.labels 的形状是 (10000,)。
"""


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
        # img 的形状是 (batch_size, 1, 28, 28)（MNIST 图像，批次大小为 128，单通道，28x28 像素）
        # label 的形状是 (batch_size,)，包含每个图像的类别标签（0 到9
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
            # output(batch_size,num_classes)
            # (128, 10) 表示一个批次有 128 个样本，每个样本有 10 个类别的得分（logits，未经 softmax）
            output = model(img)
            loss = criterion(output, label)
            # total_loss是整个test_dataset的损失
            # img.size(0) = batch_size
            # loss.item() 是批次平均损失 loss.item() * img.size(0) 计算该批次中所有样本的总损失（未平均）。
            total_loss += loss.item() * img.size(0)
            """经过softmax归一化的版本将原始得分（logits）转换为概率分布（每个类别的概率和为 1）"""
            prob = F.softmax(output, dim=1)
            # torch.max(input,dim)返回(values,indices)(最大值张量,最大张量的索引张量)
            # output(128,10)每行表示一个样本在各个类别上的“得分”（通常是 logits，Logits 通常是全连接层或卷积层的输出，未经过任何归一化处理。未经 softmax 归一化）。
            # 用dim=1得到的values是每个样本的最大得分,indices表示每个样本最大得分对应的类别索引(预测标签)
            # nn.CrossEntropyLoss 作为损失函数，它期望输入是原始 logits，而不是 softmax 后的概率。
            # 原因：nn.CrossEntropyLoss 内部结合了 log_softmax 和负对数似然损失（NLLLoss），直接处理 logits：
            _, predicted = torch.max(prob, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        # len(test_loader) 是测试集的批次数量.10000 / 128 ≈ 78.125，向上取整为 79
        # 测试集的总样本数len(test_loader.dataset
        avg_loss = total_loss / len(test_loader.dataset)
        acc = 100 * correct / total
        print(f'Test  :accuracy:{acc:.2f}%, avg_loss:{avg_loss:.4f}')


def main():
    transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = MNISTDataset(train=True, transform=transform, root='./c1w2/dataset/mnist')
    # print(len(train_dataset),len(train_dataset.labels),len(train_dataset.images))
    test_dataset = MNISTDataset(train=False, transform=transform, root='./c1w2/dataset/mnist')
    # print(len(test_dataset),len(test_dataset.labels),len(test_dataset.images))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model()
    callback = Callback()
    epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(epochs=epochs, optimizer=optimizer, train_loader=train_loader, criterion=criterion, device=device,
          model=model, test_loader=test_loader, callback=callback)


if __name__ == '__main__':
    main()
