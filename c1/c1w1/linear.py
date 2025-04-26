import torch
import numpy as np
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        """
        PyTorch 的神经网络模块（如 nn.Linear）设计为批量处理数据，批次维度（batch_size）允许并行处理多个样本。
        期望输入的x维度是形状为 (batch_size, in_features)
        即使你只处理一个样本（batch_size=1），仍需提供二维张量，形状为 (1, 1)，以符合 PyTorch 的输入规范。
        """
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        # 定义一维np数组,形状为(9,) 第一维度长度9
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
        y = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
        """"
        tensor.unsqueeze(dim)在张量的dim维度增加一个维度,大小为1.返回一个新的张量，原始张量保持不变（除非使用 unsqueeze_ 原地操作）
                                   0维   1维  2维度...
                                   |     |   |
                                   v     v   v
         dim = 1插入后是[[1],[2]...](9,              )->(9,1)
         dim = 0插入后是[[1,2,3,4...]](9,)->(1,9)
         """
        self.x = torch.tensor(x).unsqueeze(1)
        self.y = torch.tensor(y).unsqueeze(1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def train(model, dataset, optimizer, epochs, criterion, device, dataloader):
    model.to(device)
    model.train(True)  # 设置训练
    # epochs是int不可迭代,要换成range()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)  # 累计损失
            avg_loss = epoch_loss / len(dataset)
        # y_pred = model.forward(dataset)
        # 这里dataset是整个数据集,并不是张量无法计算.
        # loss = criterion(y_pred, dataset)
        print(f'我的损失Epoch: {epoch}, Loss: {loss.item():.4f},avg_loss:{avg_loss:.4f}')


def main():
    model = Model()
    dataset = Dataset()
    # 如果改成lr = 0.1 会梯度爆炸,训练的loss变成nan,inf
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    epochs = 20
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model, dataset, optimizer, epochs, criterion, device, dataloader)
    print('Finished Training')
    # Model.forward(20)是类方法,不是实例方法要用model调用,要用eval
    # print(model.forward(20))
    model.eval()
    with torch.no_grad():
        inputdata = torch.tensor([20.0], device=device, dtype=torch.float32)
        pred = model(inputdata)
        print(f'20预测值为{pred}')


if __name__ == '__main__':
    main()
