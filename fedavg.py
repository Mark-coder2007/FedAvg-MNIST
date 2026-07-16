import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset  # 从模块中取名字，不用加前缀
from torchvision import datasets, transforms

# --超参数部分--
NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS = 5  # E越大，每轮通信本地端算的越多，通信轮数越少
LOCAL_BATCH_SIZE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ROUNDS = 100
LR = 0.01
IID = True
SEED = 42
# 三元表达式
# 功能上，torch.cuda.is_available() 检测有没有 NVIDIA 显卡，有就用 GPU 训练


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
# 把 Python、NumPy、PyTorch 三套随机数发生器全部固定，保证每次运行结果一致

# --CNN模型--


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # relu引入非线性，先卷积，再ReLU，再池化
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))  # 没有接 softmax，直接输出原始得分
        return self.fc2(x)

# --数据划分--


def partition_iid(dataset, num_clients):
    idxs = np.random.permutation(len(dataset))
    return np.array_split(idxs, num_clients)
# 每个客户端拿到的数字分布都近似整体分布
# IID（独立同分布）


def partition_noniid(dataset, num_clients, shards_per_client=2):
    labels = np.array(dataset.targets)
    idxs = np.argsort(labels)  # np.argsort 返回"能让标签有序的索引"
    num_shards = num_clients*shards_per_client
    # 100 个客户端 × 每人 2 片 = 200 个 shard，每片 300 张图
    shard_size = len(dataset)//num_shards
    shards = [idxs[i*shard_size:(i+1)*shard_size]
              for i in range(num_shards)]
    # 把排好序的索引切成 200 片。因为切之前按标签排过序，每一片内几乎只含一种数字
    random.shuffle(shards)
    return [np.concatenate(shards[i*shards_per_client:(i+1)*shards_per_client])
            for i in range(num_clients)]


# --本地训练--
def local_train(global_model, dataset, idxs):
    model = copy.deepcopy(global_model).to(DEVICE)
    model.train()
    loader = DataLoader(Subset(dataset, list(idxs)),
                        batch_size=LOCAL_BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for _ in range(LOCAL_EPOCHS):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    return model.state_dict(), len(idxs)


# --FedAvg聚合--
def fed_avg(state_dicts, sizes):
    total = sum(sizes)
    avg = copy.deepcopy(state_dicts[0])
    for key in avg.keys():
        avg[key] = avg[key]*(sizes[0]/total)
        for sd, n in zip(state_dicts[1:], sizes[1:]):
            avg[key] += sd[key]*(n/total)
    return avg

# --评估--


@torch.no_grad()  # 让函数内部不构建计算图
def evaluate(model, loader):
    model.eval()  # 切换到评估模式
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(dim=1)
        # model(images) 前向传播,得到得分矩阵
        # "沿着 dim=1 的方向找最大值的位置",也就是每行内部横着扫一遍
        correct += (preds == labels).sum().item()
        # item把张量变成普通数字2
        total += labels.size(0)
    return correct/total


# --主流程--
def main():
    set_seed(SEED)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(
        "./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False,
                              download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256)

    if IID:
        client_idxs = partition_iid(train_set, NUM_CLIENTS)
    else:
        client_idxs = partition_noniid(train_set, NUM_CLIENTS)

    global_model = CNN().to(DEVICE)

    for rnd in range(1, NUM_ROUNDS + 1):
        selected = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)

        state_dicts, sizes = [], []
        for cid in selected:
            sd, n = local_train(global_model, train_set, client_idxs[cid])
            state_dicts.append(sd)
            sizes.append(n)

        global_model.load_state_dict(fed_avg(state_dicts, sizes))

        acc = evaluate(global_model, test_loader)
        print(f"Round {rnd:3d} | test acc = {acc:.4f}")


if __name__ == "__main__":
    main()