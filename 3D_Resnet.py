import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import os

class mydatasets(Dataset):
    def __init__(self, data,label):
        self.data = data  # 加上通道数
        self.label = label

    def __getitem__(self, index):
        data = self.data[index]  # 获取高阶FCN
        label = self.label[index]
        return data,label
    def __len__(self):
        return self.data.shape[0]  # 返回数据集的长度


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二个卷积层
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels * self.expansion)

        # 残差连接（shortcut connection）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels * self.expansion),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = 64

        # 第一个卷积层
        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet的四个阶段
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)

        # 全局平均池化层和全连接层
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def data_load(data_path,batch_size):

        data_label = np.load(data_path,batch_size)

        torch.manual_seed(9)
        random_index = np.random.permutation(data_label['data'].shape[0])
        data = torch.from_numpy(data_label['data'][random_index]).float()
        label= torch.from_numpy(data_label['label'][random_index]).float()

        train_data = data[:160]
        train_label = label[:160]
        test_data = data[160:]
        test_label = label[160:]

        train_dataset = mydatasets(train_data,train_label)
        test_dataset = mydatasets(test_data,test_label)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)  # 创建训练数据加载器
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)  # 创建测试数据加载器

        return train_loader,test_loader


def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 设置模型为训练模式
    train_loss = 0
    for data, label in train_loader:
        data = data.to(device)
        data = torch.unsqueeze(data,1)
        optimizer.zero_grad()  # 清除梯度
        output = model(data)  # 前向传播
        loss = criterion(output, label.to(device).long())  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        train_loss += loss.item() * data.size(0)

    train_loss /= len(train_loader.dataset)  # 计算平均训练损失
    return train_loss

def validate(model, val_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    val_loss = 0 
    correct = 0 #正确个数
    total = 0 #总数
    with torch.no_grad():
        for data, label in val_loader:
            data = data.to(device)
            data = torch.unsqueeze(data,1)
            output = model(data)  # 前向传播

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label.to(device)).sum().item()
            loss = criterion(output, label.to(device).long())  # 计算损失
            val_loss += loss.item() * data.size(0)

    accuracy = 100 * correct / total
    # print('Accuracy on the test set: %d %%' % accuracy)           
    val_loss /= len(val_loader.dataset)  # 计算平均验证损失
    return accuracy,val_loss 

if __name__ == "__main__":
    epoch_times = 100
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader,test_loader = data_load('/home/yeshixin/work/newwork/DDPM-main/data/mri_ad90_cn113_data_label_normal.npz',batch_size)
    model = ResNet18(2)
    model.to(device)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    best_val_acc = 0
    if not os.path.exists('ckpt'):
        os.mkdir('./ckpt')
    # 训练模型
    for epoch in range(epoch_times):
        train_loss = train(model, train_loader, criterion, optimizer, device)  # 训练模型
        val_acc,val_loss = validate(model, test_loader, criterion, device)  # 验证模型
        train_losses.append(train_loss)  # 保存训练损失
        val_losses.append(val_loss)  # 保存验证损失
        # 存储最小损失模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            torch.save(best_model, 'ckpt/BestLoss_'+str(best_val_loss)+'_model.ckpt')  # 保存最佳模型参数
            print("best_val_loss: " + str(best_val_loss))
            with open("ckpt/model_loss.txt", "w") as f:
                f.write(str(val_loss))
        # 存储最大准确率模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            torch.save(best_model, 'ckpt/BestAcc_'+str(best_val_acc)+'_model.ckpt')  # 保存最佳模型参数
            print("best_val_acc: " + str(best_val_acc))
            with open("ckpt/model_acc.txt", "w") as f:
                f.write(str(best_val_acc))

        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f} %'.format(epoch+1, epoch_times, train_loss, val_loss,val_acc))