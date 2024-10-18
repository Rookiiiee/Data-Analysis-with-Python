import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 512)  # 输入层到隐藏层，CIFAR-10 图像大小是32x32x3
        self.fc2 = nn.Linear(512, 10)       # 隐藏层到输出层，10个类别

    def forward(self, x):
        x = x.view(-1, 32*32*3)  # 展平图像
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 适用于3个颜色通道
])

# 下载和加载训练数据
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 下载和加载测试数据
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
def train_model(model, data_loader, epochs):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

model = CIFAR10Classifier()
train_model(model, train_loader, 10)  # 训练10个epoch
# 保存模型
torch.save(model.state_dict(), 'cifar10_classifier.pth')

# 加载模型
loaded_model = CIFAR10Classifier()
loaded_model.load_state_dict(torch.load('cifar10_classifier.pth'))
loaded_model.eval()

# 验证加载的模型
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = loaded_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')
