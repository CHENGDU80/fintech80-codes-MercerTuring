import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. 加载数据
data = pd.read_csv('20240208_120000_vld.csv')

# 特征在前面几列，标签在最后一列
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 标签编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 2. 数据预处理
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 3. 转换为PyTorch的张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.LongTensor(y_val)

# 4. 定义改进后的神经网络模型
class ImprovedNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 5. 实例化模型、损失函数和优化器
input_size = X_train.shape[1]
output_size = len(label_encoder.classes_)
model = ImprovedNN(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. 训练模型
num_epochs = 250
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # 记录训练损失
    train_losses.append(loss.item())

    # 验证集监控
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())
        _, val_predicted = torch.max(val_outputs, 1)
        val_accuracy = (val_predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
        val_accuracies.append(val_accuracy)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

# 7. 绘制损失和准确率曲线
plt.figure(figsize=(12, 5))

# 绘制训练和验证损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# 绘制验证准确率曲线
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()

# 8. 测试模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_val_tensor)  # 如果有测试集请替换为测试集张量
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
    print(f'Final Accuracy: {accuracy:.4f}')
torch.save(model.state_dict(), 'HMDriveriskmodel_weights.pth')