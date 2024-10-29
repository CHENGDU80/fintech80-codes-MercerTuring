import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from joblib import dump, load

# 定义一个自定义的数据集
class CSVDataset(Dataset):
    def __init__(self, csv_file, features, target):
        # 加载数据
        self.data_frame = pd.read_csv(csv_file)
        # 提取特征和目标列
        self.features = self.data_frame[features].values
        self.target = self.data_frame[target].values

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 128)        # 隐藏层到隐藏层
        self.fc3 = nn.Linear(128, 1)         # 隐藏层到输出层
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))       # 使用 Leaky ReLU
        x = self.leaky_relu(self.fc2(x))       # 使用 Leaky ReLU
        x = self.fc3(x)                         # 输出层
        return x

# 参数
csv_file = './data/insurance_claims_onehot.csv'  # 你的CSV文件路径
features = [col for col in pd.read_csv(csv_file).columns if col != 'total_claim_amount']  # 特征列
target = 'total_claim_amount'  # 目标列

# 加载数据
dataset = CSVDataset(csv_file, features, target)
# loaded_model = load('oracle.joblib')
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=1454)

# 创建DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

# 初始化模型、损失函数和优化器
model = NeuralNetwork(input_size=len(features))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00025)

# 初始化列表以存储损失
losses = []

# 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    epoch_loss = 0  # 每个epoch的总损失
    for inputs, labels in train_loader:
        optimizer.zero_grad()   # 清零梯度
        outputs = model(inputs.float())  # 前向传播
        loss = criterion(outputs, labels.float().view(-1, 1))  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        epoch_loss += loss.item()  # 累加损失

    avg_loss = epoch_loss / len(train_loader)  # 计算平均损失
    losses.append(avg_loss)  # 存储平均损失
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')

# 测试模型
model.eval()  # 设置为评估模式
all_predictions = []
all_targets = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.float())
        all_predictions.extend(outputs.view(-1).tolist())
        all_targets.extend(labels.view(-1).tolist())

# 计算R^2
r2 = r2_score(all_targets, all_predictions)
print(f'R^2 on test set: {r2}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss', color='blue')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# new_sample = [[100000]]
# predicted_premium = loaded_model.predict(new_sample)
# print(f'Predicted policy annual premium for new sample: {predicted_premium[0]}')