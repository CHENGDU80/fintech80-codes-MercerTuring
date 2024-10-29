import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

# 读取CSV文件
df = pd.read_csv('./data/insurance_claims.csv')

# 选择需要的列
X = df[['total_claim_amount']]  # 输入特征
y = df['policy_annual_premium']  # 输出目标

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 创建神经网络回归模型
# 这里我们使用一个简单的两层网络，每层有10个神经元
model = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=10000, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 保存模型
dump(model, 'oracle.joblib')


# 使用模型进行预测
new_sample = [[100000]]  # 例如，total_claim_amount为100000
predicted_premium = model.predict(new_sample)
print(f'Predicted policy annual premium for new sample: {predicted_premium[0]}')