import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 数据已经加载到DataFrame中
data = pd.read_csv('./data/insurance_claims_onehot.csv')
# 计算目标属性与所有其他属性的相关系数
correlation_with_Y = data.corr()['total_claim_amount'].drop('total_claim_amount')

random_seed = 78154213

# 选择特征和目标变量
X = data.drop('total_claim_amount', axis=1)
y = data['total_claim_amount']

# 训练随机森林模型
rf = RandomForestRegressor(n_estimators=10, random_state=random_seed)

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)
# 初始化KFold交叉验证器
kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)
# 用于存储每次交叉验证的性能指标
mse_scores = []
r2_scores = []
train_r2_scores = []

# 执行K折交叉验证
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 训练模型
    rf.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = rf.predict(X_test)
    y_train_pred = rf.predict(X_train)

    # 计算性能指标
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    train_r2 = r2_score(y_train_pred, y_train)

    # 存储性能指标
    mse_scores.append(mse)
    r2_scores.append(r2)
    train_r2_scores.append(train_r2)

# rf.fit(X_train, y_train)
#
# # 进行预测
# y_pred = rf.predict(X_test)
# # 在训练集上进行预测
# y_train_pred = rf.predict(X_train)

# 计算平均性能指标
mean_mse = sum(mse_scores) / len(mse_scores)
mean_r2 = sum(r2_scores) / len(r2_scores)
mean_train_r2 = sum(train_r2_scores) / len(train_r2_scores)

print(f"Mean Squared Error (MSE): {mean_mse}")
print(f"R^2 Score: {mean_r2}")
print(f"Train R^2 Score: {mean_train_r2}")