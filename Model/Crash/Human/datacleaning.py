# import pandas as pd
#
# # 读取 CSV 文件
# df = pd.read_csv('C:/Users/25303/Desktop/成都80/CD80_dataset/Human-Driving and AV Crash Data/Human Driver Crash Datasets/NHTSA/2016/CRSS2016CSV/PERSON.CSV')  # 替换为你的文件名
#
# # 删除 SCH_BUS 列中值为 1 的行
# df = df[df['SCH_BUS'] != 1]
#
# # 将结果保存为新的 CSV 文件
# df.to_csv('C:/Users/25303/Desktop/成都80/CD80_dataset/Human-Driving and AV Crash Data/Human Driver Crash Datasets/NHTSA/2016/CRSS2016CSV/PERSON1.CSV', index=False)  # 替换为你想要的输出文件名
# import pandas as pd
#
# # 读取 CSV 文件
# df = pd.read_csv('C:/Users/25303/Desktop/成都80/CD80_dataset/Human-Driving and AV Crash Data/Human Driver Crash Datasets/NHTSA/2016/CRSS2016CSV/PERSON1.CSV')  # 替换为你的文件名
#
# # 删除 PER_NO 列中值大于等于 2 的行
# df = df[df['PER_NO'] < 2]
#
# # 将结果保存为新的 CSV 文件
# df.to_csv('C:/Users/25303/Desktop/成都80/CD80_dataset/Human-Driving and AV Crash Data/Human Driver Crash Datasets/NHTSA/2016/CRSS2016CSV/PERSON2.CSV', index=False)  # 替换为你想要的输出文件名

# import pandas as pd
#
# # 读取 CSV 文件
# df = pd.read_csv('C:/Users/25303/Desktop/成都80/CD80_dataset/Human-Driving and AV Crash Data/Human Driver Crash Datasets/NHTSA/2016/CRSS2016CSV/PERSON1.CSV')  # 替换为你的文件名
#
# # 选择要保留的列
# # 这里我们使用列索引（0-索引），所以1-6列是索引0-5，25、26、28列是索引24、25、27
# selected_columns = df.iloc[:, list(range(6)) + [24, 25, 27]]
#
# # 将结果保存为新的 CSV 文件
# selected_columns.to_csv('C:/Users/25303/Desktop/成都80/CD80_dataset/Human-Driving and AV Crash Data/Human Driver Crash Datasets/NHTSA/2016/CRSS2016CSV/PERSON2.CSV', index=False)  # 替换为你想要的输出文件名

# import pandas as pd
#
# # 读取 CSV 文件
# df = pd.read_csv('C:/Users/25303/Desktop/成都80/CD80_dataset/Human-Driving and AV Crash Data/Human Driver Crash Datasets/NHTSA/2016/CRSS2016CSV/PERSON.CSV')  # 替换为你的文件名
#
# # 使用条件替换 INJ_SEV 列的值
# df['INJ_SEV'] = df['INJ_SEV'].replace({0: 1, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4, 9: 4})
#
# # 将结果保存为新的 CSV 文件
# df.to_csv('C:/Users/25303/Desktop/成都80/CD80_dataset/Human-Driving and AV Crash Data/Human Driver Crash Datasets/NHTSA/2016/CRSS2016CSV/PERSON1.CSV', index=False)  # 替换为你想要的输出文件名
import pandas as pd

# 1. 加载数据
file_path = 'C:/Users/25303/Desktop/成都80/CD80_dataset/Human-Driving and AV Crash Data/Human Driver Crash Datasets/NHTSA/2016/CRSS2016CSV/PERSON2.CSV'  # 替换为您的文件路径
data = pd.read_csv(file_path)

# 2. 显示原始数据的形状
print("原始数据形状:", data.shape)

# 3. 删除不符合条件的行
filtered_data = data[(data.iloc[:, 0] >= 18) & (data.iloc[:, 0] <= 75) & (data.iloc[:, 1].isin([1, 2]))]

# 4. 显示过滤后的数据的形状
print("过滤后的数据形状:", filtered_data.shape)

# 5. 保存过滤后的数据
filtered_data.to_csv('PERSON3.CSV', index=False)  # 可以替换为所需的输出文件路径
