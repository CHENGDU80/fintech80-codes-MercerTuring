import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import re

def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def split_csl(df, csl_column):
    # 拆分 policy_csl 列
    try:
        df[csl_column + '_lower'] = df[csl_column].str.split('/').str[0].astype(int)
        df[csl_column + '_upper'] = df[csl_column].str.split('/').str[1].astype(int)
    except Exception as e:
        print(f"Error in split_csl: {e}")
    df.drop(csl_column, axis=1, inplace=True)
    return df

def split_date(df, date_column):
    # 拆分日期列
    try:
        df[date_column + '_year'] = pd.to_datetime(df[date_column]).dt.year
        df[date_column + '_month'] = pd.to_datetime(df[date_column]).dt.month
        df[date_column + '_day'] = pd.to_datetime(df[date_column]).dt.day
    except Exception as e:
        print(f"Error in split_date: {e}")
    df.drop(date_column, axis=1, inplace=True)
    return df

def data_processor(df):
    # 预先处理 policy_csl 列
    df = split_csl(df, 'policy_csl')

    # 填补空缺值
    df.fillna(-1, inplace=True)

    # 分割日期并创建新列
    date_columns = ['policy_bind_date', 'incident_date']
    for date_column in date_columns:
        df = split_date(df, date_column)

    # 数值特征和分类特征的分离
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # 排除总赔付金额，不做归一化
    if 'total_claim_amount' in numerical_features:
        numerical_features = numerical_features.drop('total_claim_amount')
    if 'incident_location' in categorical_features:
        categorical_features = categorical_features.drop('incident_location')

    # 将分类特征转换为字符串类型
    for col in categorical_features:
        df[col] = df[col].astype(str)

    # 填补分类特征中的缺失值，使用 loc 方法避免警告
    for col in categorical_features:
        df.loc[df[col].isna(), col] = 'missing'  # 使用 loc 来避免 SettingWithCopyWarning

    # 数值特征归一化
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # 分类特征OneHot编码
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_categorical = encoder.fit_transform(df[categorical_features])
    encoded_categorical_df = pd.DataFrame(encoded_categorical.toarray(),
                                           columns=encoder.get_feature_names_out(input_features=categorical_features))

    # 合并数值特征和编码后的分类特征
    df_preprocessed = pd.concat([df[numerical_features], df[['total_claim_amount']], encoded_categorical_df], axis=1)

    return df_preprocessed

# 读取数据
df = pd.read_csv('./data/insurance_claims.csv')

# 调用数据处理器函数
df_preprocessed = data_processor(df)

# 保存预处理后的数据
df_preprocessed.to_csv('./data/insurance_claims_onehot.csv', index=False)
