# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 19:23:59 2023
该代码负责构造新特征和筛选
@author: Administrator
"""
##特征构造，并给出可能的公式(输出一个Excel文件包含公式)
import numpy as np
import pandas as pd
import sympy
from pysr import PySRRegressor

data = pd.read_csv(r"test.csv", encoding ='gb2312')
data = data.iloc[:, 1:]

data.drop_duplicates(inplace=True)
data.index = range(data.shape[0])  #恢复打乱的索引


X = data.iloc[:,1:]
y = data.iloc[:,0]
# y = y / 100   #对硬度数据的处理，因为硬度量级比成分高 


model = PySRRegressor(
    niterations=1000,  # < Increase me for better results
    populations=8,
    # ^ Alternatively, stop after 24 hours have passed.
    maxsize=100,
    # ^ Allow greater complexity.
    maxdepth=20,
    precision=64,
    # ^ Higher precision calculations.
    binary_operators=["+", "-", "*", "/","^"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "cube",
        "log10",
        "sqrt",
        "inv(x) = 1/x",

        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    denoise=True,
    # select_k_features=5,
    # ^ Define operator for SymPy as well
    # loss="loss(x, y) = abs(x - y)^0.5",
    loss= "LPDistLoss{3}()",
    # ^ Custom loss function (julia syntax)
)

model.fit(X, y)

print(model)
print(model.equations_)
model.equations_.to_csv("SRequations.csv") #输出公式


##将构造的特征存入原数据集(将原数据集和计算后的数据整合到同一个csv文件中)
import pandas as pd
from sympy import *

# 读取第一个数据集
df1 = data

# 读取第二个数据集
df2 = pd.DataFrame(model.equations_["sympy_format"])

# 将第二个数据集中的公式转换为可计算的Lambda函数
var_map = {col: Symbol(col) for col in df1.columns}
for idx, row in df2.iterrows():
    formula = lambdify(var_map.keys(), sympify(row['sympy_format']))
    df2.at[idx, 'formula'] = formula

# 计算公式并将结果添加到第一个数据集中
for idx, row in df1.iterrows():
    for idx2, row2 in df2.iterrows():
        formula = row2['formula']
        if callable(formula):
            result = formula(*row)
            # df1.at[idx, str(row2['sympy_format'])] = result
            df1.at[idx, f"result_{idx2}"] = result
        else:
            break
    else:
        continue
    break

# 将结果保存到新的Excel文件中
df1.to_csv("Addequations.csv")