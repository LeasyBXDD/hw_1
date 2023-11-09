# 导入需要的包
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入数据
table_b11 = pd.DataFrame({
  'Clarity': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.8, 0.7, 1, 0.9, 1, 1, 1, 0.9, 0.9, 1, 0.7, 0.7, 1, 1, 1, 1, 1, 1, 1, 0.8, 1, 1, 0.8, 0.8, 0.8, 0.8],
  'Aroma': [3.3, 4.4, 3.9, 3.9, 5.6, 4.6, 4.8, 5.3, 4.3, 4.3, 5.1, 3.3, 5.9, 7.7, 7.1, 5.5, 6.3, 5, 4.6, 3.4, 6.4, 5.5, 4.7, 4.1, 6, 4.3, 3.9, 5.1, 3.9, 4.5, 5.2, 4.2, 3.3, 6.8, 5, 3.5, 4.3, 5.2],
  'Body': [2.8, 4.9, 5.3, 2.6, 5.1, 4.7, 4.8, 4.5, 4.3, 3.9, 4.3, 5.4, 5.7, 6.6, 4.4, 5.6, 5.4, 5.5, 4.1, 5, 5.4, 5.3, 4.1, 4, 5.4, 4.6, 4, 4.9, 4.4, 3.7, 4.3, 3.8, 3.5, 5, 5.7, 4.7, 5.5, 4.8],
  'Flavor': [3.1, 3.5, 4.8, 3.1, 5.5, 5, 4.8, 4.3, 3.9, 4.7, 4.5, 4.3, 7, 6.7, 5.8, 5.6, 4.8, 5.5, 4.3, 3.4, 6.6, 5.3, 5, 4.1, 5.7, 4.7, 5.1, 5, 5, 2.9, 5, 3, 4.3, 6, 5.5, 4.2, 3.5, 5.7],
  'Oakiness': [4.1, 3.9, 4.7, 3.6, 5.1, 4.1, 3.3, 5.2, 2.9, 3.9, 3.6, 3.6, 4.1, 3.7, 4.1, 4.4, 4.6, 4.1, 3.1, 3.4, 4.8, 3.8, 3.7, 4, 4.7, 4.9, 5.1, 5.1, 4.4, 3.9, 6, 4.7, 4.5, 5.2, 4.8, 3.3, 5.8, 3.5],
  'Quality': [9.8, 12.6, 11.9, 11.1, 13.3, 12.8, 12.8, 12, 13.6, 13.9, 14.4, 12.3, 16.1, 16.1, 15.5, 15.5, 13.8, 13.8, 11.3, 7.9, 15.1, 13.5, 10.8, 9.5, 12.7, 11.6, 11.7, 11.9, 10.8, 8.5, 10.7, 9.1, 12.1, 14.9, 13.5, 12.2, 10.3, 13.2]
})
data = table_b11 

# 分割数据为训练集和测试集
X = data[['Clarity', 'Aroma', 'Body', 'Flavor', 'Oakiness']]
y = data['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y)

print("ex1:建立 $y$ 关于 $x_1, x_2, x_3. x_4, x_5$ 的回归模型,并对回归方程和回归系数进行显著性检验。")

# 建立回归模型
lr = LinearRegression().fit(X_train, y_train)

# 评估模型性能
print(f'回归方程的R^2值为:{lr.score(X_test, y_test)}')

print(f'Coeff: \n{lr.coef_}')
model = sm.OLS(y, X)
results = model.fit()
print("\n ex1显著性检验结果如下:\n", results.summary())

# 逐步回归
print("\nex2:采用逐步回归法建立 $y$ 关于 $x_1, x_2, x_3. x_4, x_5$ 的线性回归模型,并对回归方程和回归系数进行显著性检验。")
rfe = RFE(lr)
rfe = rfe.fit(X_train, y_train)
print(f'选出的特征为:{rfe.support_}')
print(f'Coeff: \n{rfe.estimator_.coef_}')

# 将X_train转换为一个只包含你想要的特征的数据框
X_train_selected = X_train[X_train.columns[rfe.support_]]

# 添加一个常数项，因为statsmodels的OLS类不会自动添加常数项
X_train_with_constant = sm.add_constant(X_train_selected)

# 创建一个线性回归模型
model = sm.OLS(y_train, X_train_with_constant)

# 拟合模型
results = model.fit()

# 打印显著性检验的结果
print("\n ex2显著性检验结果如下：")
print(results.summary())

# 预测
print("\n ex3:给定条件下预测y值和置信区间")

# 输入x值
x1, x2, x3, x4, x5 = 1.1, 5.1, 5.6, 5.5, 14
X = [[x1, x2, x3, x4, x5]] 

# 将X转换为一个只包含你想要的特征的数组
X_selected = [X[0][i] for i in range(len(X[0])) if rfe.support_[i]]

# 添加一个常数项，因为statsmodels的OLS类不会自动添加常数项
X_with_constant = sm.add_constant([X_selected])
# 添加一个额外的列
X_with_constant = np.hstack([X_with_constant, np.ones((X_with_constant.shape[0], 1))])

# 使用已经拟合的模型进行预测
pred = results.predict(X_with_constant)
print(f"预测值:{pred}")

# 计算预测值的置信区间
ci = results.get_prediction(X_with_constant).conf_int()
print(f"95%置信区间:{ci}")

# 打印显著性检验的结果
print("\n ex3显著性检验结果如下：")
print(results.summary())