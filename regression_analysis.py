import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import set_config
import warnings
warnings.simplefilter('ignore', FutureWarning)

set_config(display='diagram') # 可视化管道

# 1. 数据
table_b11 = pd.DataFrame({
  'Clarity': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.8, 0.7, 1, 0.9, 1, 1, 1, 0.9, 0.9, 1, 0.7, 0.7, 1, 1, 1, 1, 1, 1, 1, 0.8, 1, 1, 0.8, 0.8, 0.8, 0.8],
  'Aroma': [3.3, 4.4, 3.9, 3.9, 5.6, 4.6, 4.8, 5.3, 4.3, 4.3, 5.1, 3.3, 5.9, 7.7, 7.1, 5.5, 6.3, 5, 4.6, 3.4, 6.4, 5.5, 4.7, 4.1, 6, 4.3, 3.9, 5.1, 3.9, 4.5, 5.2, 4.2, 3.3, 6.8, 5, 3.5, 4.3, 5.2],
  'Body': [2.8, 4.9, 5.3, 2.6, 5.1, 4.7, 4.8, 4.5, 4.3, 3.9, 4.3, 5.4, 5.7, 6.6, 4.4, 5.6, 5.4, 5.5, 4.1, 5, 5.4, 5.3, 4.1, 4, 5.4, 4.6, 4, 4.9, 4.4, 3.7, 4.3, 3.8, 3.5, 5, 5.7, 4.7, 5.5, 4.8],
  'Flavor': [3.1, 3.5, 4.8, 3.1, 5.5, 5, 4.8, 4.3, 3.9, 4.7, 4.5, 4.3, 7, 6.7, 5.8, 5.6, 4.8, 5.5, 4.3, 3.4, 6.6, 5.3, 5, 4.1, 5.7, 4.7, 5.1, 5, 5, 2.9, 5, 3, 4.3, 6, 5.5, 4.2, 3.5, 5.7],
  'Oakiness': [4.1, 3.9, 4.7, 3.6, 5.1, 4.1, 3.3, 5.2, 2.9, 3.9, 3.6, 3.6, 4.1, 3.7, 4.1, 4.4, 4.6, 4.1, 3.1, 3.4, 4.8, 3.8, 3.7, 4, 4.7, 4.9, 5.1, 5.1, 4.4, 3.9, 6, 4.7, 4.5, 5.2, 4.8, 3.3, 5.8, 3.5],
  'Quality': [9.8, 12.6, 11.9, 11.1, 13.3, 12.8, 12.8, 12, 13.6, 13.9, 14.4, 12.3, 16.1, 16.1, 15.5, 15.5, 13.8, 13.8, 11.3, 7.9, 15.1, 13.5, 10.8, 9.5, 12.7, 11.6, 11.7, 11.9, 10.8, 8.5, 10.7, 9.1, 12.1, 14.9, 13.5, 12.2, 10.3, 13.2]
})

# 2. 建立完整模型
X = table_b11[['Clarity', 'Aroma', 'Body', 'Flavor', 'Oakiness']]
y = table_b11['Quality']
X_train, X_test, y_train, y_test = train_test_split(X, y)

lr = LinearRegression().fit(X_train, y_train)
print(f'完整模型R^2为:{r2_score(y_test,lr.predict(X_test))}')

# 3. 逐步选择
rfe = RFE(lr,n_features_to_select=3).fit(X_train, y_train)
print(f'选出的变量为:{X.columns[rfe.support_]}\n')

# 4. 预测和区间
x_new = pd.DataFrame({'Clarity': [1.1], 'Aroma': [5.1], 'Body': [5.6], 'Flavor': [5.5], 'Oakiness': [14]})
pred = lr.predict(x_new)
conf_interval = lr.predict(x_new) * 1.96 * np.std(lr.coef_) # 置信区间
print(f'预测值:{pred}')
print(f'置信区间:{conf_interval}')

