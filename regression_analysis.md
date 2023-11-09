# 2024第一次美赛培训作业

> 计算机217 李奕哲 2021212205098

01：安装必要的库


```python
%pip install numpy
%pip install pandas
%pip install scikit-learn
%pip install sklearn
%pip install matplotlib
%pip install statsmodels
```

02：导入所需的库文件


```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import warnings
```

03：去除报错提示


```python
warnings.filterwarnings('ignore')
```

04：导入数据

> 加载和准备葡萄酒质量评估数据


```python
table_b11 = pd.DataFrame({
  'Clarity': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.8, 0.7, 1, 0.9, 1, 1, 1, 0.9, 0.9, 1, 0.7, 0.7, 1, 1, 1, 1, 1, 1, 1, 0.8, 1, 1, 0.8, 0.8, 0.8, 0.8],
  'Aroma': [3.3, 4.4, 3.9, 3.9, 5.6, 4.6, 4.8, 5.3, 4.3, 4.3, 5.1, 3.3, 5.9, 7.7, 7.1, 5.5, 6.3, 5, 4.6, 3.4, 6.4, 5.5, 4.7, 4.1, 6, 4.3, 3.9, 5.1, 3.9, 4.5, 5.2, 4.2, 3.3, 6.8, 5, 3.5, 4.3, 5.2],
  'Body': [2.8, 4.9, 5.3, 2.6, 5.1, 4.7, 4.8, 4.5, 4.3, 3.9, 4.3, 5.4, 5.7, 6.6, 4.4, 5.6, 5.4, 5.5, 4.1, 5, 5.4, 5.3, 4.1, 4, 5.4, 4.6, 4, 4.9, 4.4, 3.7, 4.3, 3.8, 3.5, 5, 5.7, 4.7, 5.5, 4.8],
  'Flavor': [3.1, 3.5, 4.8, 3.1, 5.5, 5, 4.8, 4.3, 3.9, 4.7, 4.5, 4.3, 7, 6.7, 5.8, 5.6, 4.8, 5.5, 4.3, 3.4, 6.6, 5.3, 5, 4.1, 5.7, 4.7, 5.1, 5, 5, 2.9, 5, 3, 4.3, 6, 5.5, 4.2, 3.5, 5.7],
  'Oakiness': [4.1, 3.9, 4.7, 3.6, 5.1, 4.1, 3.3, 5.2, 2.9, 3.9, 3.6, 3.6, 4.1, 3.7, 4.1, 4.4, 4.6, 4.1, 3.1, 3.4, 4.8, 3.8, 3.7, 4, 4.7, 4.9, 5.1, 5.1, 4.4, 3.9, 6, 4.7, 4.5, 5.2, 4.8, 3.3, 5.8, 3.5],
  'Quality': [9.8, 12.6, 11.9, 11.1, 13.3, 12.8, 12.8, 12, 13.6, 13.9, 14.4, 12.3, 16.1, 16.1, 15.5, 15.5, 13.8, 13.8, 11.3, 7.9, 15.1, 13.5, 10.8, 9.5, 12.7, 11.6, 11.7, 11.9, 10.8, 8.5, 10.7, 9.1, 12.1, 14.9, 13.5, 12.2, 10.3, 13.2]
})
data = table_b11 
```


```python
# 分割数据为训练集和测试集
X = data[['Clarity', 'Aroma', 'Body', 'Flavor', 'Oakiness']]
y = data['Quality']

X_train, X_test, y_train, y_test = train_test_split(X, y)
```

将数据集分割为训练集和测试集。

1. `X = data[['Clarity', 'Aroma', 'Body', 'Flavor', 'Oakiness']]`：这行代码从`data`数据框中选择了'Clarity', 'Aroma', 'Body', 'Flavor', 'Oakiness'这五个特征，并将它们赋值给`X`。`X`现在是一个包含这五个特征的数据框。

2. `y = data['Quality']`：这行代码从`data`数据框中选择了'Quality'这个特征，并将它赋值给`y`。`y`现在是一个包含'Quality'特征的Series。

3. `X_train, X_test, y_train, y_test = train_test_split(X, y)`：这行代码使用`train_test_split`函数将`X`和`y`分割为训练集和测试集。默认情况下，`train_test_split`函数将数据分割为75%的训练集和25%的测试集。`X_train`和`y_train`是训练集的特征和标签，`X_test`和`y_test`是测试集的特征和标签。


```python
print("ex1:建立 $y$ 关于 $x_1, x_2, x_3. x_4, x_5$ 的回归模型,并对回归方程和回归系数进行显著性检验。")

# 建立回归模型
lr = LinearRegression().fit(X_train, y_train)

# 评估模型性能
print(f'回归方程的R^2值为:{lr.score(X_test, y_test)}')

print(f'Coeff: \n{lr.coef_}')
model = sm.OLS(y, X)
results = model.fit()
print("回归系数的显著性检验结果如下:\n", results.summary())
```

    ex1:建立 $y$ 关于 $x_1, x_2, x_3. x_4, x_5$ 的回归模型,并对回归方程和回归系数进行显著性检验。
    回归方程的R^2值为:0.6785810707723199
    Coeff: 
    [ 2.57668111  0.67928144  0.2739577   0.95624146 -0.64154221]
    回归系数的显著性检验结果如下:
                                      OLS Regression Results                                
    =======================================================================================
    Dep. Variable:                Quality   R-squared (uncentered):                   0.992
    Model:                            OLS   Adj. R-squared (uncentered):              0.991
    Method:                 Least Squares   F-statistic:                              830.1
    Date:                Thu, 09 Nov 2023   Prob (F-statistic):                    1.11e-33
    Time:                        20:39:58   Log-Likelihood:                         -58.192
    No. Observations:                  38   AIC:                                      126.4
    Df Residuals:                      33   BIC:                                      134.6
    Df Model:                           5                                                  
    Covariance Type:            nonrobust                                                  
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Clarity        4.6830      1.176      3.981      0.000       2.289       7.076
    Aroma          0.4216      0.279      1.510      0.141      -0.146       0.990
    Body           0.5944      0.289      2.055      0.048       0.006       1.183
    Flavor         1.1906      0.314      3.789      0.001       0.551       1.830
    Oakiness      -0.5693      0.272     -2.091      0.044      -1.123      -0.015
    ==============================================================================
    Omnibus:                        0.136   Durbin-Watson:                   0.957
    Prob(Omnibus):                  0.934   Jarque-Bera (JB):                0.214
    Skew:                          -0.128   Prob(JB):                        0.899
    Kurtosis:                       2.736   Cond. No.                         57.7
    ==============================================================================
    
    Notes:
    [1] R² is computed without centering (uncentered) since the model does not contain a constant.
    [2] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

建立一个回归模型，然后对模型的回归方程和回归系数进行显著性检验。

1. `lr = LinearRegression().fit(X_train, y_train)`：这行代码使用`LinearRegression`类创建了一个线性回归模型，并使用`fit`方法将模型拟合到训练数据`X_train`和`y_train`。

2. `print(f'回归方程的R^2值为:{lr.score(X_test, y_test)}')`：这行代码使用`score`方法计算了模型在测试数据`X_test`和`y_test`上的R^2值，并打印出来。R^2值是一个衡量模型拟合优度的统计量，值越接近1，表示模型的拟合效果越好。

3. `print(f'Coeff: \n{lr.coef_}')`：这行代码打印了模型的回归系数。`lr.coef_`是一个数组，包含了模型的每个特征的回归系数。

4. `model = sm.OLS(y, X)`：这行代码使用`statsmodels`库的`OLS`类创建了一个普通最小二乘回归模型。这个模型用于进行更详细的统计分析。

5. `results = model.fit()`：这行代码使用`fit`方法将模型拟合到数据`X`和`y`。

6. `print("回归系数的显著性检验结果如下:\n", results.summary())`：这行代码使用`summary`方法打印了模型的详细统计结果，包括每个特征的回归系数、标准误差、t统计量、p值等，用于进行回归系数的显著性检验。


```python
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
```

    
    ex2:采用逐步回归法建立 $y$ 关于 $x_1, x_2, x_3. x_4, x_5$ 的线性回归模型,并对回归方程和回归系数进行显著性检验。
    选出的特征为:[ True False False  True False]
    Coeff: 
    [2.65914023 1.5170914 ]
    
     ex2显著性检验结果如下：
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Quality   R-squared:                       0.613
    Model:                            OLS   Adj. R-squared:                  0.582
    Method:                 Least Squares   F-statistic:                     19.77
    Date:                Thu, 09 Nov 2023   Prob (F-statistic):           7.10e-06
    Time:                        20:39:58   Log-Likelihood:                -44.941
    No. Observations:                  28   AIC:                             95.88
    Df Residuals:                      25   BIC:                             99.88
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.8727      2.142      1.341      0.192      -1.539       7.285
    Clarity        2.6591      1.893      1.405      0.172      -1.239       6.557
    Flavor         1.5171      0.246      6.179      0.000       1.011       2.023
    ==============================================================================
    Omnibus:                        0.595   Durbin-Watson:                   1.649
    Prob(Omnibus):                  0.743   Jarque-Bera (JB):                0.696
    Skew:                          -0.242   Prob(JB):                        0.706
    Kurtosis:                       2.398   Cond. No.                         57.5
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

使用逐步回归法建立线性回归模型，并对模型的回归方程和回归系数进行显著性检验。

1. `rfe = RFE(lr)`：这行代码使用`RFE`（递归特征消除）类创建了一个递归特征消除模型。这个模型会逐步消除不重要的特征，以找出最重要的特征。

2. `rfe = rfe.fit(X_train, y_train)`：这行代码使用`fit`方法将模型拟合到训练数据`X_train`和`y_train`。

3. `print(f'选出的特征为:{rfe.support_}')`：这行代码打印了模型选出的特征。`rfe.support_`是一个布尔数组，表示每个特征是否被选出。

4. `print(f'Coeff: \n{rfe.estimator_.coef_}')`：这行代码打印了模型的回归系数。`rfe.estimator_.coef_`是一个数组，包含了模型的每个特征的回归系数。

5. `X_train_selected = X_train[X_train.columns[rfe.support_]]`：这行代码创建了一个新的数据框`X_train_selected`，它只包含`X_train`中被选出的特征。

6. `X_train_with_constant = sm.add_constant(X_train_selected)`：这行代码使用`sm.add_constant()`函数添加了一个常数项。

7. `model = sm.OLS(y_train, X_train_with_constant)`：这行代码使用`statsmodels`库的`OLS`类创建了一个普通最小二乘回归模型。

8. `results = model.fit()`：这行代码使用`fit`方法将模型拟合到数据。

9. `print("\n ex2显著性检验结果如下：")`和`print(results.summary())`：这两行代码打印了模型的详细统计结果，包括每个特征的回归系数、标准误差、t统计量、p值等，用于进行回归系数的显著性检验。


```python
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
```

    
     ex3:给定条件下预测y值和置信区间
    预测值:[19.30233901]
    95%置信区间:[[ 1.72450357 36.88017446]]
    
     ex3显著性检验结果如下：
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                Quality   R-squared:                       0.613
    Model:                            OLS   Adj. R-squared:                  0.582
    Method:                 Least Squares   F-statistic:                     19.77
    Date:                Thu, 09 Nov 2023   Prob (F-statistic):           7.10e-06
    Time:                        20:39:58   Log-Likelihood:                -44.941
    No. Observations:                  28   AIC:                             95.88
    Df Residuals:                      25   BIC:                             99.88
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.8727      2.142      1.341      0.192      -1.539       7.285
    Clarity        2.6591      1.893      1.405      0.172      -1.239       6.557
    Flavor         1.5171      0.246      6.179      0.000       1.011       2.023
    ==============================================================================
    Omnibus:                        0.595   Durbin-Watson:                   1.649
    Prob(Omnibus):                  0.743   Jarque-Bera (JB):                0.696
    Skew:                          -0.242   Prob(JB):                        0.706
    Kurtosis:                       2.398   Cond. No.                         57.5
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

使用已经拟合的模型对新的观测值进行预测，并计算预测值的置信区间。

1. `x1, x2, x3, x4, x5 = 1.1, 5.1, 5.6, 5.5, 14`和`X = [[x1, x2, x3, x4, x5]]`：这两行代码创建了一个新的观测值`X`，它是一个包含5个特征的二维数组。

2. `X_selected = [X[0][i] for i in range(len(X[0])) if rfe.support_[i]]`：这行代码创建了一个新的数组`X_selected`，它只包含`X`中被选出的特征。

3. `X_with_constant = sm.add_constant([X_selected])`：这行代码使用`sm.add_constant()`函数添加了一个常数项。

4. `X_with_constant = np.hstack([X_with_constant, np.ones((X_with_constant.shape[0], 1))])`：这行代码添加了一个额外的列，以使`X_with_constant`的形状与`results.params`的形状兼容。

5. `pred = results.predict(X_with_constant)`：这行代码使用`results.predict()`方法预测新的观测值。

6. `ci = results.get_prediction(X_with_constant).conf_int()`：这行代码使用`results.get_prediction().conf_int()`方法计算预测值的95%置信区间。

7. `print(results.summary())`：这行代码使用`results.summary()`方法打印显著性检验的结果。
