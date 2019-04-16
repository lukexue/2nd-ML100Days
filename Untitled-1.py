# %% [markdown]
# ## 練習時間
# #### 請寫一個函式用來計算 Mean Square Error
# $ MSE = \frac{1}{n}\sum_{i=1}^{n}{(Y_i - \hat{Y}_i)^2} $
#
# ### Hint: [如何取平方](https://googoodesign.gitbooks.io/-ezpython/unit-1.html)

"""
作業1：

請上 Kaggle, 在 Competitions 或 Dataset 中找一組競賽或資料並寫下：

https://www.kaggle.com/samratp/bikeshare-analysis

1. 你選的這組資料為何重要
公眾利益,這資料用於分析共享單車的相關資訊,透過資料可以找出很多洞察

2. 資料從何而來 (tips: 譬如提供者是誰、以什麼方式蒐集)
資料由Udacity是由私立教育組織蒐集,以新的技術從系統內底座解鎖返回自行車的相關資訊

3. 蒐集而來的資料型態為何
目前為三個城市的 csv資料集，纽约市，芝加哥和華盛頓特區

4. 這組資料想解決的問題如何評估
從資料集的借還地點與其他相關資訊,可以找出預測租借尖峰時段,改善租借體驗


作業2：

想像你經營一個自由載客車隊，你希望能透過數據分析以提升業績，請你思考並描述你如何規劃整體的分析/解決方案：

1. 核心問題為何 (tips：如何定義 「提升業績 & 你的假設」)
假設: 如果能夠預測使用者可能的上下車地點,就能提升業績

2. 資料從何而來 (tips：哪些資料可能會對你想問的問題產生影響 & 資料如何蒐集)
上下車時間/位置 會影響想問的問題, 資料的蒐集 可以配合 車內計費車錶紀錄((加上使用者代號))上傳到後端

3. 蒐集而來的資料型態為何
結構化數值表單 csv 

4. 你要回答的問題，其如何評估 (tips：你的假設如何驗證)
蒐集使用資料訓練模型,測試模型準確度, 最後從數據中驗證 當我們能準確知道使用者可能的位置時,能不能提高載客率
以及給予適當的在地折價券

作業3：
如下:
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def mean_squared_error(d,d_hat):
    """
    請完成這個 Function 後往下執行
    """
    mse =  sum((d - d_hat)**2) / len(d)
    return mse 

        
# %%
def mean_absolute_error(d,d_hat):
    mae = sum(abs(d - d_hat)) / len(d)
    return mae 
 

# %%
w = 3
b = 0.5

x_lin = np.linspace(0, 100, 101)

y = (x_lin + np.random.randn(101) * 5) * w + b

plt.plot(x_lin, y, 'b.', label='data points')
plt.title("Assume we have data points")
plt.legend(loc=2)
plt.show()


# %%
y_hat = x_lin * w + b
plt.plot(x_lin, y, 'b.', label='data')
plt.plot(x_lin, y_hat, 'r-', label='prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc=2)
plt.show()


# %%
# 執行 Function, 確認有沒有正常執行
MSE = mean_squared_error(y, y_hat)
MAE = mean_absolute_error(y, y_hat)
print("The Mean squared error is %.3f" % (MSE))
print("The Mean absolute error is %.3f" % (MAE))
