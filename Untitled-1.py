# %% [markdown]
# ## 練習時間
# #### 請寫一個函式用來計算 Mean Square Error
# $ MSE = \frac{1}{n}\sum_{i=1}^{n}{(Y_i - \hat{Y}_i)^2} $
#
# ### Hint: [如何取平方](https://googoodesign.gitbooks.io/-ezpython/unit-1.html)

# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def mean_squared_error():
    """
    請完成這個 Function 後往下執行
    """


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
