# Importing Libraries
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Downloading and Preparing Data
df = yf.download("BTC-USD")
df = df[:900]
y = df["Close"].values.reshape(-1,1)
X = np.arange(len(y)).reshape(-1, 1)

# Algorithm Settings
learning_rate = 0.000001
epochs = 1500 #( تعداد تکرارها برای آموزش مدل)
X_bias = np.c_[np.ones_like(X), X] #(اضافه کردن یک ستون از یک‌ها به ماتریس X برای محاسبه بایاس (bias).)
sample , features = X_bias.shape#(تعیین تعداد نمونه‌ها و ویژگی‌ها)
theta = np.zeros((features,1))#(پارامترهای مدل که با مقدار اولیه صفر مقداردهی می‌شوند.)
cost_function = []#(لیستی برای ذخیره مقادیر تابع هزینه در هر تکرار.)

# animate Function
def animate(frame):
    global X_bias, theta, learning_rate, sample,cost_function
    indices = np.random.permutation(sample)#(ایجاد یک ترتیب تصادفی از اندیس‌ها برای پیاده‌سازی SGD.)
    for i in range(sample):
        index = indices[i]
        X_i = X_bias[index:index+1]
        y_i = y[index:index+1]
        h = np.dot(X_i, theta)#( محاسبه فرضیه (hypothesis))
        error = h - y_i
        grdient = np.dot(X_i.T,error)
        print(theta)
        theta -= learning_rate * grdient

    h = np.dot(X_bias, theta)
    error = h - y
    mean_squared_error = np.mean(np.square(error))
    cost_function.append(mean_squared_error)
    line1.set_ydata(h)
    line2.set_xdata(np.arange(len(cost_function)))
    line2.set_ydata(cost_function)
    ax2.set_ylim(0, max(cost_function) +max(cost_function) * 0.1)
    return line1,line2

# Creating Plot and Animation

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(X, y)

line1, = ax1.plot(X, y)
line2, = ax2.plot([], [], lw=2)

ax1.set_xlim(0, len(X))
ax1.set_ylim(0, y.max() + y.max() * 0.1)

ax2.set_xlim(-10, epochs)

ani = FuncAnimation(fig, animate, frames=epochs, interval=1)
plt.show()
