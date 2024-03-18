import numpy as np
import matplotlib.pyplot as plt
import polars as pl

data_0 = pl.read_csv("Features_by_window_size/sero_features_4.csv")
print(data_0)

target = data_0[:,"Target"]
x = np.linspace(0,target.len(),target.len())
y = data_0[:,"AccMeanX"]
y2 = data_0[:,"AccMeanY"]
y3 = data_0[:,"AccMeanZ"]

y4 = data_0[:,"GyroMeanX"]
y5 = data_0[:,"GyroMeanY"]
y6 = data_0[:,"GyroMeanZ"]


plt.figure()
plt.plot(x,y,color="red")
plt.plot(x,y2,color="blue")
plt.plot(x,y3,color="green")

plt.figure()
plt.plot(x,y4,color="red")
plt.plot(x,y5,color="blue")
plt.plot(x,y6,color="green")

plt.show()

