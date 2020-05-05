import numpy as np
import matplotlib.pyplot as plt

train = [0.0168, 0.0178, 0.0214, 0.0239, 0.0280]
train = np.log(train)
test = [0.0271, 0.0348, 0.0495, 0.0696, 0.1015]
test = np.log(test)
num = [60000, 30000, 15000, 7500, 3750]
num = np.log(num)
plt.plot(num, train)
plt.plot(num, test)
plt.legend(["train error", "test error"])
plt.xlabel("log scale training examples")
plt.ylabel("log scale error")
plt.title("log-log scale plot for train and test error versus training examples")
plt.show()
