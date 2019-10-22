import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0.0, 100.0, 11)
y1 = x
y2 = x*x
y3 = np.exp(x)
plt.plot(x,y1,'r')
plt.plot(x,y2, 'g')
plt.plot(x,y3, 'b')
plt.show()