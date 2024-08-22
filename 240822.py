import numpy as np
import matplotlib.pyplot as plt

# y=a*x^2 + b*x+ c
a=2
b=3
c=5

x = np.linspace(-8, 8, 100)
y = a*x**2 + b*x + c
plt.plot(x, y, color="black")

# y=a*x**3 + b*x**2 + c*x + d
# y=a*x**4 + b*x**3 + c*x**2 + d*x + e
a=2
b=3
c=1
d=-1
e=5

x = np.linspace(-8, 8, 100)
# y = a*x**3 + b*x**2 + c*x + d
y = a*x**4 + b*x**3 + c*x**2 + d*x + e
plt.plot(x, y, color="black")