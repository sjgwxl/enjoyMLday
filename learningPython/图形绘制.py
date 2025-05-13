import math
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    x = np.arange(0,50,1)
    a = -1
    b = 50
    c = -1
    y = a*x + b*x**2 + c*x**3
    plt.plot(x,y,'bo-',linewidth=1,label='f(x)')
    plt.show()