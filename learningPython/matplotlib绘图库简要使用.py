import math
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    x = np.arange(0.05,3,0.1)
    y1 = [math.log(a,1.5) for a in x]
    plt.plot(x,y1,'bo-',linewidth=1,label='log1.5(x)')
    plt.plot([1,1],[y1[0],y1[-1]],'r--',linewidth=1)
    plt.show()


if __name__ == "__main__":
    x = [float(i)/100.0 for i in range(1,1000)]
    y = [math.log(i) for i in x]
    plt.plot(x,y,'r-',linewidth=1,label='log Curve')
    a = [x[20],x[175]]
    b = [y[20],y[175]]
    plt.plot(a,b,'g-',linewidth=1)
    plt.plot(a,b,'b*',markersize=5,alpha=0.75)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('log(x)')
    plt.show()


if __name__ == "__main__":
    times = 10000
    u = np.random.uniform(0.0,1.0,times)
    print(u)
    print(u.shape)
    print(type(u))
    plt.hist(u,80,facecolor='g',alpha=0.75)
    plt.grid(True)
    plt.show()

    for time in range(times):
        u += np.random.uniform(0.0,1.0,times)
    u /= times
    plt.hist(u,80,facecolor='g',alpha=0.75)
    plt.grid(True)
    plt.show()
