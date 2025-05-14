import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = [float(i)/10.0 for i in range(1,100)]
    y = [math.log(i) for i in x]
    plt.plot([1,1],[-3,5],"r--") #x:1->1 y:-3->5
    plt.plot([0,10],[1,1],"y--") #x:0->10 y:1->1
    plt.plot(x,y,'bo-',linewidth=1,label='f(x)')
    plt.show()