import numpy as np
import time

def my_dot(a,b,num):
    result = 0
    for i in range(0,num):
        result += a[i] * b[i]
    return result


allNum = 1000000
x = np.random.rand(allNum)
y = np.random.rand(allNum)
a = np.array(x)
b = np.array(y)

tic = time.time()
c = np.dot(a,b)
toc = time.time()
print(f"np.dot(a,b)={c:.4f}")
print(f"use time:={1000*(toc-tic):.4f}ms")

tic = time.time()
d = my_dot(a,b,allNum)
toc = time.time()
print(f"my_dot(a,b)={d:.4f}")
print(f"use time:={1000*(toc-tic):.4f}ms")