import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 仅显示Error级别日志

def sigmoid(z):
    result = 1/(1+np.exp(-z))
    return result

def my_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w = W[:,j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)

    # #可以直接使用矩阵乘法-矢量化
    # Z = np.matmul(a_in,W)+b  #这里numpy会自动拓展b(1*j)为(m*j)向np.matmul(a_in,W)结果匹配
    # a_out = sigmoid(Z)
    return(a_out)

def my_sequential(x, W1, b1, W2, b2):
    #前向传播
    a1 = my_dense(x,  W1, b1)
    a2 = my_dense(a1, W2, b2)
    return(a2)

def my_predict(X, W1, b1, W2, b2):
    #预测函数
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)

if __name__ == "__main__":
    X = np.array([[185.32,12.69],[259.92,11.87], [231.01,14.41],[175.37,11.72],[187.12,14.13],[225.91,12.1],[208.41,14.18],[207.08,14.03],
        [280.6,14.23],[202.87,12.25],[196.7,13.54],[270.31,14.6],[192.95,15.2],[213.57,14.28],[164.47,11.92],[177.26,15.04],
        [241.77,14.9],[237.,13.13],[219.74,13.87],[266.39,13.25],[270.45,13.95],[261.96,13.49],[243.49,12.86],[220.58,12.36],
        [163.59,11.65],[244.76,13.33],[271.19,14.84],[201.99,15.39],[229.93,14.56],[204.97,12.28],[173.19,12.22],[231.51,11.95],
        [152.69,14.83],[163.42,13.3],[215.95,13.98],[218.04,15.25],[251.3,13.8],[233.33,13.53],[280.24,12.41],[243.02,13.72],
        [155.67,12.68],[275.17,14.64],[151.73,12.69],[151.32,14.81],[164.9,11.73],[282.55,13.28],[192.98,11.7],[202.6,12.96],
        [220.67,11.53],[169.97,12.34],[209.47,12.71],[232.8,12.64],[272.8,15.35],[158.02,12.34],[226.01,14.58],[158.64,12.24],
        [211.66,14.17],[271.95,14.97],[257.16,11.71],[281.85,13.96],[161.63,12.52],[233.8,13.04],[210.29,14.72],[261.24,13.69],
        [256.98,13.12],[281.56,13.92],[280.64,11.68],[269.16,13.74],[246.34,12.27],[224.07,12.66],[164.24,11.51],[272.42,14.18],
        [177.68,12.53],[212.86,14.77],[165.88,15.37],[277.43,12.48],[236.51,12.94],[244.14,11.85],[213.45,13.85],[234.57,14.27],
        [270.34,12.47],[170.68,13.06],[226.79,15.34],[245.92,14.45],[281.32,12.57],[185.03,13.19],[189.88,14.1],[278.48,12.11],
        [219.92,14.21],[216.58,15.15],[249.48,15.03],[165.09,12.28],[158.87,14.82],[279.98,11.56],[256.55,14.41],[272.61,12.58],
        [246.49,12.45],[160.26,14.48],[155.7,14.3],[188.27,13.45]])
    Y = np.array([[1.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                  [0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                  [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                  [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                  [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])
    print(X.shape)
    print(Y.shape)
    print(f"Temperature Max, Min pre normalization: {np.max(X[:, 0]):0.2f}, {np.min(X[:, 0]):0.2f}")
    print(f"Duration    Max, Min pre normalization: {np.max(X[:, 1]):0.2f}, {np.min(X[:, 1]):0.2f}")
    norm_l = tf.keras.layers.Normalization(axis=-1)
    norm_l.adapt(X)
    Xn = norm_l(X) #数据标准归一化，优化计算数据
    print(f"Temperature Max, Min post normalization: {np.max(Xn[:, 0]):0.2f}, {np.min(Xn[:, 0]):0.2f}")
    print(f"Duration    Max, Min post normalization: {np.max(Xn[:, 1]):0.2f}, {np.min(Xn[:, 1]):0.2f}")

    #假设已拿到训练过的w和b参数
    W1_tmp = np.array([[-8.93, 0.29, 12.9], [-0.1, -7.32, 10.81]]) #2*3
    b1_tmp = np.array([-9.82, -9.28, 0.96])
    W2_tmp = np.array([[-31.18], [-27.59], [-32.56]]) #3*1
    b2_tmp = np.array([15.41])

    #训练后的参数写入，并输入数据用于预测
    X_tst = np.array([
        [200, 13.9],  # postive example
        [200, 17]])  # negative example
    X_tstn = norm_l(X_tst)  # remember to normalize
    predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

    #显示预测结果
    yhat = np.zeros_like(predictions)
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            yhat[i] = 1
        else:
            yhat[i] = 0
    print(f"decisions = \n{yhat}")

    netf = lambda x: my_predict(norm_l(x), W1_tmp, b1_tmp, W2_tmp, b2_tmp)
    listX = norm_l(X[:]).numpy()
    listYhat = netf(X)
    listX1 = np.array(listX)[:, 0]
    listX2 = np.array(listX)[:, 1]

    #绘制所有位置点
    plt.plot(listX1, listX2, 'o')

    #将预测结果中大于0.5的位置画出来
    for idx in range(len(listYhat)):
        if listYhat[idx] >= 0.5:
            plt.plot(listX1[idx], listX2[idx], 'ro')

    plt.show()