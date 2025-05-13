import copy,math
import numpy as np
import matplotlib.pyplot as plt


def predict_single_loop(x, w, b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):  model parameter
    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p += x[i] * w[i]
    p += b
    return p


def predict(x, w, b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):             model parameter
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b
    return p


def compute_cost(X, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i]) ** 2  # scalar
    cost = cost / (2 * m)  # scalar
    return cost


def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape  # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    print(f'X.shape:{X.shape}')
    print(f'X[0].shape:{X[0].shape}')
    print(f'y.shape:{y.shape}')
    print(f'w.shape:{w.shape}')

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i] #预测值f(x[i])-y[i] 得到差值
        for j in range(n):
            dj_dw[j] += err * X[i, j] #计算得到矢量w中每个行的偏导标量值
        dj_db += err #计算b的偏导
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    多元变量的方向导数
    Performs batch gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)  ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw  ##None
        b = b - alpha * dj_db  ##None

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history  # return final w,b and J history for graphing

if __name__ == "__main__":
    np.set_printoptions(precision=2)

    #初始化参数_训练数组
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    #初始化参数_预测函数
    b_init = 785.1811367994083
    w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

    # #从测试数据中取出一行
    # x_vec = X_train[0,:]
    #
    # #输入函数预测
    # f_wb = predict_single_loop(x_vec, w_init, b_init)
    #
    # #计算和显示损失函数,输入m个包含n个特征的输入数据X_train(m,n),输入目标参数y_train(n,),使用选用的w(4,)和b()参数
    # cost = compute_cost(X_train, y_train, w_init, b_init)
    # print(f'Cost at optimal w : {cost}')
    #
    # #计算和显示gradient
    # tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
    # print(f'dj_db at initial w,b: {tmp_dj_db}')
    # print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

    #参数初始化
    initial_w = np.zeros_like(w_init)
    initial_b = 0.
    #梯度参数设置 迭代次数和学习率抉择
    # iterations = 1000
    iterations = 3000
    # alpha = 5.0e-7
    alpha = 1.0e-9
    #执行梯度下降，获取最终训练结果5
    w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                compute_cost, compute_gradient,
                                                alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    #特定数据预测
    m, n = X_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
    #绘制预测曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration")
    ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')
    ax2.set_ylabel('Cost')
    ax1.set_xlabel('iteration step')
    ax2.set_xlabel('iteration step')
    plt.show()