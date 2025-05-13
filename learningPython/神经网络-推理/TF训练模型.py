import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 仅显示Error级别日志

if __name__ == "__main__":
    input_data = np.array([[185.32,12.69],[259.92,11.87], [231.01,14.41],[175.37,11.72],[187.12,14.13],[225.91,12.1],[208.41,14.18],[207.08,14.03],
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
    output_data = np.array([[1.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                  [0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                  [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                  [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                  [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])
    print(input_data.shape)
    print(output_data.shape)
    norm_l = tf.keras.layers.Normalization(axis=-1)
    listX = norm_l(input_data[:]).numpy()
    listY = output_data[:]
    listX1 = np.array(listX)[:, 0]
    listX2 = np.array(listX)[:, 1]
    #绘制所有位置点
    plt.plot(listX1, listX2, 'o')
    #将真实结果中大于0.5的位置画出来
    for idx in range(len(listY)):
        if listY[idx] >= 0.5:
            plt.plot(listX1[idx], listX2[idx], 'ro')
    plt.show()

    # 数据标准化
    print(f"Temperature Max, Min pre normalization: {np.max(input_data[:, 0]):0.2f}, {np.min(input_data[:, 0]):0.2f}")
    print(f"Duration    Max, Min pre normalization: {np.max(input_data[:, 1]):0.2f}, {np.min(input_data[:, 1]):0.2f}")
    norm_l.adapt(input_data)  # learns mean, variance
    Xn = norm_l(input_data)
    print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
    print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

    #建立训练集
    Xt = np.tile(Xn, (1000, 1))
    Yt = np.tile(output_data, (1000, 1))
    print(Xt.shape, Yt.shape)

    #建立训练模型
    tf.random.set_seed(1234)  # applied to achieve consistent results
    model = Sequential(
        [
            tf.keras.Input(shape=(2,)),
            Dense(3, activation='sigmoid', name='layer1'),
            Dense(1, activation='sigmoid', name='layer2')
        ]
    )
    model.summary()
    #模型参数数量显示
    L1_num_params = 2 * 3 + 3  # W1 parameters  + b1 parameters
    L2_num_params = 3 * 1 + 1  # W2 parameters  + b2 parameters
    print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params)
    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
    print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

    #模型训练
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    )
    model.fit(
        Xt, Yt,
        epochs=10,
    )

    #获取训练后的拟合参数
    W1, b1 = model.get_layer("layer1").get_weights()
    W2, b2 = model.get_layer("layer2").get_weights()
    print("W1:\n", W1, "\nb1:", b1)
    print("W2:\n", W2, "\nb2:", b2)

    #模型更新特定参数(如果不需要调整参数 则该步骤可以不进行)
    model.get_layer("layer1").set_weights([W1, b1])
    model.get_layer("layer2").set_weights([W2, b2])

    #拿模型进行预测
    X_test = np.array([
        [200, 13.9],  # positive example
        [200, 17]])  # negative example
    X_testn = norm_l(X_test)
    predictions = model.predict(X_testn)
    print("predictions = \n", predictions)