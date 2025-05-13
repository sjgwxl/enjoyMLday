import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.models
import logging ,os,matplotlib
matplotlib.use('TkAgg')  # 使用Tkinter后端
from PIL import Image
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


def load_20x20_grayscale_images(directory):
    """
    加载指定目录下所有20x20分辨率的灰度图像

    参数:
        directory (str): 图片所在目录路径

    返回:
        tuple: 包含两个元素的元组：
            - images (list): 图像数据列表，每个元素是20x20的numpy数组
            - filenames (list): 对应的文件名列表
    """
    # 验证目录有效性
    if not os.path.isdir(directory):
        raise ValueError(f"无效目录: {directory}")

    supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = []
    filenames = []

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # 跳过子目录和非图片文件
        if not os.path.isfile(filepath):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_ext:
            print(f"跳过不支持的文件格式: {filename}")
            continue

        try:
            with Image.open(filepath) as img:
                # 验证图像模式和尺寸
                if img.mode != 'L':
                    print(f"跳过非灰度图像: {filename} (模式: {img.mode})")
                    continue

                if img.size != (20, 20):
                    print(f"跳过非20x20图像: {filename} (实际尺寸: {img.size})")
                    continue

                # 转换为numpy数组并归一化
                img_array = np.array(img, dtype=np.float32) / 255.0
                # flattened = img_array.reshape(1, -1)  # 从(20,20)变为(1,400)
                flattened = img_array.flatten()  # 完全转换为标量从(20,20)变为(400,)
                images.append(flattened)
                filenames.append(filename)

        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {str(e)}")
            continue

    if not images:
        return np.empty((0, 400)), []

    return np.array(images), filenames


def display_flattened_images(features, filenames, images_per_row=5):
    """
    将展平的图像数组还原并显示

    参数:
        features (ndarray): (n,400)形状的展平图像数组
        filenames (list): 对应的文件名列表
        images_per_row (int): 每行显示的图像数量，默认为5
    """
    n_samples = features.shape[0]

    # 计算需要的行数
    n_cols = images_per_row
    n_rows = (n_samples + n_cols - 1) // n_cols     #整除向上取行数

    # 创建画布
    plt.figure(figsize=(n_cols * 3, n_rows * 3))

    for idx in range(n_samples):
        # 获取单个样本数据
        sample = features[idx]

        # 数据验证
        if sample.min() < 0 or sample.max() > 1:
            print(f"警告: 样本 {filenames[idx]} 的数值范围异常: [{sample.min():.2f}, {sample.max():.2f}]")
            # 自动归一化处理
            sample = (sample - sample.min()) / (sample.max() - sample.min())

        # 转换为uint8类型并reshape
        img_array = (sample * 255).astype(np.uint8).reshape(20, 20)

        # 创建子图
        plt.subplot(n_rows, n_cols, idx + 1)

        # 显示图像
        plt.imshow(img_array, cmap='gray')
        plt.title(f"{filenames[idx][:15]}...")  # 截断长文件名
        plt.axis('off')

        # 每行最后一个图像后添加分隔线
        if (idx + 1) % n_cols == 0 and idx != n_samples - 1:
            plt.plot([0, 20], [20, 0], color='white', linewidth=2)

    # 调整布局
    plt.tight_layout(pad=1.0)
    plt.show()

def getTestPic(file_path):
    try:
        with Image.open(file_path) as img:
            gray_img = img.convert('L')
            resized_img = gray_img.resize(
                (20, 20),
                resample=Image.LANCZOS
            )
    except Exception as e:
        print(f"处理失败: {file_path} - {str(e)}")

    img_array = np.array(resized_img, dtype=np.float32) / 255.0
    return img_array


if __name__ == "__main__":
    X,name = load_20x20_grayscale_images("D:\\ZK_WORK\\working\\PyCharmProject\\learningPython\\训练集\\out")
    y = np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
                  [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
                  [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                  [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                  [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
                  [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0],
                  [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                  [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
                  ])
    print(f"X.shape:{X.shape},Y.shape{y.shape}")
    print(f"type(X):{type(X)}")
    m,n = X.shape

    # display_flattened_images(X, name, images_per_row=10) #图形显示可以略

    #构建模型
    model = tensorflow.keras.models.Sequential(
        [
            tf.keras.Input(shape=(400,)),
            tensorflow.keras.layers.Dense(25,activation="sigmoid"),
            tensorflow.keras.layers.Dense(15, activation="sigmoid"),
            tensorflow.keras.layers.Dense(1, activation="sigmoid")
        ], name = "my_model"
    )

    #模型参数确认
    model.summary()
    L1_num_params = 400 * 25 + 25  # W1 parameters  + b1 parameters
    L2_num_params = 25 * 15 + 15  # W2 parameters  + b2 parameters
    L3_num_params = 15 * 1 + 1  # W3 parameters  + b3 parameters
    print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params, ",  L3 params = ", L3_num_params)

    #层具体参数确认
    [layer1, layer2, layer3] = model.layers
    W1, b1 = layer1.get_weights()
    W2, b2 = layer2.get_weights()
    W3, b3 = layer3.get_weights()
    print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

    #模型训练
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )
    model.fit(
        X, y,
        epochs=200
    )

    #训练完成的模型进行预测试试
    # prediction = model.predict(X[6].reshape(1, 400))  # a zero
    # print(f" predicting a one: {prediction}")
    # prediction = model.predict(X[31].reshape(1, 400))  # a one
    # print(f" predicting a zero:  {prediction}")
    # yhat = np.zeros(m)
    # for idx in range(m):
    #     pred = model.predict(X[idx].reshape(1, 400))
    #     if pred >= 0.5:
    #         yhat[idx] = 1
    #     else:
    #         yhat[idx] = 0
    # print(f"prediction after threshold: {yhat}")

    #用新图像做预测
    prediction = model.predict(getTestPic("D:\\ZK_WORK\\working\\PyCharmProject\\learningPython\\校验用数据\\0和1识别\\0.png").reshape(1, 400))
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    print(f"prediction after threshold: {yhat}")

