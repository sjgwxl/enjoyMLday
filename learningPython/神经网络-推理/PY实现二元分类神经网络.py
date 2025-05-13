import numpy as np
import matplotlib.pyplot as plt
import logging ,os,matplotlib
from PIL import Image

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

def sigmoid(z):
    result = 1/(1+np.exp(-z))
    return result

def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)

    for j in range(units):
        #取出第j列
        w = W[:,j]
        z = np.dot(a_in,w) + b[j]
        a_out[j] = g(z)

    # 矢量化写法更加便捷
    # x = X[0].reshape(-1, 1)  # column vector (400,1)
    # z1 = np.matmul(x.T, W1) + b1  # (1,400)(400,25) = (1,25)
    # a1 = sigmoid(z1)

    return(a_out)

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    a1 = my_dense(x,  W1, b1, sigmoid)
    a2 = my_dense(a1, W2, b2, sigmoid)
    a3 = my_dense(a2, W3, b3, sigmoid)
    return(a3)

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

    # Quick Check
    x_tst = 0.1 * np.arange(1, 3, 1).reshape(2, )  # (1 examples, 2 features)
    W_tst = 0.1 * np.arange(1, 7, 1).reshape(2, 3)  # (2 input features, 3 output features)
    b_tst = 0.1 * np.arange(1, 4, 1).reshape(3, )  # (3 features)
    A_tst = my_dense(x_tst, W_tst, b_tst, sigmoid)
    print(A_tst)

    #假设已经完成训练，拿到推理所需的参数，由于真实参数太多，此处用0初始化验证流程正确
    W1 = np.zeros(shape=(400,25))
    W2 = np.zeros(shape=(25,15))
    W3 = np.zeros(shape=(15,1))
    b1 = np.zeros(shape=(25,))
    b2 = np.zeros(shape=(15,))
    b3 = np.zeros(shape=(1,))
    print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
    print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
    print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

    #以特定参数带入函数，用第一二张图像做预测
    prediction = my_sequential(X[0], W1, b1, W2, b2, W3, b3)
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    print("yhat = ", yhat, " label= ", y[0, 0])
    prediction = my_sequential(X[1], W1, b1, W2, b2, W3, b3)
    if prediction >= 0.5:
        yhat = 1
    else:
        yhat = 0
    print("yhat = ", yhat, " label= ", y[1, 0])

