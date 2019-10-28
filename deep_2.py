import numpy as np
import matplotlib.pyplot as plt
import h5py
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 保存的是训练集里面的图像数据（本训练集有209张64x64的图像）。
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 保存的是训练集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # 保存的是测试集里面的图像数据（本训练集有50张64x64的图像）。
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # 保存的是测试集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
    classes = np.array(test_dataset["list_classes"][:])  # 保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]。
    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    print("训练集_图片的维数 : " + str(train_set_x_orig.shape))
    print("训练集_标签的维数 : " + str(train_set_y.shape))
    print("测试集_图片的维数: " + str(test_set_x_orig.shape))
    print("测试集_标签的维数: " + str(test_set_y.shape))
    print()
    train_set_x = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T / 255  # 降维,化为区间（0，1）内的数
    test_set_x = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T / 255  # 降维,化为区间（0，1）内的数
    return train_set_x, train_set_y, test_set_x, test_set_y, classes
def relu(z):#relu函数
    return np.maximum(0,z)
def relu_1(z):#relu函数的导数
    return np.maximum(0, z/np.abs(z))
def tanh(z):  # 用tanh函数作为激活函数
    e1 = np.exp(z)
    e2 = np.exp(-z)
    return (e1 - e2) / (e1 + e2)
def tanh_1(a):  # tanh函数的导数
    return 1 - a ** 2

def sigmoid(z):  # sigmoid函数作为激活函数
    return 1 / (1 + np.exp(-z))

def sigmoid_1(a):
    return a*(1-a)
def rand(n, m, dim):
    w1 = np.random.randn(dim, m)*0.001
    w2 = np.random.randn(1, dim)*0.001  # 随机从产生符合正态分布的数
    b1 = np.zeros((dim, 1), dtype='float')  # 初始化为0
    b2 = np.zeros((1, 1), dtype='float')  # python中的广播
    w = {
        "w1": w1,
        "w2": w2
    }
    b = {
        "b1": b1,
        "b2": b2
    }
    return w, b

def backward(X, Y, w1, w2, b1, b2, learn):
    m = X.shape[1]  # 样本个数
    z1 = np.dot(w1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    L = -1 / m * np.sum(Y * np.log(a2) + (1 - Y) * np.log(1 - a2))
    L = np.squeeze(L)  # 去除多余的维度

    da2 = - (np.divide(Y, a2) - np.divide(1 - Y, 1 - a2))
    dz2 = da2 * sigmoid_1(a2)
    dw2 = 1 / m * np.dot(dz2, a1.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * relu_1(z1)

    dw1 = 1 / m * np.dot(dz1, X.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

    w1 = w1 - learn * dw1
    w2 = w2 - learn * dw2
    b1 = b1 - learn * db1
    b2 = b2 - learn * db2

    w = {
        "w1": w1,
        "w2": w2
    }
    b = {
        "b1": b1,
        "b2": b2
    }
    return w, b, L


def test(w1,w2,b1 ,b2, X):  # 通过优化后的参数w,b预测出 y的值
    m = X.shape[1]  # 样本个数
    z1 = np.dot(w1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)
    y = np.zeros(shape= (1,m),dtype = float)
    for i in range(a2.shape[1]):
        y[0, i] = 1 if a2[0, i] > 0.5 else 0
    return y

def trainback(w1,w2, b1,b2, X, Y):  # 测试
    y = test(w1,w2,b1 ,b2, X)
    lop = 100 * (1 - np.mean(np.abs(y - Y)))
    print("训练集准确性：{0}%".format(lop))
    return 0
def testback(w1,w2, b1,b2, X, Y):  # 测试
    y = test(w1,w2,b1 ,b2, X)
    lop = 100 * (1 - np.mean(np.abs(y - Y)))
    print("测试集准确性：{0}%".format(lop))
    return 0
if __name__ == "__main__":
    np.random.seed(1)
    learning_rate = 0.0075
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
    n, m, dim, i = train_set_x.shape[1], train_set_x.shape[0], 4, 0#n表示样本个数，m表示特征个数，dim表示节点个数

    w, b = rand(n, m, dim)
    w1, w2 = w["w1"], w["w2"]
    b1, b2 = b["b1"], b["b2"]
    L = []
    for i in range(3000):
        w, b, l = backward(train_set_x, train_set_y, w1, w2, b1, b2, learning_rate)
        w1, w2 = w["w1"], w["w2"]
        b1, b2 = b["b1"], b["b2"]
        if i % 500 == 0:
            L.append(l)
            print("损失函数Loss:", l)
    trainback(w1,w2 ,b1,b2, train_set_x, train_set_y)
    testback(w1, w2, b1, b2, test_set_x, test_set_y)
    # 绘制图
    #plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)  # 绘制散点图
    plt.plot(L)
    plt.ylabel('Loss')
    plt.xlabel('Number of training rounds')
    plt.title("learning_rate =" + str(learning_rate))
    plt.show()
