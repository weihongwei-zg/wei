# conding:utf-8
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
    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes
def tanh(z):#tanh函数
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
def relu(z):#relu函数
    return np.maximum(0,z)
def tanh_1(z):#tanh函数的导数
    return 1-tanh(z)**2
def relu_1(z):#relu函数的导数
    return np.maximum(0, z/np.abs(z))
def sigmoid(z):
    return 1/(1+np.exp(-z))

def ward(L,n,m,dim):#对参数进行初始化
    np.random.seed(1)
    w = []
    b = []
    for i in range(0, L):
        if i != 0 and i != L - 1:
            # p = np.random.randn(dim, dim) *0.001
            p = np.random.randn(dim, dim) * np.sqrt(2 / dim)
        elif i == 0:
            # p = np.random.randn(dim, m) * 0.001
            p = np.random.randn(dim, m) * np.sqrt(2 / m)
        else:
            # p = np.random.randn(1, dim) * 0.001
            p = np.random.randn(1, dim) * np.sqrt(2 / dim)
        w.append(p)
        b.append(1)

    return w,b

def forward1(w,b,a,Y,L,m,dL,keep_prob):#前向传播
    z = []
    J = 0
    for i in range(0,L):
        zL = np.dot(w[i],a) + b[i]
        z.append(zL)
        a = relu(zL)
        if i < 2 or i==L-1:#增加dorpout正则化项
            p = np.random.rand(a.shape[0],a.shape[1]) < 1
            dL.append(p)
            a = np.multiply(a,p)/1
        else:
            p = np.random.rand(a.shape[0], a.shape[1]) < keep_prob
            dL.append(p)
            a = np.multiply(a, p)/keep_prob
    a = sigmoid(zL)

    J = (-1/m)*np.sum(1 * Y * np.log(a) + (1 - Y) * np.log(1 - a))#损失函数
    return z,a,J,dL
def forward2(w,b,a,Y,L,m):#测试时使用的前向传播
    z = []
    J = 0
    for i in range(0,L):
        zl = np.dot(w[i],a) + b[i]
        z.append(zl)
        a = relu(zl)
    a = sigmoid(zl)

    J = (-1/m)*np.sum(1 * Y * np.log(a) + (1 - Y) * np.log(1 - a))#损失函数
    return z,a,J
def backward(w,b,X,Y,learning,m,L,keep_prob):#反向传播
    dL = []#表示消除的神经元，是一个布尔值矩阵
    z,a,J,dL = forward1(w,b,X,Y,L,m,dL,keep_prob)

    for i in range(L-1,0,-1):
        if i == L-1:
            dz = a-Y
        else:
            dz = np.dot(w[i+1].T,dz)*(relu_1(z[i])*dL[i])
        dw = 1/m * np.dot(dz,(relu(z[i-1])*dL[i]).T)
        db = 1/m * np.sum(dz,axis=1,keepdims=True)
        w[i] -= learning*dw
        b[i] -= learning*db

        #b[i] = np.mean(b[i] - learning*db)
    dz = np.dot(w[1].T,dz)*(relu_1(z[0])*dL[0])
    dw = 1/m * np.dot(dz,X.T)
    db = 1/m * np.sum(dz,axis=1,keepdims=True)

    w[0] -= learning*dw
    b[0] -= learning * db
    #b[0] = np.mean(b[0] - learning*db)
    return w,b,J

def train(x,y,w,b,L,m):#查看训练集准确率
    A = np.zeros(shape=(1, x.shape[1]))
    z, a,J = forward2(w, b, x, y,L,m)

    for i in range(x.shape[1]):
        # A.append(0 if a[0,i] <0.5 else 1)
        A[0, i] = 0 if a[0, i] < 0.5 else 1
    lop = 100 * (1 - np.mean(np.abs(y - A)))
    print("训练集准确性：{0}%".format(lop))
    return 0

def text(x,y,w,b,L,m):#查看测试集准确率
    A = np.zeros(shape=(1, x.shape[1]))
    z, a,J = forward2(w, b, x, y, L, m)

    for i in range(x.shape[1]):
        # A.append(0 if a[0,i] <0.5 else 1)
        A[0, i] = 0 if a[0, i] < 0.5 else 1
    lop = 100 * (1 - np.mean(np.abs(y - A)))
    print("测试集准确性：{0}%".format(lop))
    return 0

if __name__ == "__main__":
    L = 5#神经网络层数
    dim = 15#隐藏层节点个数
    learning = 0.008#学习率
    loss= []#损失函数
    keep_prob = 0.8#dropout正则化参数
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    train_set_x = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T / 255  # 降维,化为区间（0，1）内的数
    test_set_x = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T / 255  # 降维,化为区间（0，1）内的数
    print("训练集降维后的维度： " + str(train_set_x.shape))
    print("训练集_标签的维数 : " + str(train_set_y.shape))
    print("测试集降维后的维度: " + str(test_set_x.shape))
    print("测试集_标签的维数 : " + str(test_set_y.shape))
    print()
    w,b = ward(L,train_set_x.shape[1],train_set_x.shape[0],dim)
    for i in range(2000):
        w,b,J = backward(w,b,train_set_x,train_set_y,learning,train_set_x.shape[1],L,keep_prob)
        if i % 500 == 0:
            print("loss：",J)
            loss.append(J)
    plt.plot(loss)
    plt.show()
    train(train_set_x,train_set_y,w,b,L,train_set_x.shape[1])
    text(test_set_x,test_set_y,w,b,L,test_set_x.shape[1])







