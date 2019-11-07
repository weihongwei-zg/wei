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
    Vdw, Vdb, Sdw, Sdb = [],[],[],[]#优化算法的值
    for i in range(0, L):
        if i != 0 and i != L - 1:
            #p = np.random.randn(dim, dim) *0.001
            p = np.random.randn(dim, dim) * np.sqrt(2 / dim)
        elif i == 0:
            #p = np.random.randn(dim, m) * 0.001
            p = np.random.randn(dim, m) * np.sqrt(2 / m)
        else:
            #p = np.random.randn(1, dim) * 0.001
            p = np.random.randn(1, dim) * np.sqrt(2 / dim)
        w.append(p)
        b.append(1)
        Vdw.append(0)#初始化为0
        Vdb.append(0)
        Sdw.append(0)
        Sdb.append(0)

    return w, b,Vdw,Vdb,Sdw,Sdb

def forward(w,b,a,Y,L,m,lambd):#前向传播
    z = []
    J = 0
    add = 0
    for i in range(0, L):
        zl = np.dot(w[i], a) + b[i]
        add += np.sum((lambd / (2 * m)) * np.dot(w[i], w[i].T))#L2正则化项
        z.append(zl)
        a = relu(zl)
    a = sigmoid(zl)

    J = (-1/m)*np.sum(1 * Y * np.log(a) + (1 - Y) * np.log(1 - a)) + add  # 损失函数
    return z, a, J
def backward(w,b,X,Y,m,L,lambd):#反向传播
    z,a,J = forward(w,b,X,Y,L,m,lambd)
    dw,db = [],[]
    for i in range(L - 1, 0, -1):
        if i == L - 1:
            dz = a - Y
        else:
            dz = np.dot(w[i + 1].T, dz) * relu_1(z[i])
        Dw = 1 / m * (np.dot(dz, relu(z[i - 1]).T)) + (lambd / m) * w[i]
        Db = 1 / m * np.sum(dz, axis=1, keepdims=True)
        dw.append(Dw)
        db.append(Db)

    dz = np.dot(w[1].T, dz) * relu_1(z[0])
    Dw = 1 / m * np.dot(dz, X.T) + (lambd / m) * w[0]
    Db = 1 / m * np.sum(dz, axis=1, keepdims=True)
    dw.append(Dw)
    db.append(Db)
    return dw, db, J
def back_momentum(w,b,X,Y,learning,m,L,lambd,beta,Vdw,Vdb):#使用momentum梯度下降的反向传播
    dw, db, J = backward(w,b,X,Y,m,L,lambd)
    # 不使用偏差修正不影响最后结果
    for i in range(0,L):#注意dw和db是由后往前存的，位置与w和b相反
        Vdw[i] = beta * Vdw[i] + (1 - beta) * dw[L-i-1]
        Vdb[i] = beta * Vdb[i] + (1 - beta) * db[L-i-1]
        w[i] = w[i] - learning * Vdw[i]
        b[i] = b[i] - learning * Vdb[i]

    return w,b,J,Vdw,Vdb
def back_RMSprop(w,b,X,Y,learning,m,L,lambd,beta,Sdw,Sdb):#使用RMSprop梯度下降的反向传播
    dw, db, J = backward(w,b,X,Y,m,L,lambd)
    # 不使用偏差修正不影响最后结果
    for i in range(0,L):#注意dw和db是由后往前存的，位置与w和b相反
        Sdw[i] = beta * Sdw[i] + (1 - beta) * np.power(dw[L-i-1],2)
        Sdb[i] = beta * Sdb[i] + (1 - beta) * np.power(db[L-i-1],2)
        w[i] = w[i] - learning * dw[L-i-1] / (np.power(Sdw[i],1/2) + 0.00000001)#加上一个较小的值，防止整个值变成无穷大
        b[i] = b[i] - learning * db[L-i-1] / (np.power(Sdb[i],1/2) + 0.00000001)#加上一个较小的值，防止整个值变成无穷大

    return w,b,J,Sdw,Sdb
def back_Adam(w,b,X,Y,learning,m,L,lambd,beta,Vdw,Vdb,Sdw,Sdb,t):#使用Adam梯度下降的反向传播
    dw, db, J = backward(w,b,X,Y,m,L,lambd)
    # 通常使用偏差修正
    for i in range(0,L):#注意dw和db是由后往前存的，位置与w和b相反
        Vdw[i] = beta * Vdw[i] + (1 - beta) * dw[L - i - 1]
        Vdb[i] = beta * Vdb[i] + (1 - beta) * db[L - i - 1]
        Sdw[i] = beta * Sdw[i] + (1 - beta) * np.power(dw[L-i-1],2)
        Sdb[i] = beta * Sdb[i] + (1 - beta) * np.power(db[L-i-1],2)
        '''
        Vdw[i] /= 1 - np.power(beta, t)#t表示代数（所有mini_batch训练一次称为一代）
        Vdb[i] /= 1 - np.power(beta, t)
        Sdw[i] /= 1 - np.power(beta, t)
        Sdb[i] /= 1 - np.power(beta, t)
        '''
        w[i] = w[i] - learning * Vdw[i] / (np.power(Sdw[i],1/2) + 0.00000001)#加上一个较小的值，防止整个值变成无穷大
        b[i] = b[i] - learning * Vdb[i] / (np.power(Sdb[i],1/2) + 0.00000001)#加上一个较小的值，防止整个值变成无穷大

    return w,b,J,Vdw,Vdb,Sdw,Sdb

def train(x,y,w,b,L,m,lambd):#查看训练集准确率
    A = np.zeros(shape=(1, x.shape[1]))
    z, a,J = forward(w, b, x, y,L,m,lambd)

    for i in range(x.shape[1]):
        # A.append(0 if a[0,i] <0.5 else 1)
        A[0, i] = 0 if a[0, i] < 0.5 else 1
    lop = 100 * (1 - np.mean(np.abs(y - A)))
    print("训练集准确性：{0}%".format(lop))
    return 0

def text(x,y,w,b,L,m,lambd):#查看测试集准确率
    A = np.zeros(shape=(1, x.shape[1]))
    z, a,J = forward(w, b, x, y, L, m,lambd)

    for i in range(x.shape[1]):
        # A.append(0 if a[0,i] <0.5 else 1)
        A[0, i] = 0 if a[0, i] < 0.5 else 1
    lop = 100 * (1 - np.mean(np.abs(y - A)))
    print("测试集准确性：{0}%".format(lop))
    return 0

if __name__ == "__main__":
    L = 10#神经网络层数
    dim = 10#隐藏层节点个数
    learning_rate = 0.01#学习率
    loss= []#损失函数
    lambd = 0.01 # L2正则化参数
    beta = 0.9#β值；1/（1-β）表示平均前多少轮的指数加权平均数
    decay_rate = 0.00009#学习率衰减率
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    train_set_x = train_set_x_orig.reshape((train_set_x_orig.shape[0], -1)).T / 255  # 降维,化为区间（0，1）内的数
    test_set_x = test_set_x_orig.reshape((test_set_x_orig.shape[0], -1)).T / 255  # 降维,化为区间（0，1）内的数
    print("训练集降维后的维度： " + str(train_set_x.shape))
    print("训练集_标签的维数 : " + str(train_set_y.shape))
    print("测试集降维后的维度: " + str(test_set_x.shape))
    print("测试集_标签的维数 : " + str(test_set_y.shape))
    print()
    w,b,Vdw,Vdb,Sdw,Sdb = ward(L,train_set_x.shape[1],train_set_x.shape[0],dim)#vdw表示momentum，Sdw表示RMSprop
    for i in range(0,2000):
        #w,b,J,Vdw,Vdb = back_momentum(w,b,train_set_x,train_set_y,learning_rate,train_set_x.shape[1],L,lambd,beita,Vdw,Vdb)
        #w,b,J,Vdw,Vdb = back_RMSprop(w, b, train_set_x,train_set_y,learning_rate,train_set_x.shape[1],L,lambd,beita, Sdw, Sdb)
        w,b,J,Vdw,Vdb,Sdw,Sdb = back_Adam(w,b,train_set_x,train_set_y,learning_rate,train_set_x.shape[1],L,lambd,beta,Vdw,Vdb, Sdw, Sdb,i)
        learning_rate = learning_rate * (1 / (1 + i*decay_rate) )#使用学习率衰减
        if i % 50 == 0:
            print("loss：",J)
            loss.append(J)
    plt.plot(loss)#打印损失函数
    plt.show()
    train(train_set_x,train_set_y,w,b,L,train_set_x.shape[1],lambd)
    text(test_set_x,test_set_y,w,b,L,test_set_x.shape[1],lambd)


