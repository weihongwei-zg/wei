#conding:utf-8
import numpy as np
import h5py
import matplotlib.pyplot as plt
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # 保存的是训练集里面的图像数据（本训练集有209张64x64的图像）。
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # 保存的是训练集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # 保存的是测试集里面的图像数据（本训练集有50张64x64的图像）。
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # 保存的是测试集的图像对应的分类值（【0 | 1】，0表示不是猫，1表示是猫）。
    classes = np.array(test_dataset["list_classes"][:]) # 保存的是以bytes类型保存的两个字符串数据，数据为：[b’non-cat’ b’cat’]。
    train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    print ("训练集_图片的维数 : " + str(train_set_x_orig.shape))
    print ("训练集_标签的维数 : " + str(train_set_y.shape))
    print ("测试集_图片的维数: " + str(test_set_x_orig.shape))
    print ("测试集_标签的维数: " + str(test_set_y.shape))
    print ()
    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes
def sigmoid(z):#激活函数
    return 1/(1+np.exp(-z))

def ward(dim):#初始化参数w,b为0
    w = np.zeros(shape = (dim,1),dtype = float)

    b = 0
    gid = {
        "w" : w,
        "b" : b
    }
    return gid
def backward(w,b,X,Y):#前，反向传播
    m = X.shape[1]
    z = np.dot(w.T,X) + b
    A = sigmoid(z)
    L = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) #计算成本
    L = np.squeeze(L)#.squeeze()函数，去掉维度为1的shape
    
    dz = A - Y
    dw = (1 / m)*np.dot(X,dz.T)
    db = (1 / m)*np.sum(dz)#对dz中所有元素求和，取平均值
    rid = {
        "dw" : dw,
        "db" : db
    }
    return rid,L

def gradientdescent(w,b,X,Y,learning_rate):#梯度下降优化参数w，b
    l = []
    for i in range(3000):
        rid,L = backward(w,b,X,Y)
        dw,db = rid["dw"],rid["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            print("训练轮数：{0}-------误差值（损失函数）：{1}".format(i,L))
            
        if i % 100 == 0:#记录成本
            l.append(L)
    w = w.reshape(X.shape[0],1)
    gid = {
        "w" : w,
        "b" : b
    }
    return gid,l

def text(w,b,X):#通过优化后的参数w,b预测出 y的值
    m  = X.shape[1] #图片的数量
    y = np.zeros((1,m),dtype = float) 
    
    z = np.dot(w.T,X) + b
    
    for i in range(z.shape[1]):
        y[0,i] = 1 if z[0,i] > 0.5 else 0
    return y

def textback(w,b,X,Y):#测试
    y = text(w,b,X)
    lop = 100*(1 - np.mean(np.abs(y - Y)))
    print("测试集准确性：{0}%".format(lop))
    return 0

def trainback(w, b, X, Y):  # 测试
    y = text(w, b, X)
    lop = 100 * (1 - np.mean(np.abs(y - Y)))
    print("训练集准确性：{0}%".format(lop))
    return 0
if __name__ == '__main__':
    train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()
    
    train_set_x = train_set_x_orig.reshape((train_set_x_orig.shape[0],-1)).T / 255 #降维,化为区间（0，1）内的数
    test_set_x = test_set_x_orig.reshape((test_set_x_orig.shape[0],-1)).T / 255 #降维,化为区间（0，1）内的数
    
    print ("训练集降维后的维度： " + str(train_set_x.shape))
    print ("训练集_标签的维数 : " + str(train_set_y.shape))
    print ("测试集降维后的维度: " + str(test_set_x.shape))
    print ("测试集_标签的维数 : " + str(test_set_y.shape))
    print ()
    gid = ward(train_set_x.shape[0])
    w,b = gid["w"],gid["b"]
    gid,l = gradientdescent(w,b,train_set_x,train_set_y,0.0075)
    w,b = gid["w"],gid["b"]
    trainback(w,b,train_set_x,train_set_y)
    textback(w,b,test_set_x,test_set_y)
    
    #绘制图
    l = np.squeeze(l)
    plt.plot(l)
    plt.ylabel('Loss')
    plt.xlabel('Number of training rounds')
    plt.title("learning_rate =" + str(0.005))
    plt.show()
    
    

