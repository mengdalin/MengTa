
# coding: utf-8

# 第一個Python程式

# In[3]:


#單行註解
'''
多行註解
多行註解
多行註解
'''
a=5
b=3
a+b


# 第二個Python程式

# In[5]:


print("Hello world")


# 第三個Python程式(變數的設定)

# In[7]:


x=1
y=10*x
x=x+y
print(x)
print(y)


# In[1]:


#平行設定
x,y,z=3,4,5
print(x)
print(y)
print(z)


# In[2]:


#交換2個變數
x,y=3,4
x,y=y,x
print(x)
print(y)


# In[4]:


#複合設定運算
x,y,z=3,4,5
x+=1 #x=x+1
y*=2 #y=y*2
z**=3 #z=z**3 #z=z^3
print(x,y,z)


# In[6]:


#算三角形面積
import math
a,b,c=3,4,5
s=(a+b+c)/2
area=math.sqrt(s*(s-a)*(s-b)*(s-c))
print(area)

a,b,c=12,33,25
s=(a+b+c)/2
area=math.sqrt(s*(s-a)*(s-b)*(s-c))
print(area)


# In[7]:


#Python 的動態類型整合
#變數使用前不需要宣告資料類型，使用時只要根據變數存放的資料決定其資料類型
x=254
print(type(x))
x="write"
print(type(x))
x=254.0
print(type(x))
x=True
print(type(x))


# In[10]:


#2,8,16進位
print(0o137) #o為8進位
print(0b111) #b為2進位
print(0xff)  #x為16進位
print(type(28346283742874))
print(type(0o137))


# In[11]:


import math
print(4*(math.pi*4.5*4.5*4.5)/3)


# In[13]:


#浮點數的誤差
x=3.141592627
print(x-3.14)
print(2.1-2.0)


# In[22]:


#畫圖(折線圖)
import matplotlib.pyplot as pt
x=[1,2,3,4,5]
y=[7,2,3,5,9]
z=[1,3,5,7,10]
pt.plot(x,y,color="red",label="March")
pt.plot(x,z,"--",label="April") #虛線;-- ,三角形;^ ,正方形;s
pt.legend() #show出標籤
pt.show()


# In[26]:


#畫圖(長條圖)
import matplotlib.pyplot as pt
x=[1,2,3,4,5]
y=[7,2,3,5,9]
pt.bar(x,y)
pt.show()


# In[27]:


#畫圖(點陣圖)
import matplotlib.pyplot as pt
x=[1,2,3,4,5]
y=[7,2,3,5,9]
pt.scatter(x,y)
pt.show()


# In[35]:


#畫圖(點陣圖 隨機50個)
import numpy as np
import matplotlib.pyplot as pt
x=np.random.random(50)
y=np.random.random(50)
pt.scatter(x,y)
pt.show()


# In[40]:


#畫圖(曲線圖)
import numpy as np
import matplotlib.pyplot as pt
x=np.arange(0,360)
y=np.sin(x*np.pi/180)
z=np.cos(x*np.pi/180)
pt.xlim(0,360)
pt.ylim(-1.2,1.2)
pt.title("Sin & Cos Wave")
pt.xlabel("Degree")
pt.ylabel("Value")
pt.plot(x,y,label="Sin")
pt.plot(x,z,label="Cos")
pt.legend()
pt.show()


# In[41]:


#3D繪圖
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()


# In[6]:


from sklearn import datasets, cluster, metrics
iris=datasets.load_iris()
#print(iris["DESCR"])
#print(iris["data"])
print(iris["target"])
iris_kmeans=cluster.KMeans(n_clusters=3).fit(iris["data"])
print(iris_kmeans.labels_)
silhouette_avg=metrics.silhouette_score(iris["data"],iris_kmeans.labels_)
print(silhouette_avg)


# In[8]:


from sklearn import datasets, cluster, metrics
import matplotlib.pyplot as plt
iris=datasets.load_iris()
silhouette_avgs=[]
ks=range(2,10)
for k in ks:
    iris_kmeans=cluster.KMeans(n_clusters=k).fit(iris["data"])
    silhouette_avg=metrics.silhouette_score(iris["data"],iris_kmeans.labels_)
    silhouette_avgs.append(silhouette_avg)
plt.bar(ks,silhouette_avgs)
plt.show()


# In[10]:


from sklearn import datasets
digits=datasets.load_digits()
print(digits["DESCR"])
print(digits["data"])
print(digits["target"])


# In[12]:


#畫圖
from sklearn import datasets
import matplotlib.pyplot as plt
digits=datasets.load_digits()
plt.figure(1,figsize=(3,3))
plt.imshow(digits.images[0],cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


# In[5]:


from sklearn import datasets, svm
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
digits=datasets.load_digits()
X_train, X_test, y_train, y_test, image_train, image_test = train_test_split(digits.data, digits.target, digits.images, test_size=0.25, random_state=42 )
svc_model=svm.SVC(gamma=0.01, C=100, kernel='linear')
svc_model.fit(X_train, y_train)
predicted=svc_model.predict(X_test)
images_and_predictions = list(zip(image_test, predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:5]):
    plt.subplot(1, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.binary)
    plt.title('Predicted: ' + str(prediction))
plt.show()


# In[1]:


from sklearn import datasets
boston=datasets.load_boston()
#print(boston.DESCR)print(boston.target)print(boston.data)
#CRIM(犯罪率) ZN(房屋大於25000ft比率)
#INDUS(住宅比率) CHAS(有無臨河) NOX(空汙比率) RM(房間數)
#AGE(自有住宅比率) PTRATIO(小學老師比例) B(黑人比率)
#LSTAT(低收入比率) MEDV(受雇者收入)
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict
lr = linear_model.LinearRegression()
predict = cross_val_predict(lr, boston.data, boston.target, cv=10)
#print(predict)
import matplotlib.pyplot as plt
plt.scatter(boston.target, predict)
y=boston.target
plt.plot([y.min(), y.max()],[y.min(),y.max()] ,'k--',lw=4)
plt.plot()
plt.show()


# In[2]:


from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
data=datasets.fetch_olivetti_faces()
#print(data.DESCR)
#print(data.target)
#print(data.data)
#plt.imshow(data.images[0],cmap='gray',interpolation='nearest')
#plt.show()
#把影像變成一列
targets=data.target
data=data.images.reshape(len(data.images),-1)
#訓練資料30張臉(300張圖片)，測試資料10張臉(100張圖片)
train=data[targets<30]
test=data[targets>=30]
# 從100張測試影像中,亂數選5張出來，變數test的大小變成(5,4096)
n_faces = 5
from sklearn.utils import check_random_state
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]
#把每張訓練影像和測試影像都切割成上下兩部分: X人臉上半部分
#， Y人臉下半部分。
n_pixels = data.shape[1]
X_train = train[:, :(n_pixels + 1) // 2]
y_train = train[:, n_pixels // 2:]
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]
#決定預測的演算法
from sklearn.linear_model import LinearRegression
ESTIMATORS = {
    "Linear regression": LinearRegression(),
}
y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train) #模型訓練
    y_test_predict[name] = estimator.predict(X_test) 
    #模型預測
# Plot the completed faces
image_shape = (64, 64)
n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)
for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))
    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title="true faces")
    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest")
    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))
        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j, title=est)
        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest")
plt.show()





























from sklearn import datasets
from sklearn.utils import check_random_state
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = datasets.fetch_olivetti_faces()
#print(data.DESCR)
#print(data.target)
#print(data.data)
targets = data.target
data = data.images.reshape((len(data.images), -1)) #把影像變成一列
train = data[targets < 30]
test = data[targets >= 30]
# 測試影像從100張亂數選5張出來，變數test的大小變成(5,4096)
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]

#把每張訓練影像和測試影像都切割成上下兩部分: X人臉上半部分， Y人臉下半部分。
n_pixels = data.shape[1]
X_train = train[:, :(n_pixels + 1) // 2]
# Lower half of the faces
y_train = train[:, n_pixels // 2:]
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]

ESTIMATORS = {
    "Linear regression": LinearRegression(),
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

# Plot the completed faces
image_shape = (64, 64)

n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)

for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))

    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                          title="true faces")

    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                              title=est)

        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

plt.show()


# In[3]:


from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics
iris = load_iris()
iris_X = iris.data
iris_y = iris.target
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
test_y_predicted = clf.predict(test_X)
print(test_y_predicted)
print(test_y)
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)

from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# 讀入鳶尾花資料
iris = load_iris()
iris_X = iris.data
iris_y = iris.target

# 切分訓練與測試資料
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)

# 建立分類器
clf = neighbors.KNeighborsClassifier()
iris_clf = clf.fit(train_X, train_y)

# 預測
test_y_predicted = iris_clf.predict(test_X)
print(test_y_predicted)

# 標準答案
print(test_y)
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)

from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

# 讀入鳶尾花資料
iris = load_iris()
iris_X = iris.data
iris_y = iris.target

# 切分訓練與測試資料
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)

# 選擇 k
range = np.arange(1, round(0.2 * train_X.shape[0]) + 1)
accuracies = []

for i in range:
    clf = neighbors.KNeighborsClassifier(n_neighbors = i)
    iris_clf = clf.fit(train_X, train_y)
    test_y_predicted = iris_clf.predict(test_X)
    accuracy = metrics.accuracy_score(test_y, test_y_predicted)
    accuracies.append(accuracy)

# 視覺化
plt.scatter(range, accuracies)
plt.show()
appr_k = accuracies.index(max(accuracies)) + 1
print(appr_k)

