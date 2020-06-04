import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve
import itertools

#%%
#  数据加载
data = pd.read_csv("./creditcard.csv")
#  数据探索
print(data.describe())
#  设置plt正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 绘制类别分布
plt.figure()
ax = sns.countplot(x='Class',data=data)
plt.title('类别分布')
plt.show()

#%%
#  显示交易笔数，欺诈交易笔数
num = len(data)
num_fraud = len(data[data['Class']==1])
print(f'总交易笔数：{num}')
print(f'欺诈交易笔数：{num_fraud}')
print(f'诈骗交易比例：{num_fraud/num:.6f}')

#%%
#  欺诈和正常交易可视化
f,(ax1,ax2) = plt.subplots(2,1,sharex=True,figsize=(15,8))
bins=50
ax1.hist(data.Time[data.Class==1],bins=bins,color='deeppink')
ax1.set_title('诈骗交易')
ax2.hist(data.Time[data.Class==0],bins=bins,color='deepskyblue')
ax2.set_title('正常交易')
plt.xlabel('时间')
plt.ylabel('交易次数')
plt.show()

#%%
#  对amount进行数据规范化
data['Amount_Norm'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
#  特征选择
y = np.array(data.Class.tolist())
data2 = data.drop(['Time','Amount','Class'],axis=1)
x = np.array(data2.values)
#  准备训练集和测试集
train_x, test_x, train_y, test_y = train_test_split (x, y, test_size = 0.1, random_state = 33)

#%%
clf = LogisticRegression()
clf.fit(train_x,train_y)
predict_y = clf.predict(test_x)
#  预测样本的置信分数
score_y = clf.decision_function(test_x)
#  计算混淆矩阵，并显示
cm = confusion_matrix(test_y,predict_y)
class_names = [0,1]
#  显示混淆矩阵
plt.figure()
plt.imshow(cm,interpolation='nearest',cmap='Blues')
plt.title('逻辑回归 混淆矩阵')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names,rotation=0)
plt.yticks(tick_marks,class_names)

threshold = cm.max()/2
for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
    plt.text(j,i,cm[i,j],horizontalalignment='center',color='white' if cm[i,j]>threshold else 'black')
    
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

#%%
#  显示模型评估分数
tp = cm[1,1]
fn = cm[1,0]
fp = cm[0,1]
tn = cm[0,0]
#  求出精确率,召回率,F1值
precision_rate = tp/(tp+fp)
recall_rate = tp/(tp+fn)
F1_value =  (precision_rate * recall_rate)/(precision_rate + recall_rate) * 2
print(f'精确率：{precision_rate:.3f},\t召回率：{recall_rate:.3f},\tF1值：{F1_value:.3f}')
#  计算精确率，召回率，阈值用于可视化
precision,recall,thresholds = precision_recall_curve(test_y,score_y)
plt.step(recall,precision,color='b',alpha=0.2,where='post')
plt.fill_between(recall,precision,step='post',alpha=0.2,color='b')
plt.plot(recall,precision,linewidth=2)
plt.xlim([0.0,1])
plt.ylim([0.0,1.05])
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('精确率-召回率 曲线')
plt.show()






