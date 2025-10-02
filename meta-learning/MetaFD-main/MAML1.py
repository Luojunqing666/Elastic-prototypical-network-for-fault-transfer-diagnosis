# 利用MAML进行回归预测
import torch
import learn2learn
import numpy as np
# # 准备数据
x = torch.tensor([[1.0],[2.0],[3.0]])
y= torch.tensor([[2.0],[4.0],[6.0]])

b=torch.tensor([[1,2],[2,3],[3,4]])
a = np.array([[1,  2],  [3,  4]])
z=~a
d=~b

# 设计线性模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

# 定义maml的内环和外环学习率
meta_lr = 0.005
fast_lr = 0.05

# 建立MAML模型
maml_qiao = learn2learn.algorithms.MAML(model, lr=fast_lr)

# 定义优化器
opt = torch.optim.Adam(maml_qiao.parameters(), meta_lr)

# 定义损失函数
loss = torch.nn.MSELoss()

#开始训练
for epoch in range(100):
    clone = maml_qiao.clone()
    #进行预测
    y_pred=clone(x)
    error = loss(y_pred, y)
    print(epoch, error)
    clone.adapt(error)
    opt.zero_grad()
    error.backward()
    opt.step()

