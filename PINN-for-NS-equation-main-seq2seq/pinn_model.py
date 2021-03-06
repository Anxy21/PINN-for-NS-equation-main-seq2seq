import scipy.io
import numpy as np
import torch
import torch.nn as nn

# 定义PINN网络模块，包括数据读取函数，参数初始化
# 正问题和反问题的偏差和求导函数
filename_load_model = './NS_model_Euler/NS_model_train_'
filename_save_model = './NS_model_Euler/NS_model_train_'
filename_data = './cylinder_nektar_wake.mat'
filename_loss = './loss_Euler/loss_'
# 训练设备为GPU还是CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 读取原始数据,并转化为x,y,t--u,v,p(N*T,1),返回值为Tensor类型
def read_data(filename):
    # 读取原始数据
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']  # N*dimension*T
    X_star = data_mat['X_star']  # N*dimension
    T_star = data_mat['t']  # T*1
    P_star = data_mat['p_star']  # N*T
    # 读取坐标点数N和时间步数T
    N = X_star.shape[0]
    T = T_star.shape[0]

    # 将数据化为x,y,t---u,v,p(N*T,1)
    XX = np.tile(X_star[:, 0:1], (1, T)).T #转置以后矩阵每行都是一个时间节点处u,v,p对应的坐标值
    YY = np.tile(X_star[:, 1:2], (1, T)).T #转置以后变成时间层
    TT = np.tile(T_star, (1, N))
    UU = U_star[:, 0, :].T #转置以后矩阵每行都是一个时间节点处的值
    VV = U_star[:, 1, :].T #转置以后矩阵每行都是一个时间节点处的值
    PP = P_star.T #转置以后矩阵每行都是一个时间节点处的值
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = UU.flatten()[:, None]
    v = VV.flatten()[:, None]
    p = PP.flatten()[:, None]
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    u = torch.tensor(u, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)

    U = UU[0:2,:].T
    V = VV[0:1,:].T
    P = PP[0:1,:].T
    return x, y, t, u, v, p, N, T


# 定义网络结构,由layer列表指定网络层数和神经元数
class PINN_Net(nn.Module):
    def __init__(self, layer_mat):
        super(PINN_Net, self).__init__()
        self.layer_num = len(layer_mat) - 1
        self.base = nn.Sequential()
        for i in range(0, self.layer_num - 1):
            self.base.add_module(str(i) + "linear", nn.Linear(layer_mat[i], layer_mat[i + 1]))
            # nn.init.kaiming_normal()
            self.base.add_module(str(i) + "Act", nn.Tanh())
        self.base.add_module(str(self.layer_num - 1) + "linear",
                             nn.Linear(layer_mat[self.layer_num - 1], layer_mat[self.layer_num]))
        #self.lam1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.lam1 = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.lam2 = nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.Initial_param()

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], 1).requires_grad_(True)
        predict = self.base(X)
        return predict

    #对参数进行初始化
    def Initial_param(self):
        for name, param in self.base.named_parameters():
            if name.endswith('weight'):
                nn.init.xavier_normal_(param)
            elif name.endswith('bias'):
                nn.init.zeros_(param)

# 定义偏微分方程（的偏差）inverse为反问题
def f_equation_inverse(x, y, t, pinn_example):
    lam1 = pinn_example.lam1
    lam2 = pinn_example.lam2
    predict_out = pinn_example.forward(x, y, t)
    # 获得预测的输出psi,p
    psi = predict_out[:, 0].reshape(-1, 1)
    p = predict_out[:, 1].reshape(-1, 1)
    # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
    u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
    v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    # 计算偏微分方程的残差
    f_equation_x = u_t + (u * u_x + v * u_y) + lam1 * p_x - lam2 * (u_xx + u_yy)
    f_equation_y = v_t + (u * v_x + v * v_y) + lam1 * p_y - lam2 * (v_xx + v_yy)
    return u, v, p, f_equation_x, f_equation_y

def f_equation_inverse_Eular(x, y, t, delta_t, pinn_example,need_next_step=False):
    lam1 = pinn_example.lam1
    lam2 = pinn_example.lam2
    t_next_step = torch.ones((len(x),1)).to(device) * delta_t + t
    predict_out = pinn_example.forward(x, y, t)
    predict_out_next_step = pinn_example.forward(x,y,t)
    # 获得预测的输出psi,p
    psi = predict_out[:, 0].reshape(-1, 1)
    p = predict_out[:, 1].reshape(-1, 1)
    p_next_step = predict_out_next_step[:, 1].reshape(-1, 1)
    # 通过自动微分计算各个偏导数,其中.sum()将矢量转化为标量，并无实际意义
    u = torch.autograd.grad(psi.sum(), y, create_graph=True)[0]
    v = -torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    v_t = torch.autograd.grad(v.sum(), t, create_graph=True)[0]
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0]
    # 计算偏微分方程的残差
    f_equation_x = u_t + (u * u_x + v * u_y) + lam1 * p_x - lam2 * (u_xx + u_yy)
    f_equation_y = v_t + (u * v_x + v * v_y) + lam1 * p_y - lam2 * (v_xx + v_yy)
    # 计算下一个delta_t节点u,v的值
    f_next_step_x = u + delta_t * (-(u * u_x + v * u_y) - lam1 * p_x + lam2 * (u_xx + u_yy)) #欧拉前向插值
    f_next_step_y = v + delta_t * (-(u * v_x + v * v_y) - lam1 * p_y + lam2 * (v_xx + v_yy)) #欧拉前向插值
    if need_next_step:
        return u,v,p,f_next_step_x,f_next_step_y,p_next_step
    else:
        return u, v, p, f_equation_x, f_equation_y
# 得到撒的点，感觉这个函数可以去掉
def cut_space(x,y, N_next_step,step):
    x_cut = x[N_next_step*step : N_next_step*(step+1)].to(device).requires_grad_(True)
    y_cut = y[N_next_step*step : N_next_step*(step+1)].to(device).requires_grad_(True)
    t = torch.ones((len(x_cut),1)).to(device).requires_grad_(True) * (step + 1) / 10
    return x_cut, y_cut,t

def total_data(x, y, t, u, v, p):
    X_total = torch.cat([x, y, t, u, v, p], 1)
    return X_total

def shuffle_data_seq(X_total, N, step):
    X_slice = X_total[N*step : N*(step+1), :]
    X_slice_arr = X_slice.data.numpy()
    np.random.shuffle(X_slice_arr)
    X_slice_random = torch.tensor(X_slice_arr)
    return X_slice_random


def shuffle_data_seq_lagrange(X_total):
    X_total_arr = X_total.data.numpy()
    np.random.shuffle(X_total_arr)
    X_total_random = torch.tensor(X_total_arr)
    return X_total_random

