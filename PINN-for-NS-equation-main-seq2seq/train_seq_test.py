from pinn_model import *
import time
import pandas as pd
import os
from plot import *
from interpolate import *

# 训练代码主体
x, y, t, u, v, p, N_next_step, T = read_data(filename_data)
layer_mat = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
X_total_seq = total_data(x, y, t, u, v, p)

X_total = lagrange_start(filename_data,0.1) # 第一次lagrange用到0.1处的值
delta_t = 0.1
# 创建PINN模型实例，并将实例分配至对应设备
pinn_net = PINN_Net(layer_mat)
pinn_net = pinn_net.to(device)
# 损失函数和优化器
mse = torch.nn.MSELoss()
losses = []
#if os.path.exists(filename_save_model + str(0.1) + 's.pt'):
pinn_net.load_state_dict(torch.load(filename_load_model + str(0.1) + 's.pt', map_location=device))
print('ok')
#if os.path.exists(filename_loss):
#   loss_read = pd.read_csv('loss.csv', header=None)
#    losses = loss_read.values
#    losses = list(losses)
optimizer = torch.optim.Adam(pinn_net.parameters(), lr=0.00001)
epochs = 100
start_time = time.time()

# 选取batch size 此处也可使用data_loader
N = 50000
batch_size = 500
inner_iter = int(N / batch_size)


for step in range(T):
    losses = []
    # 得到分片矩阵 X_random
    X_random = shuffle_data_seq_lagrange(X_total) # 把一个小区间内的所有点打乱
    for epoch in range(epochs):
        for batch_iter in range(inner_iter):
            optimizer.zero_grad()
            # 在分片矩阵中随机取batch
            x_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 0].view(batch_size, 1)
            y_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 1].view(batch_size, 1)
            t_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 2].view(batch_size, 1)
            u_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 3].view(batch_size, 1)
            v_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 4].view(batch_size, 1)
            p_train = X_random[batch_iter*batch_size:((batch_iter+1)*batch_size), 5].view(batch_size, 1)

            # 定于zeros用于计算微分方程误差的MSE
            zeros = np.zeros((batch_size, 1))
            # 将batch从全集中clone出
            batch_t_x = x_train.clone().requires_grad_(True).to(device)
            batch_t_y = y_train.clone().requires_grad_(True).to(device)
            batch_t_t = t_train.clone().requires_grad_(True).to(device)
            batch_t_u = u_train.clone().requires_grad_(True).to(device)
            batch_t_v = v_train.clone().requires_grad_(True).to(device)
            batch_t_p = p_train.clone().requires_grad_(True).to(device)
            batch_t_zeros = torch.from_numpy(zeros).float().requires_grad_(True).to(device)
            # 删除不需要的内存空间
            del x_train, y_train, t_train, u_train, v_train, p_train, zeros

            # 调用f_equation函数进行损失函数各项计算
            u_predict, v_predict, p_predict, f_equation_x, f_equation_y = f_equation_inverse_Eular(batch_t_x, batch_t_y, batch_t_t,delta_t,
                                                                                             pinn_net, need_next_step=False)

            # 计算损失函数
            mse_predict = mse(u_predict, batch_t_u) + mse(v_predict, batch_t_v) + mse(p_predict, batch_t_p)
            mse_equation = mse(f_equation_x, batch_t_zeros) + mse(f_equation_y, batch_t_zeros)
            loss = mse_predict + mse_equation
            loss.backward()
            optimizer.step()

            with torch.autograd.no_grad():
                # 添加loss到losses
                # losses.append(loss.cpu().data.numpy().reshape(1, 1))
                losses.append(loss.item())
                print("Time:", (step+1)/10, "Epoch:", (epoch+1), "  Bacth_iter:", batch_iter + 1, " Training Loss:", round(float(loss.data), 8),'lam1 = ', pinn_net.lam1, "lam2 = ", pinn_net.lam2)
            # 每1个epoch保存状态（模型状态,loss,迭代次数）
                if (batch_iter + 1) % inner_iter == 0:
                    second = (step+1)/10
                    loss_save = pd.DataFrame(losses)
                    loss_save.to_csv(filename_loss + str(second) + 's.csv', index=False, header=False)
    second = (step+1)/10
    torch.save(pinn_net.state_dict(), filename_save_model + str(second) + 's.pt')
    x_temp,y_temp,t = cut_space(x, y, N_next_step, step) # 得到一开始撒的点和时间
    u0, v0, p0, u_next_step, v_next_step, p_next_step = f_equation_inverse_Eular(x_temp, y_temp, t, delta_t, pinn_net, need_next_step=True) # 得到delta_t时刻的初值，以及预测下一个delta_t时刻的值来进行插值

    U = torch.cat((u0, u_next_step), 1).cpu().detach().numpy().T
    V = torch.cat((v0, v_next_step), 1).cpu().detach().numpy().T
    P = torch.cat((p0, p_next_step), 1).cpu().detach().numpy().T
    X_total = lagrange(filename_data, U, V, P, (step+1)/10, delta_t, N_next_step) # 计算Lagrange插值并进行矩阵拼接

    del u_next_step,v_next_step,p_next_step,x_temp,y_temp

print("one oK")
