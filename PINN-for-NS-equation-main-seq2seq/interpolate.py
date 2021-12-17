import scipy.io
import numpy as np
import torch

# 这里没想好怎么放入Lagrange中
def lagrange_start(filename, t): 
    t_slice = t / 10
    data_mat = scipy.io.loadmat(filename)
    U_star = data_mat['U_star']  # N*dimension*T
    X_star = data_mat['X_star']  # N*dimension
    P_star = data_mat['p_star']  # N*T

    N = X_star.shape[0]

    # 将数据化为x,y,t---u,v,p(N*T,1)
    UU = U_star[:, 0, :].T #转置以后矩阵每行都是一个时间节点处的值
    VV = U_star[:, 1, :].T #转置以后矩阵每行都是一个时间节点处的值
    PP = P_star.T #转置以后矩阵每行都是一个时间节点处的值

    U = UU[0, :]
    V = VV[0, :]
    P = PP[0, :]
    for t_step in np.linspace(1,9,9) * t_slice:
        lagrange_0 = (t - t_step) / t
        lagrange_1 = t_step / t      

        u0 = UU[0,:]
        u1 = UU[1,:]
        val_u = np.dot([lagrange_1, lagrange_0], np.vstack((u1, u0)))
        U = np.vstack((U,val_u))
        
        v0 = VV[0,:]
        v1 = VV[1,:]
        val_v = np.dot([lagrange_1, lagrange_0], np.vstack((v1, v0)))
        V = np.vstack((V,val_v))

        p0 = PP[0,:]
        p1 = PP[1,:]
        val_p = np.dot([lagrange_1, lagrange_0], np.vstack((p1, p0)))
        P = np.vstack((P, val_p))

    XX = np.tile(X_star[:, 0:1], (1, 10)).T #转置以后矩阵每行都是一个时间节点处u,v,p对应的坐标值
    YY = np.tile(X_star[:, 1:2], (1, 10)).T
    TT = np.tile(np.linspace(0,9,10) * t_slice, (N, 1)).T
    t = TT.flatten()[:, None]
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    U = U.flatten()[:, None]
    V = V.flatten()[:, None]
    P = P.flatten()[:, None]
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    U= torch.tensor(U, dtype=torch.float32)
    V = torch.tensor(V, dtype=torch.float32)
    P = torch.tensor(P, dtype=torch.float32)
    data_total = torch.cat([x, y, t, U, V, P],1)
    return data_total

def lagrange(filename, UU, VV, PP, t, delta_t, N):
    U = UU[0, :]
    V = VV[0, :]
    P = PP[0, :]
    t_slice = delta_t / 10
    t_next = t + delta_t
    data_mat = scipy.io.loadmat(filename)
    X_star = data_mat['X_star']  # N*dimension

    for t_step in np.linspace(1,9,9) * t_slice + t:
        lagrange_0 = (t_next - t_step) / delta_t
        lagrange_1 = (t_step - t) / delta_t      

        val_u = np.dot([lagrange_0, lagrange_1], UU) #用矩阵乘法来表示Lagrange插值
        U = np.vstack((U,val_u))
        
        val_v = np.dot([lagrange_1, lagrange_0], VV)
        V = np.vstack((V,val_v))

        val_p = np.dot([lagrange_1, lagrange_0], PP)
        P = np.vstack((P, val_p))

    XX = np.tile(X_star[:, 0:1], (1, 10)).T #转置以后矩阵每行都是一个时间节点处u,v,p对应的坐标值
    YY = np.tile(X_star[:, 1:2], (1, 10)).T
    TT = np.tile(np.linspace(0,9,10) * t_slice + t, (N, 1)).T
    t = TT.flatten()[:, None]
    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    U = U.flatten()[:, None]
    V = V.flatten()[:, None]
    P = P.flatten()[:, None]
    
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.float32)
    U= torch.tensor(U, dtype=torch.float32)
    V = torch.tensor(V, dtype=torch.float32)
    P = torch.tensor(P, dtype=torch.float32)
    data_total = torch.cat([x, y, t, U, V, P],1)
    return data_total