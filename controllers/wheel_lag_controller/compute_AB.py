import numpy as np

def compute_AB(ll, lr, lwl, lwr, lbl, lbr, Ill, Ilr):
    """
    计算 WBR 系统 A(10,10) 和 B(10,4) 矩阵。
    
    输入: ll, lr - 左右腿腿长 [m]
          lwl, lwr - 驱动轮到腿质心距离 [m]
          lbl, lbr - 腿质心到机体关节距离 [m]
          Ill, Ilr - 腿转动惯量 [kg*m^2]
    输出: A(10,10), B(10,4)
    """

    # 固定参数
    R_w = 0.1
    R_l = 0.2545
    l_c = 0
    m_w = 1
    m_l = 0.88
    m_b = 6.21
    I_w = 0.00263333
    I_b = 0.0836491
    I_z = 0.1525
    g = 9.8

    # M 矩阵 (5x5): M * ddq + K * q + E * u = 0
    M = np.zeros((5, 5))
    M[0,0] = (11*lbl)/125 + (728261891589740913*ll)/5764607523034234880
    M[0,2] = (22*lbl*lwl)/25 - Ill
    M[1,1] = (11*lbr)/125 + (728261891589740913*lr)/5764607523034234880
    M[1,3] = (22*lbr*lwr)/25 - Ilr
    M[2,0] = -75636449737972087817/1441151880758558720000
    M[2,1] = -75636449737972087817/1441151880758558720000
    M[2,2] = - (621*ll)/2000 - (11*lwl)/125
    M[2,3] = - (621*lr)/2000 - (11*lwr)/125
    M[3,4] = -3013776444719019/36028797018963968
    M[4,0] = 8605972816799302745057/234734818337954044313600
    M[4,1] = -8605972816799302745057/234734818337954044313600
    M[4,2] = (305*ll)/1018
    M[4,3] = -(305*lr)/1018

    # K 矩阵 (5x3): 重力项, 对应 [theta_ll, theta_lr, theta_b]
    K = np.zeros((5, 3))
    K[0,0] = (30429*ll)/1000 + (1078*lwl)/125
    K[1,1] = (30429*lr)/1000 + (1078*lwr)/125

    # E 矩阵 (5x4): 控制输入项, 对应 [T_lwl, T_lwr, T_bll, T_blr]
    E = np.zeros((5, 4))
    E[0,0] = - 10*ll - 1
    E[0,2] = 1
    E[1,1] = - 10*lr - 1
    E[1,3] = 1
    E[2,0] = 1
    E[2,1] = 1
    E[3,2] = -1
    E[3,3] = -1
    E[4,0] = -509/200
    E[4,1] = 509/200

    # 求解 ddq = -inv(M)*K*q - inv(M)*E*u
    coeff_q = -np.linalg.solve(M, K)   # 5x3: coeff_q[i,j] = d(ddq_i)/d(q_j)
    coeff_u = -np.linalg.solve(M, E)   # 5x4: coeff_u[i,j] = d(ddq_i)/d(u_j)

    # 构建 A (10x10)
    A = np.zeros((10, 10))
    A[0,1] = 1.0    # s_dot = s_dot
    A[2,3] = 1.0    # phi_dot = phi_dot
    A[4,5] = 1.0    # theta_ll_dot = theta_ll_dot
    A[6,7] = 1.0    # theta_lr_dot = theta_lr_dot
    A[8,9] = 1.0    # theta_b_dot = theta_b_dot

    # 行2: s_ddot (对应 x5=theta_ll, x7=theta_lr, x9=theta_b)
    A[1,4] = R_w/2 * (coeff_q[0,0] + coeff_q[1,0])   # d(s_ddot)/d(theta_ll)
    A[1,6] = R_w/2 * (coeff_q[0,1] + coeff_q[1,1])   # d(s_ddot)/d(theta_lr)
    A[1,8] = R_w/2 * (coeff_q[0,2] + coeff_q[1,2])   # d(s_ddot)/d(theta_b)

    # 行4: phi_ddot
    A[3,4] = R_w/(2*R_l)*(-coeff_q[0,0] + coeff_q[1,0]) - ll/(2*R_l)*coeff_q[2,0] + lr/(2*R_l)*coeff_q[3,0]
    A[3,6] = R_w/(2*R_l)*(-coeff_q[0,1] + coeff_q[1,1]) - ll/(2*R_l)*coeff_q[2,1] + lr/(2*R_l)*coeff_q[3,1]
    A[3,8] = R_w/(2*R_l)*(-coeff_q[0,2] + coeff_q[1,2]) - ll/(2*R_l)*coeff_q[2,2] + lr/(2*R_l)*coeff_q[3,2]

    # 行6,8,10: theta_ll_ddot, theta_lr_ddot, theta_b_ddot
    A[5,4] = coeff_q[2,0];  A[5,6] = coeff_q[2,1];  A[5,8] = coeff_q[2,2]
    A[7,4] = coeff_q[3,0];  A[7,6] = coeff_q[3,1];  A[7,8] = coeff_q[3,2]
    A[9,4] = coeff_q[4,0];  A[9,6] = coeff_q[4,1];  A[9,8] = coeff_q[4,2]

    # 构建 B (10x4)
    B = np.zeros((10, 4))
    # 行2: s_ddot
    B[1,0] = R_w/2 * (coeff_u[0,0] + coeff_u[1,0])
    B[1,1] = R_w/2 * (coeff_u[0,1] + coeff_u[1,1])
    B[1,2] = R_w/2 * (coeff_u[0,2] + coeff_u[1,2])
    B[1,3] = R_w/2 * (coeff_u[0,3] + coeff_u[1,3])

    # 行4: phi_ddot
    B[3,0] = R_w/(2*R_l)*(-coeff_u[0,0] + coeff_u[1,0]) - ll/(2*R_l)*coeff_u[2,0] + lr/(2*R_l)*coeff_u[3,0]
    B[3,1] = R_w/(2*R_l)*(-coeff_u[0,1] + coeff_u[1,1]) - ll/(2*R_l)*coeff_u[2,1] + lr/(2*R_l)*coeff_u[3,1]
    B[3,2] = R_w/(2*R_l)*(-coeff_u[0,2] + coeff_u[1,2]) - ll/(2*R_l)*coeff_u[2,2] + lr/(2*R_l)*coeff_u[3,2]
    B[3,3] = R_w/(2*R_l)*(-coeff_u[0,3] + coeff_u[1,3]) - ll/(2*R_l)*coeff_u[2,3] + lr/(2*R_l)*coeff_u[3,3]

    # 行6,8,10: theta_ll_ddot, theta_lr_ddot, theta_b_ddot
    for j in range(4):
        B[5,j] = coeff_u[2,j]
        B[7,j] = coeff_u[3,j]
        B[9,j] = coeff_u[4,j]

    return A, B
