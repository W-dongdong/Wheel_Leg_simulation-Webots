"""
compute_K.py  —  LQR 状态反馈矩阵 K

K = compute_K(A, B, Q, R) → 4x10
控制律: u = K (x_d - x)
"""

import numpy as np
from scipy.linalg import solve_continuous_are


def compute_K(A, B, Q=None, R=None):
    if Q is None:
        Q = np.diag([1, 150, 500, 1, 500, 1, 500, 1, 5000, 1])
    if R is None:
        R = np.diag([1, 1, 1, 1])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)

    return K

