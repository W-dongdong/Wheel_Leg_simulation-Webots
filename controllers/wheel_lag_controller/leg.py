import numpy as np

class FiveBarLeg:
    def __init__(self, L1a, L1u, L1d, L2a, L2u, L2d):
        self.L1a = L1a
        self.L1u = L1u
        self.L1d = L1d
        self.L2a = L2a
        self.L2u = L2u
        self.L2d = L2d

        self.phi1 = 0.0
        self.phi2 = 0.0
        self.x1 = 0.0; self.y1 = 0.0
        self.x2 = 0.0; self.y2 = 0.0
        self.xe = 0.0
        self.ye = 0.0
        self.l = 0.0
        self.theta = 0.0
        self.dot_l = 0.0       # 虚拟腿长变化速度
        self.dot_theta = 0.0   # 虚拟杆角速度
        self.J_matrix = np.zeros((2,2))
        self.F_act = 0.0
        self.T_act = 0.0

    def update_kinematics(self, phi1, phi2):
        self.phi1 = phi1
        self.phi2 = phi2

        x1 = self.L1a - self.L1u * np.cos(phi1)
        y1 = self.L1u * np.sin(phi1)
        x2 = self.L2a - self.L2u * np.cos(phi2)
        y2 = self.L2u * np.sin(phi2)
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

        dx = x2 + x1
        dy = y2 - y1
        d2 = dx*dx + dy*dy
        d  = np.sqrt(d2)

        d_min = abs(self.L1d - self.L2d)
        d_max = self.L1d + self.L2d
        d_clamped = np.clip(d, d_min + 1e-12, d_max - 1e-12)
        d2_clamped = d_clamped * d_clamped

        ux = dx / d_clamped
        uy = dy / d_clamped

        a = (d2_clamped + self.L1d**2 - self.L2d**2) / (2.0 * d_clamped)
        h2 = self.L1d**2 - a*a
        h = np.sqrt(np.maximum(0.0, h2))

        xe = -x1 + a * ux - h * uy
        ye =  y1 + a * uy + h * ux
        self.xe, self.ye = xe, ye

        self.l = np.sqrt(xe**2 + ye**2)
        self.theta = np.arctan2(xe, ye)

        # 雅可比 M 矩阵:  [dphi1; dphi2] = M @ [dl; dtheta]
        den1 = self.L1u * (-(xe + x1) * np.sin(phi1) + (ye - y1) * np.cos(phi1))
        dphi1_dl  = ((xe + x1) * np.sin(self.theta) + (ye - y1) * np.cos(self.theta)) / den1
        dphi1_dth = (self.l * ((xe + x1) * np.cos(self.theta) - (ye - y1) * np.sin(self.theta))) / den1

        den2 = self.L2u * ((xe - x2) * np.sin(phi2) + (ye - y2) * np.cos(phi2))
        dphi2_dl  = ((xe - x2) * np.sin(self.theta) + (ye - y2) * np.cos(self.theta)) / den2
        dphi2_dth = (self.l * ((xe - x2) * np.cos(self.theta) - (ye - y2) * np.sin(self.theta))) / den2

        M_matrix = np.array([[dphi1_dl, dphi1_dth],
                             [dphi2_dl, dphi2_dth]])
        det_M = np.linalg.det(M_matrix)

        if abs(det_M) < 1e-6:
            self.J_matrix = np.zeros((2,2))
        else:
            self.J_matrix = np.linalg.inv(M_matrix)

    def compute_velocity(self, dphi1, dphi2):
        """
        由关节角速度计算虚拟杆速度。
        [dot_l; dot_theta] = J @ [dphi1; dphi2]
        """
        vel = self.J_matrix @ np.array([dphi1, dphi2])
        self.dot_l = vel[0]
        self.dot_theta = vel[1]
        return self.dot_l, self.dot_theta

    def compute_motor_torques(self, F_des, T_des):
        self.F_act = F_des
        self.T_act = T_des

        if np.allclose(self.J_matrix, 0):
            return 0.0, 0.0

        F_virt = np.array([F_des, T_des])
        tau_motor = self.J_matrix.T @ F_virt
        return tau_motor[0], tau_motor[1]

    def get_kinematics_torque(self, phi1, phi2, F_des, T_des):
        self.update_kinematics(phi1, phi2)
        return self.compute_motor_torques(F_des, T_des)
