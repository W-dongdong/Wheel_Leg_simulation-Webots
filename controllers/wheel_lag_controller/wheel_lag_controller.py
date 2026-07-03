"""
main.py — wheel_lag_controller.py 修 bug 版
==========================================
保留原有 PID 腿长控制逻辑，仅修 bug。
"""

import numpy as np
from controller import Robot
from imu import IMU
from pid import PID
from leg import *
from compute_AB import *
from compute_K import compute_K

# ============================================================
robot = Robot()
timestep = int(robot.getBasicTimeStep())

motor1 = robot.getDevice("motor1")
motor2 = robot.getDevice("motor2")
motor3 = robot.getDevice("motor3")
motor4 = robot.getDevice("motor4")
motor5 = robot.getDevice("motor5")
motor6 = robot.getDevice("motor6")

imu = IMU(robot, timestep)

# P1: 纯力矩模式
for m in [motor1, motor2, motor3, motor4]:
    m.setAvailableTorque(15)
motor5.setAvailableTorque(5)
motor6.setAvailableTorque(5)

motors = [motor1, motor2, motor3, motor4, motor5, motor6]
position_sensors = [m.getPositionSensor() for m in motors]
for ps in position_sensors:
    ps.enable(timestep)

gps = robot.getDevice('gps')
if gps:
    gps.enable(timestep)

gyro = robot.getDevice('gyro')
if gyro:
    gyro.enable(timestep)

# 腿部运动学
R_leg = FiveBarLeg(L1a=0.06, L1u=0.2, L1d=0.3, L2a=0.06, L2u=0.2, L2d=0.3)
L_leg = FiveBarLeg(L1a=0.06, L1u=0.2, L1d=0.3, L2a=0.06, L2u=0.2, L2d=0.3)

R_LengthControl = PID(2000, 1, 6000, 100, 200)
R_ThetaControl  = PID(1, 0, 5000, 0, 500)
L_LengthControl = PID(2000, 1, 6000, 100, 200)
L_ThetaControl  = PID(1, 0, 5000, 0, 500)

PitchControl = PID(1.1, 0.1, 1, 5, 100)
RollControl  = PID(0, 0.1, 1, 0.05, 0.1)


R_F, R_T = 0.0, 0.0
L_F, L_T = 0.0, 0.0

L_LegLength = R_LegLength = 0.2
target_speed = 0.0

# ---- 手柄 / 键盘 ----
MAX_SPEED     = 2.5    # 最大前进速度 [m/s]
MAX_YAW_RATE  = 3    # 最大偏航角速度 [rad/s]

keyboard = robot.getKeyboard()
keyboard.enable(timestep)


def read_input():
    """读取键盘, 支持同时按住多个键 [W/S前后, A/D转弯]"""
    spd, yaw = 0.0, 0.0

    key = keyboard.getKey()
    while key != -1:
        if key == ord('W'):   spd = -1.0
        if key == ord('S'):   spd = 1.0
        if key == ord('A'):   yaw = 1.0
        if key == ord('D'):   yaw = -1.0
        key = keyboard.getKey()

    return spd, yaw


# 编码器差分 (腿关节 + 轮子)
p1_prev = p2_prev = p3_prev = p4_prev = None
p5_prev = p6_prev = None

R_W = 0.1  # 轮半径 [m]
DEG2RAD = np.pi / 180.0

def clamp(v, lim):
    return max(-lim, min(lim, v))

# ============================================================
while robot.step(timestep) != -1:

    # ---- 传感器 ----
    pitch = imu.pitch * DEG2RAD     # → [rad]
    yaw   = imu.yaw   * DEG2RAD
    roll  = imu.roll  * DEG2RAD

    if gyro:
        gv = gyro.getValues()
        dot_pitch, dot_roll, dot_yaw = gv[0], gv[1], gv[2]
    else:
        dot_pitch, dot_roll, dot_yaw = 0.0, 0.0, 0.0

    # ---- 编码器 ----
    Pos1 = -(motor1.getPositionSensor().getValue() - 0.3) + 3.1415926
    Pos2 = -(motor2.getPositionSensor().getValue() - 0.3) + 3.1415926
    Pos3 =  (motor3.getPositionSensor().getValue() + 0.3) + 3.1415926
    Pos4 =  (motor4.getPositionSensor().getValue() + 0.3) + 3.1415926
    Pos5 =  -motor5.getPositionSensor().getValue()
    Pos6 =  -motor6.getPositionSensor().getValue()

    s = -R_W/2 * (Pos5 + Pos6)

    if p1_prev is None:
        p1_prev, p2_prev = Pos1, Pos2
        p3_prev, p4_prev = Pos3, Pos4
        p5_prev, p6_prev = Pos5, Pos6
        dphi1 = dphi2 = dphi3 = dphi4 = 0.0
        dot_s_enc = 0.0
    else:
        dt = timestep / 1000.0
        dphi1 = (Pos1 - p1_prev) / dt
        dphi2 = (Pos2 - p2_prev) / dt
        dphi3 = (Pos3 - p3_prev) / dt
        dphi4 = (Pos4 - p4_prev) / dt
        p1_prev, p2_prev = Pos1, Pos2
        p3_prev, p4_prev = Pos3, Pos4

        # 轮子编码器速度 (用于判断方向)
        dot_theta_wl = (Pos5 - p5_prev) / dt
        dot_theta_wr = (Pos6 - p6_prev) / dt
        p5_prev, p6_prev = Pos5, Pos6
        dot_s_enc = R_W/2 * (dot_theta_wl + dot_theta_wr)

    # GPS 速率大小 + 编码器方向 = 前进速度
    if gps:
        gps_vel = gps.getSpeedVector()
        gps_speed = np.sqrt(gps_vel[0]**2 + gps_vel[1]**2)  # 世界坐标速率模长
    else:
        gps_speed = abs(dot_s_enc)

    dot_s = -np.sign(dot_s_enc) * gps_speed if abs(dot_s_enc) > 0.01 else 0.0

    # ---- 运动学 (P2 + P3: 先 kinematics, 再 PID, 再 torque) ----
    R_leg.update_kinematics(Pos1, Pos3)
    L_leg.update_kinematics(Pos2, Pos4)

    R_leg.compute_velocity(dphi1, dphi3)   # P2
    L_leg.compute_velocity(dphi2, dphi4)

    # P3: PID update 在 torque 之前
    R_F = R_LengthControl.update(R_LegLength, R_leg.l * np.cos(R_leg.theta))
    L_F = L_LengthControl.update(L_LegLength, L_leg.l * np.cos(L_leg.theta))

    # ---- A, B, K (原逻辑不变) ----
    ll  = L_leg.l
    lr  = R_leg.l
    lwl = L_leg.l * 0.4
    lwr = R_leg.l * 0.4
    lbl = L_leg.l * 0.6
    lbr = R_leg.l * 0.6        # P8

    I_ll = 0.1047 * ll + 0.0134
    I_lr = 0.1047 * lr + 0.0134

    A, B = compute_AB(ll, lr, lwl, lwr, lbl, lbr, I_ll, I_lr)
    K = compute_K(A, B, Q=None, R=None)

    # ---- 手柄/键盘输入 ----
    cmd_spd, cmd_yaw = read_input()

    # ---- LQR 状态向量 (原逻辑, pitch 用弧度) ----
    x = np.array([
        s,
        dot_s,
        yaw,
        dot_yaw,
        L_leg.theta,
        L_leg.dot_theta,    # P2: 现在非零
        R_leg.theta,
        R_leg.dot_theta,    # P2: 现在非零
        pitch,
        dot_pitch
    ])

    x_d = np.array([
        s,                    # [0] 不控位置
        cmd_spd * MAX_SPEED,  # [1] 目标前进速度
        yaw,                  # [2] 不纠偏航角，保持当前朝向
        cmd_yaw * MAX_YAW_RATE,  # [3] 目标偏航角速度
        0, 0,                 # [4][5] 左腿归零
        0, 0,                 # [6][7] 右腿归零
        0, 0                  # [8][9] 机体俯仰归零
    ])

    # ---- LQR ----
    u = K @ (x_d - x)

    motor5.setForce(clamp(u[0], 5.0))
    motor6.setForce(clamp(u[1], 5.0))
    L_T = u[2]
    R_T = u[3]

    # ---- 力矩输出 (原逻辑, 但 PID+torque 都是当前帧) ----
    t1, t3 = R_leg.compute_motor_torques(-R_F, -R_T)   # P3: 本帧 R_F/R_T
    t2, t4 = L_leg.compute_motor_torques(-L_F, -L_T)

    motor1.setForce(clamp(t1, 15.0))
    motor3.setForce(clamp(-t3, 15.0))
    motor2.setForce(clamp(t2, 15.0))
    motor4.setForce(clamp(-t4, 15.0))
    
    #-------Roll角控制--------#
    # L_LegLength += RollControl.update(0, roll)
    
        

