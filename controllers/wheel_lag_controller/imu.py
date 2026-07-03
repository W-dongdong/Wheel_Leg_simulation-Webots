"""IMU 传感器封装模块."""

import mathtool as math


class IMU:
    """IMU 传感器封装：姿态角(度)、角速度(rad/s)、线加速度(m/s²)."""

    def __init__(self, robot, timestep):
        self._imu = robot.getDevice("imu")
        self._imu.enable(timestep)
        self._gyro = robot.getDevice("gyro")
        self._gyro.enable(timestep)
        self._acc = robot.getDevice("accelerometer")
        self._acc.enable(timestep)

    @property
    def pitch(self):
        """横滚角，单位：度."""
        return math.rad2dgr(self._imu.getRollPitchYaw()[0])

    @property
    def roll(self):
        """俯仰角，单位：度."""
        return math.rad2dgr(self._imu.getRollPitchYaw()[1])

    @property
    def yaw(self):
        """偏航角，单位：度."""
        return math.rad2dgr(self._imu.getRollPitchYaw()[2])

    @property
    def rpy(self):
        """姿态角 [roll, pitch, yaw]，单位：度."""
        return [self.roll, self.pitch, self.yaw]

    @property
    def angular_vel(self):
        """角速度 [gx, gy, gz]，单位：rad/s."""
        return self._gyro.getValues()

    @property
    def linear_acc(self):
        """线加速度 [ax, ay, az]，单位：m/s²."""
        return self._acc.getValues()
