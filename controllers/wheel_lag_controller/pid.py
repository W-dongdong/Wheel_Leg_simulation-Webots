class PID:
    def __init__(self, kp, ki, kd, i_limit, out_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_limit = i_limit
        self.out_limit = out_limit
        self.integral = 0.0
        self.last_error = 0.0

    def update(self, target, measure):
        error = target - measure
        p_out = self.kp * error
        
        self.integral += self.ki * error
        self.integral = max(min(self.integral, self.i_limit), -self.i_limit) # 简化的限幅写法
        
        d_out = self.kd * (error - self.last_error)
        self.last_error = error
        
        output = p_out + self.integral + d_out
        return max(min(output, self.out_limit), -self.out_limit)

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0


class CascadePID:
    def __init__(self, 
                 kp_out, ki_out, kd_out, i_limit_out, out_limit_out,
                 kp_in, ki_in, kd_in, i_limit_in, out_limit_in,
                 outer_divider=5):
        """
        初始化双速率串级 PID
        :param outer_divider: 外环分频系数。默认 5，代表内环计算 5 次，外环才计算 1 次
        """
        # 实例化内、外环 PID
        self.outer = LightweightPID(kp_out, ki_out, kd_out, i_limit_out, out_limit_out)
        self.inner = LightweightPID(kp_in, ki_in, kd_in, i_limit_in, out_limit_in)
        
        # 分频控制参数
        self.outer_divider = outer_divider
        self.counter = 0            # 循环计数器
        self.last_target_inner = 0.0 # 锁存外环输出，供内环未更新时使用

    def update(self, target_outer, measure_outer, measure_inner):
        """
        更新串级 PID 计算（内环每步都算，外环分频计算）
        :param target_outer: 外环目标值（如：目标角度）
        :param measure_outer: 外环测量值（如：当前角度）
        :param measure_inner: 内环测量值（如：当前速度）
        """
        # 1. 判断是否到达外环计算周期
        if self.counter == 0:
            # 满足分频条件，更新外环，计算出新的“目标速度”
            self.last_target_inner = self.outer.update(target_outer, measure_outer)
        
        # 计数器累加与清零
        self.counter += 1
        if self.counter >= self.outer_divider:
            self.counter = 0

        # 2. 内环每个周期都必须计算
        # 如果外环没更新，内环将继续沿用上一次锁存的 self.last_target_inner
        final_output = self.inner.update(self.last_target_inner, measure_inner)
        
        return final_output

    def reset(self):
        """同时复位内外环数据和计数器"""
        self.outer.reset()
        self.inner.reset()
        self.counter = 0
        self.last_target_inner = 0.0