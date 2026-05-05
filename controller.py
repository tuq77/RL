"""
PD控制器模块
"""
import numpy as np


class PDController:
    """PD控制器类"""
    
    def __init__(self, kp=2.0, kd=0.5, max_output=0.5):
        """
        初始化PD控制器
        
        Args:
            kp: 比例增益
            kd: 微分增益
            max_output: 最大输出值
        """
        self.kp = kp
        self.kd = kd
        self.max_output = max_output
        self.prev_error = 0
        self.prev_time = 0
        
    def update(self, error, current_time):
        """
        计算控制输出
        
        Args:
            error: 当前误差
            current_time: 当前时间
            
        Returns:
            控制输出值
        """
        # 计算时间差
        dt = current_time - self.prev_time if self.prev_time > 0 else 0.01
        
        # 计算微分项
        if dt > 0:
            error_rate = (error - self.prev_error) / dt
        else:
            error_rate = 0
        
        # PD控制输出
        output = self.kp * error + self.kd * error_rate
        
        # 输出限制
        output = np.clip(output, -self.max_output, self.max_output)
        
        # 更新状态
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def reset(self):
        """重置控制器状态"""
        self.prev_error = 0
        self.prev_time = 0
