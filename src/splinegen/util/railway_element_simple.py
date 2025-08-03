"""
简化的铁路线元通用数学模型实现
基于论文中的统一数学模型，使用近似计算方法提高效率
"""

import numpy as np
import math
from typing import List, Tuple, Optional


class RailwayElement:
    """
    铁路线元类 - 简化版
    
    使用离散化方法代替数值积分，提高计算效率
    """
    
    def __init__(self, 
                 x0: float, y0: float, alpha0: float,
                 k_s: float, k_e: float, length: float,
                 s_ccw: int = 1):
        """
        初始化铁路线元
        
        Args:
            x0, y0: 起点坐标 (m)
            alpha0: 起点切线方位角 (弧度)
            k_s: 起点曲率 (1/m)
            k_e: 终点曲率 (1/m)
            length: 线元长度 (m)
            s_ccw: 偏向系数，左偏为-1，右偏为1
        """
        self.x0 = x0
        self.y0 = y0
        self.alpha0 = alpha0
        self.k_s = k_s
        self.k_e = k_e
        self.length = length
        self.s_ccw = s_ccw
        
        # 曲率线性变化系数
        self.k_rate = (k_e - k_s) / length if length > 0 else 0.0
    
    def curvature(self, l: float) -> float:
        """
        计算弧长l处的曲率 (线性变化)
        k(l) = k_s + (k_e - k_s) * l / L
        """
        return self.k_s + self.k_rate * l
    
    def azimuth(self, l: float) -> float:
        """
        计算弧长l处的切线方位角
        使用简化积分：α(l) = α_0 + S_ccw * (k_s * l + 0.5 * k_rate * l^2)
        """
        integral = self.k_s * l + 0.5 * self.k_rate * (l ** 2)
        alpha = self.alpha0 + self.s_ccw * integral
        return alpha
    
    def coordinates(self, l: float) -> Tuple[float, float]:
        """
        计算弧长l处的平面坐标
        使用小步长离散化计算
        """
        if l == 0:
            return self.x0, self.y0
        
        # 使用固定步长进行离散化计算
        step = min(1.0, l / 10.0)  # 步长不超过1m，且至少分10步
        steps = int(l / step)
        actual_step = l / steps if steps > 0 else l
        
        x, y = self.x0, self.y0
        current_l = 0.0
        
        for i in range(steps):
            current_alpha = self.azimuth(current_l + actual_step / 2)  # 中点法
            x += actual_step * math.cos(current_alpha)
            y += actual_step * math.sin(current_alpha)
            current_l += actual_step
        
        return x, y
    
    def generate_points(self, step: float = 0.5) -> List[Tuple[float, float]]:
        """
        生成线元上的坐标点
        
        Args:
            step: 采样步长 (m)
            
        Returns:
            坐标点列表 [(x1, y1), (x2, y2), ...]
        """
        points = []
        l_values = np.arange(0, self.length + step, step)
        
        for l in l_values:
            x, y = self.coordinates(l)
            points.append((x, y))
        
        return points
    
    def get_end_state(self) -> Tuple[float, float, float]:
        """
        获取线元终点状态
        
        Returns:
            (终点x坐标, 终点y坐标, 终点切线方位角)
        """
        x_end, y_end = self.coordinates(self.length)
        alpha_end = self.azimuth(self.length)
        return x_end, y_end, alpha_end
    
    @classmethod
    def create_straight_line(cls, x0: float, y0: float, alpha0: float, length: float):
        """创建直线段 (k_s = k_e = 0)"""
        return cls(x0, y0, alpha0, k_s=0.0, k_e=0.0, length=length)
    
    @classmethod  
    def create_circular_curve(cls, x0: float, y0: float, alpha0: float, 
                            radius: float, length: float, direction: int = 1):
        """创建圆曲线段 (k_s = k_e = 1/R)"""
        curvature = 1.0 / radius
        return cls(x0, y0, alpha0, k_s=curvature, k_e=curvature, 
                  length=length, s_ccw=direction)
    
    @classmethod
    def create_transition_curve(cls, x0: float, y0: float, alpha0: float,
                              k_start: float, k_end: float, length: float, 
                              direction: int = 1):
        """创建缓和曲线段 (曲率线性变化)"""
        return cls(x0, y0, alpha0, k_s=k_start, k_e=k_end, 
                  length=length, s_ccw=direction)


class RailwayRoute:
    """
    铁路线路类 - 由多个线元组成的复合线路
    """
    
    def __init__(self):
        self.elements: List[RailwayElement] = []
        self.total_length = 0.0
    
    def add_element(self, element: RailwayElement):
        """添加线元到线路"""
        self.elements.append(element)
        self.total_length += element.length
    
    def add_straight_line(self, length: float):
        """添加直线段"""
        if not self.elements:
            # 第一个线元，使用默认起点
            element = RailwayElement.create_straight_line(0, 0, 0, length)
        else:
            # 从上一个线元的终点开始
            last_element = self.elements[-1]
            x_end, y_end, alpha_end = last_element.get_end_state()
            element = RailwayElement.create_straight_line(x_end, y_end, alpha_end, length)
        
        self.add_element(element)
    
    def add_circular_curve(self, radius: float, length: float, direction: int = 1):
        """添加圆曲线段"""
        if not self.elements:
            element = RailwayElement.create_circular_curve(0, 0, 0, radius, length, direction)
        else:
            last_element = self.elements[-1]
            x_end, y_end, alpha_end = last_element.get_end_state()
            element = RailwayElement.create_circular_curve(x_end, y_end, alpha_end, 
                                                         radius, length, direction)
        
        self.add_element(element)
    
    def add_transition_curve(self, k_start: float, k_end: float, length: float, 
                           direction: int = 1):
        """添加缓和曲线段"""
        if not self.elements:
            element = RailwayElement.create_transition_curve(0, 0, 0, k_start, k_end, 
                                                           length, direction)
        else:
            last_element = self.elements[-1]
            x_end, y_end, alpha_end = last_element.get_end_state()
            element = RailwayElement.create_transition_curve(x_end, y_end, alpha_end,
                                                           k_start, k_end, length, direction)
        
        self.add_element(element)
    
    def generate_route_points(self, step: float = 0.5) -> List[Tuple[float, float]]:
        """生成整条线路的坐标点"""
        all_points = []
        
        for i, element in enumerate(self.elements):
            points = element.generate_points(step)
            
            # 除了第一个线元，其他线元跳过第一个点避免重复
            if i > 0:
                points = points[1:]
            
            all_points.extend(points)
        
        return all_points


def create_sample_railway() -> RailwayRoute:
    """创建示例铁路线路"""
    route = RailwayRoute()
    
    # 直线段
    route.add_straight_line(100.0)
    
    # 右偏缓和曲线 (直线到圆曲线)
    route.add_transition_curve(k_start=0.0, k_end=1/120, length=60.0, direction=1)
    
    # 右偏圆曲线
    route.add_circular_curve(radius=120.0, length=150.0, direction=1)
    
    # 右偏缓和曲线 (圆曲线到直线)
    route.add_transition_curve(k_start=1/120, k_end=0.0, length=60.0, direction=1)
    
    # 直线段
    route.add_straight_line(80.0)
    
    return route


if __name__ == "__main__":
    # 测试铁路线元模型
    print("Testing Railway Element Universal Mathematical Model")
    
    # 创建示例线路
    route = create_sample_railway()
    
    # 生成坐标点
    points = route.generate_route_points(step=1.0)
    print(f"Generated {len(points)} points")
    print(f"Total route length: {route.total_length:.1f} m")
    
    # 测试单个线元
    element = RailwayElement.create_transition_curve(0, 0, 0, 0, 1/100, 50, 1)
    test_points = element.generate_points(1.0)
    print(f"Transition curve generated {len(test_points)} points")
    
    print("Railway Element Model test completed!")
