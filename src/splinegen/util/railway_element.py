"""
铁路线元通用数学模型实现
基于论文中的统一数学模型，支持直线、圆曲线、缓和曲线的统一处理
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from scipy import integrate
import math

class RailwayElement:
    """
    铁路线元类
    
    共性：直线、圆曲线、缓和曲线的曲率均随弧长作线性变化，可用"曲线元"统一概括
    
    核心公式：
    1. 曲率：k(l) = k_s + Σ(i=1 to n) C_i * l^i
    2. 切线方位角：α(l) = α_0 + S_ccw * ∫(0 to l) k(t) dt
    3. 平面坐标：
       X(l) = x_0 + ∫(0 to l) cos(α(t)) dt
       Y(l) = y_0 + ∫(0 to l) sin(α(t)) dt
    """
    
    def __init__(self, 
                 x0: float, y0: float, alpha0: float,
                 k_s: float, k_e: float, length: float,
                 s_ccw: int = 1, curvature_coeffs: Optional[List[float]] = None):
        """
        初始化铁路线元
        
        Args:
            x0, y0: 起点坐标 (m)
            alpha0: 起点切线方位角 (弧度)
            k_s: 起点曲率 (1/m)
            k_e: 终点曲率 (1/m)
            length: 线元长度 (m)
            s_ccw: 偏向系数，左偏为-1，右偏为1
            curvature_coeffs: 曲率多项式系数 [C1, C2, ..., Cn]，如果为None则使用线性变化
        """
        self.x0 = x0
        self.y0 = y0
        self.alpha0 = alpha0
        self.k_s = k_s
        self.k_e = k_e
        self.length = length
        self.s_ccw = s_ccw
        
        # 如果没有提供曲率系数，使用线性变化 k(l) = k_s + (k_e - k_s) * l / L
        if curvature_coeffs is None:
            self.curvature_coeffs = [(k_e - k_s) / length] if length > 0 else [0.0]
        else:
            self.curvature_coeffs = curvature_coeffs
    
    def curvature(self, l: float) -> float:
        """
        计算弧长l处的曲率
        k(l) = k_s + Σ(i=1 to n) C_i * l^i
        """
        k = self.k_s
        for i, c_i in enumerate(self.curvature_coeffs, 1):
            k += c_i * (l ** i)
        return k
    
    def azimuth(self, l: float) -> float:
        """
        计算弧长l处的切线方位角
        α(l) = α_0 + S_ccw * ∫(0 to l) k(t) dt
        """
        def k_func(t):
            return self.curvature(t)
        
        # 数值积分计算方位角变化量
        integral_result, _ = integrate.quad(k_func, 0, l)
        alpha = self.alpha0 + self.s_ccw * integral_result
        return alpha
    
    def coordinates(self, l: float) -> Tuple[float, float]:
        """
        计算弧长l处的平面坐标
        X(l) = x_0 + ∫(0 to l) cos(α(t)) dt
        Y(l) = y_0 + ∫(0 to l) sin(α(t)) dt
        """
        def x_integrand(t):
            return math.cos(self.azimuth(t))
        
        def y_integrand(t):
            return math.sin(self.azimuth(t))
        
        # 数值积分计算坐标
        x_integral, _ = integrate.quad(x_integrand, 0, l)
        y_integral, _ = integrate.quad(y_integrand, 0, l)
        
        x = self.x0 + x_integral
        y = self.y0 + y_integral
        
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
        """
        创建直线段 (k_s = k_e = 0)
        """
        return cls(x0, y0, alpha0, k_s=0.0, k_e=0.0, length=length)
    
    @classmethod  
    def create_circular_curve(cls, x0: float, y0: float, alpha0: float, 
                            radius: float, length: float, direction: int = 1):
        """
        创建圆曲线段 (k_s = k_e = 1/R)
        
        Args:
            radius: 曲率半径 (m)
            direction: 方向，右偏为1，左偏为-1
        """
        curvature = 1.0 / radius
        return cls(x0, y0, alpha0, k_s=curvature, k_e=curvature, 
                  length=length, s_ccw=direction)
    
    @classmethod
    def create_transition_curve(cls, x0: float, y0: float, alpha0: float,
                              k_start: float, k_end: float, length: float, 
                              direction: int = 1):
        """
        创建缓和曲线段 (曲率线性变化)
        
        Args:
            k_start: 起点曲率 (1/m)
            k_end: 终点曲率 (1/m) 
            direction: 方向，右偏为1，左偏为-1
        """
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
    
    def plot_route(self, step: float = 0.5, save_path: Optional[str] = None):
        """绘制线路图"""
        points = self.generate_route_points(step)
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        plt.figure(figsize=(12, 8))
        plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='Railway Route')
        plt.scatter(x_coords[0], y_coords[0], s=100, c='green', marker='o', 
                   label='Start Point', zorder=5)
        plt.scatter(x_coords[-1], y_coords[-1], s=100, c='red', marker='s', 
                   label='End Point', zorder=5)
        
        plt.axis('equal')
        plt.title('Railway Route - Universal Mathematical Model')
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


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
    
    # 左偏缓和曲线
    route.add_transition_curve(k_start=0.0, k_end=1/100, length=50.0, direction=-1)
    
    # 左偏圆曲线
    route.add_circular_curve(radius=100.0, length=120.0, direction=-1)
    
    # 左偏缓和曲线
    route.add_transition_curve(k_start=1/100, k_end=0.0, length=50.0, direction=-1)
    
    # 直线段
    route.add_straight_line(100.0)
    
    return route


if __name__ == "__main__":
    # 创建示例线路
    route = create_sample_railway()
    
    # 生成坐标点
    points = route.generate_route_points(step=1.0)
    print(f"Generated {len(points)} points")
    print(f"Total route length: {route.total_length:.1f} m")
    
    # 绘制线路图
    route.plot_route(step=1.0, save_path="railway_universal_model.png")
    
    # 保存坐标数据
    import csv
    with open('railway_universal_model_points.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(points)
    
    print("Railway route generated successfully!")
