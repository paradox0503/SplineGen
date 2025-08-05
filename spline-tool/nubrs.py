import matplotlib.pyplot as plt
import numpy as np
import os


def nurbs_basis(i, p, u, knots):
    """计算NURBS基函数N_{i,p}(u)"""
    if p == 0:
        return 1.0 if knots[i] <= u < knots[i+1] else 0.0
    
    # 避免除零
    left_denom = knots[i+p] - knots[i]
    right_denom = knots[i+p+1] - knots[i+1]
    
    left_term = 0.0
    if left_denom != 0:
        left_term = (u - knots[i]) / left_denom * nurbs_basis(i, p-1, u, knots)
    
    right_term = 0.0
    if right_denom != 0:
        right_term = (knots[i+p+1] - u) / right_denom * nurbs_basis(i+1, p-1, u, knots)
    
    return left_term + right_term


def nurbs_curve(u_vals, ctrl_pts, knots, weights, p=3):
    """计算NURBS曲线点"""
    n = len(ctrl_pts)
    curve_points = []
    
    for u in u_vals:
        # 计算所有基函数
        basis_vals = []
        for i in range(n):
            if i + p + 1 < len(knots):
                basis_vals.append(nurbs_basis(i, p, u, knots))
            else:
                basis_vals.append(0.0)
        
        # 计算有理基函数
        weighted_sum = sum(basis_vals[i] * weights[i] for i in range(n))
        if weighted_sum == 0:
            curve_points.append([0, 0])
            continue
            
        rational_basis = [basis_vals[i] * weights[i] / weighted_sum for i in range(n)]
        
        # 计算曲线点
        point = np.sum([rational_basis[i] * ctrl_pts[i] for i in range(n)], axis=0)
        curve_points.append(point)
    
    return np.array(curve_points)


def get_focus_limits(points, padding=0.1):
    """计算原始点区域的聚焦范围"""
    if len(points) == 0:
        return -1, 1, -1, 1
    
    x = points[:, 0]
    y = points[:, 1]
    
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    
    x_pad = x_range * padding
    y_pad = y_range * padding
    
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def get_global_limits(curve, ctrl_pts, points, padding=0.1):
    """计算包含所有元素的全局范围"""
    all_x = np.concatenate([curve[:, 0], ctrl_pts[:, 0], points[:, 0]])
    all_y = np.concatenate([curve[:, 1], ctrl_pts[:, 1], points[:, 1]])
    
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    
    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0
    
    x_pad = x_range * padding
    y_pad = y_range * padding
    
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


if __name__ == "__main__":
    npz_file = "spline_data_1754277230.npz"
    if not os.path.exists(npz_file):
        print(f"错误: 文件 {npz_file} 不存在")
        exit(1)
    
    try:
        data = np.load(npz_file)
        required_keys = ['ctrl_pts', 'knots', 'points', 'params']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"错误: NPZ文件中缺少以下键: {missing_keys}")
            print(f"文件中包含的键: {list(data.keys())}")
            exit(1)
            
        # 处理控制点
        ctrl_pts = data['ctrl_pts']
        non_zero_mask = ~np.all(ctrl_pts == 0, axis=1)
        ctrl_pts = ctrl_pts[non_zero_mask]
        len_ctrl_pts = len(ctrl_pts)
        
        # 处理节点
        p = 3  # NURBS阶数
        all_knots = data['knots']
        max_length = len_ctrl_pts + p + 1
        
        if len(all_knots) > max_length:
            print(f"原始节点数量 {len(all_knots)} 超过推荐值 {max_length}")
            if len(all_knots) >= len_ctrl_pts + 2 * (p + 1):
                mid_knots_needed = len_ctrl_pts - (p + 1)
                if mid_knots_needed > 0:
                    start_knots = all_knots[:p+1]
                    end_knots = all_knots[-(p+1):]
                    mid_start = p + 1
                    mid_end = mid_start + mid_knots_needed
                    mid_knots = all_knots[mid_start:mid_end]
                    knots = np.concatenate([start_knots, mid_knots, end_knots])
                else:
                    knots = all_knots[:max_length]
            else:
                knots = all_knots[:max_length]
            print(f"节点数据已调整至 {len(knots)} 个元素")
        else:
            knots = all_knots
        
        # 处理权重参数
        params = data['params']
        if len(params) > len_ctrl_pts:
            weights = params[:len_ctrl_pts]
            print(f"权重参数已截取至 {len_ctrl_pts} 个元素")
        else:
            weights = params
            if len(weights) < len_ctrl_pts:
                # 如果权重不足，用1.0补足
                weights = np.concatenate([weights, np.ones(len_ctrl_pts - len(weights))])
                print(f"权重参数不足，已用1.0补足至 {len_ctrl_pts} 个元素")
        
        # 处理原始点
        points = data['points']
        non_zero_points_mask = ~np.all(points == 0, axis=1)
        points = points[non_zero_points_mask]
        
        # 打印基本信息
        print(f"控制点数量: {len(ctrl_pts)}")
        print(f"节点数量: {len(knots)}")
        print(f"权重数量: {len(weights)}")
        print(f"原始数据点数量: {len(points)}")
        print(f"权重范围: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
        
        # 验证NURBS参数
        n = len(ctrl_pts)
        m = len(knots)
        if m != n + p + 1:
            print(f"警告: NURBS参数不匹配。控制点数={n}, 节点数={m}, 阶数={p}")
            print(f"标准关系应为: 节点数 = 控制点数 + 阶数 + 1 = {n + p + 1}")
        
        if not np.all(np.diff(knots) >= 0):
            print("错误: 节点向量不是单调非递减的")
            exit(1)
            
        # 检查边界节点重复度
        unique_start = np.sum(knots == knots[0])
        unique_end = np.sum(knots == knots[-1])
        print(f"起始节点重复度: {unique_start}, 结束节点重复度: {unique_end}")
        if unique_start < p + 1 or unique_end < p + 1:
            print(f"警告: 边界节点重复度可能不足。对于{p}次NURBS，建议重复度至少为{p+1}")
            
        # 检查有效参数域
        if knots[p] >= knots[-(p+1)]:
            print("错误: 有效参数域为空")
            exit(1)

        # 构造NURBS曲线
        u_min = knots[p]
        u_max = knots[-(p+1)]
        print(f"NURBS有效参数域: [{u_min:.6f}, {u_max:.6f}]")
        
        u_vals = np.linspace(u_min, u_max, 300)
        curve = nurbs_curve(u_vals, ctrl_pts, knots, weights, p)

        # 计算视图范围
        focus_x_min, focus_x_max, focus_y_min, focus_y_max = get_focus_limits(points)
        global_x_min, global_x_max, global_y_min, global_y_max = get_global_limits(curve, ctrl_pts, points)
        
        print(f"全局视图范围: X [{global_x_min:.4f}, {global_x_max:.4f}], Y [{global_y_min:.4f}, {global_y_max:.4f}]")
        print(f"聚焦视图范围: X [{focus_x_min:.4f}, {focus_x_max:.4f}], Y [{focus_y_min:.4f}, {focus_y_max:.4f}]")

        # 创建三视图：全局、聚焦、权重分析
        fig = plt.figure(figsize=(20, 7))
        
        # 设置中文显示
        plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 全局视图
        ax1 = plt.subplot(131)
        ax1.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2, label='NURBS曲线')
        ax1.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'ro--', label='控制点', markersize=6)
        ax1.scatter(points[:, 0], points[:, 1], color='green', s=20, label='原始点', alpha=0.7)
        ax1.set_title('NURBS曲线 - 全局视图')
        ax1.set_xlim(global_x_min, global_x_max)
        ax1.set_ylim(global_y_min, global_y_max)
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 聚焦视图
        ax2 = plt.subplot(132)
        ax2.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2, label='NURBS曲线')
        ax2.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'ro--', label='控制点', markersize=6)
        ax2.scatter(points[:, 0], points[:, 1], color='green', s=20, label='原始点', alpha=0.7)
        ax2.set_title('原始点区域聚焦视图')
        ax2.set_xlim(focus_x_min, focus_x_max)
        ax2.set_ylim(focus_y_min, focus_y_max)
        ax2.set_aspect('equal', adjustable='box')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 权重分析视图
        ax3 = plt.subplot(133)
        # 显示控制点及其权重
        scatter = ax3.scatter(ctrl_pts[:, 0], ctrl_pts[:, 1], c=weights, s=100, 
                             cmap='viridis', alpha=0.8, edgecolors='black')
        ax3.plot(ctrl_pts[:, 0], ctrl_pts[:, 1], 'k--', alpha=0.5, label='控制多边形')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('权重值', rotation=270, labelpad=15)
        
        # 标注权重值
        for i, (pt, w) in enumerate(zip(ctrl_pts, weights)):
            ax3.annotate(f'{w:.2f}', (pt[0], pt[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8, alpha=0.8)
        
        ax3.set_title('控制点权重分布')
        ax3.set_aspect('equal', adjustable='box')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig('nurbs_curve_analysis.png', dpi=300, bbox_inches='tight')
        print("图片已保存为 nurbs_curve_analysis.png")
        
        # 额外保存权重信息
        print("\n权重信息:")
        for i, w in enumerate(weights):
            print(f"  控制点 {i}: 权重 = {w:.6f}")
        
        # 检查是否为标准B样条（所有权重相等）
        if np.allclose(weights, weights[0]):
            print(f"\n注意: 所有权重都接近 {weights[0]:.6f}，这实际上是一个标准B样条曲线")
        else:
            print(f"\n这是一个真正的NURBS曲线，权重变化范围: {np.max(weights) - np.min(weights):.6f}")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        exit(1)