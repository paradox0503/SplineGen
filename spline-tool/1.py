import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
import os


def convert_ctrl_pts(npz_file_path):
    data = np.load(npz_file_path)
    ctrl_pts = data['ctrl_pts']
    non_zero_mask = ~np.all(ctrl_pts == 0, axis=1)
    filtered_pts = ctrl_pts[non_zero_mask]
    
    result = "ctrl_pts = np.array([\n"
    for i, pt in enumerate(filtered_pts):
        x, y = pt
        line_end = ",\n" if i != len(filtered_pts) - 1 else "\n"
        result += f"    [{x:>8.4f}, {y:>8.4f}]{line_end}"
    result += "])"
    
    return result


def convert_knots(npz_file_path, max_length=None):
    data = np.load(npz_file_path)
    knots = data['knots']

    if 'ctrl_pts' in data:
        ctrl_pts = data['ctrl_pts']
        non_zero_ctrl = ctrl_pts[~np.all(ctrl_pts == 0, axis=1)]
        n = len(non_zero_ctrl)
        p = 3
        recommended_length = n + p + 1
        
        if max_length is None:
            max_length = recommended_length
            print(f"根据控制点数量自动确定节点合理长度: {max_length}")
    
    if max_length and len(knots) > max_length:
        knots = knots[:max_length]
        print(f"节点数据已截取至{max_length}个元素")
    
    result = "knot_u = np.array([\n"
    elements_per_line = 5
    total = len(knots)
    
    for i, val in enumerate(knots):
        line_end = ",\n" if (i + 1) % elements_per_line == 0 and i != total - 1 else ", "
        if i == total - 1:
            line_end = "\n"
        result += f"    {val:>8.4f}{line_end}"
    
    result += "])"
    return result


def convert_points(npz_file_path, keep_zero_rows=False):
    data = np.load(npz_file_path)
    points = data['points']
    
    if not keep_zero_rows:
        non_zero_mask = ~np.all(points == 0, axis=1)
        points = points[non_zero_mask]
        print(f"已过滤全零数据点，保留 {len(points)} 个有效点")
    
    result = "points = np.array([\n"
    for i, pt in enumerate(points):
        x, y = pt
        line_end = ",\n" if i != len(points) - 1 else "\n"
        result += f"    [{x:>8.4f}, {y:>8.4f}]{line_end}"
    
    result += "])"
    return result


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
    npz_file = "/mnt/data2/user_jialinhan/docker/SplineGen/spline-tool/spline_data_1754277229.npz"
    if not os.path.exists(npz_file):
        print(f"错误: 文件 {npz_file} 不存在")
        exit(1)
    
    try:
        data = np.load(npz_file)
        required_keys = ['ctrl_pts', 'knots', 'points']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"错误: NPZ文件中缺少以下键: {missing_keys}")
            exit(1)
            
        # 处理控制点
        ctrl_pts = data['ctrl_pts']
        non_zero_mask = ~np.all(ctrl_pts == 0, axis=1)
        ctrl_pts = ctrl_pts[non_zero_mask]
        len_ctrl_pts = len(ctrl_pts)
        max_length = len_ctrl_pts + 4
        import pdb; pdb.set_trace()
        
        # 处理节点
        p = 3  # B样条阶数
        all_knots = data['knots']
        
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
        # import pdb; pdb.set_trace()
        
        # 处理原始点
        points = data['points']
        non_zero_points_mask = ~np.all(points == 0, axis=1)
        points = points[non_zero_points_mask]
        # import pdb; pdb.set_trace()
        
        # 打印基本信息
        print(f"控制点数量: {len(ctrl_pts)}")
        print(f"节点数量: {len(knots)}")
        print(f"原始数据点数量: {len(points)}")
        
        # 验证B样条参数
        n = len(ctrl_pts)
        m = len(knots)
        if m != n + p + 1:
            print(f"警告: B样条参数不匹配。控制点数={n}, 节点数={m}, 阶数={p}")
            print(f"标准关系应为: 节点数 = 控制点数 + 阶数 + 1 = {n + p + 1}")
        
        if not np.all(np.diff(knots) >= 0):
            print("错误: 节点向量不是单调非递减的")
            exit(1)
            
        # 检查边界节点重复度
        unique_start = np.sum(knots == knots[0])
        unique_end = np.sum(knots == knots[-1])
        print(f"起始节点重复度: {unique_start}, 结束节点重复度: {unique_end}")
        if unique_start < p + 1 or unique_end < p + 1:
            print(f"警告: 边界节点重复度可能不足。对于{p}次B样条，建议重复度至少为{p+1}")
            
        # 检查有效参数域
        if knots[p] >= knots[-(p+1)]:
            print("错误: 有效参数域为空")
            exit(1)

        # 构造B样条曲线
        x_ctrl, y_ctrl = ctrl_pts[:, 0], ctrl_pts[:, 1]
        spl_x = BSpline(knots, x_ctrl, p)
        spl_y = BSpline(knots, y_ctrl, p)
        
        u_min = knots[p]
        u_max = knots[-(p+1)]
        print(f"B样条有效参数域: [{u_min:.6f}, {u_max:.6f}]")
        
        u_vals = np.linspace(u_min, u_max, 300)
        curve_x = spl_x(u_vals)
        curve_y = spl_y(u_vals)
        curve = np.column_stack((curve_x, curve_y))

        # 计算视图范围
        focus_x_min, focus_x_max, focus_y_min, focus_y_max = get_focus_limits(points)
        global_x_min, global_x_max, global_y_min, global_y_max = get_global_limits(curve, ctrl_pts, points)
        
        print(f"全局视图范围: X [{global_x_min:.4f}, {global_x_max:.4f}], Y [{global_y_min:.4f}, {global_y_max:.4f}]")
        print(f"聚焦视图范围: X [{focus_x_min:.4f}, {focus_x_max:.4f}], Y [{focus_y_min:.4f}, {focus_y_max:.4f}]")

        # 创建双视图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('B样条曲线对比视图', fontsize=16)
        # 设置中文显示（使用系统可用字体）
        plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans", "DejaVu Sans"]
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']  # 使用系统中可用的CJK字体
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
        # 全局视图
        ax1.plot(curve_x, curve_y, 'b-', label='B样条曲线')
        ax1.plot(x_ctrl, y_ctrl, 'ro--', label='控制点')
        ax1.scatter(points[:, 0], points[:, 1], color='green', s=20, label='原始点')
        ax1.set_title('全局视图')
        ax1.set_xlim(global_x_min, global_x_max)
        ax1.set_ylim(global_y_min, global_y_max)
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 聚焦视图（原始点区域）
        ax2.plot(curve_x, curve_y, 'b-', label='B样条曲线')
        ax2.plot(x_ctrl, y_ctrl, 'ro--', label='控制点')
        ax2.scatter(points[:, 0], points[:, 1], color='green', s=20, label='原始点')
        ax2.set_title('原始点区域聚焦视图')
        ax2.set_xlim(focus_x_min, focus_x_max)
        ax2.set_ylim(focus_y_min, focus_y_max)
        ax2.set_aspect('equal', adjustable='box')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('bspline_both_views.png', dpi=300, bbox_inches='tight')
        print("图片已保存为 bspline_both_views.png")
        # plt.show()
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        exit(1)
    