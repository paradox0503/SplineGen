import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
import os


def convert_ctrl_pts(npz_file_path, output_file=None):
    # 读取npz文件
    data = np.load(npz_file_path)
    # 提取控制点并筛选非零值（排除全为0的行）
    ctrl_pts = data['ctrl_pts']
    non_zero_mask = ~np.all(ctrl_pts == 0, axis=1)  # 排除全零行
    filtered_pts = ctrl_pts[non_zero_mask]
    
    # 生成格式字符串
    # 首行
    result = "ctrl_pts = np.array([\n"
    # 中间行（每个点一行，保留4位小数）
    for i, pt in enumerate(filtered_pts):
        x, y = pt
        # 最后一行不加逗号
        line_end = ",\n" if i != len(filtered_pts) - 1 else "\n"
        result += f"    [{x:>8.4f}, {y:>8.4f}]{line_end}"
    # 尾行
    result += "])"
    
    return result
      


def convert_knots(npz_file_path, output_file=None, max_length=None):
    data = np.load(npz_file_path)
    knots = data['knots']

    if 'ctrl_pts' in data:
        ctrl_pts = data['ctrl_pts']
        non_zero_ctrl = ctrl_pts[~np.all(ctrl_pts == 0, axis=1)]
        n = len(non_zero_ctrl)  # 控制点数量
        p = 3  # 样条阶数（可根据实际情况调整）
        recommended_length = n + p + 1
        
        # 如果未指定最大长度，使用推荐长度
        if max_length is None:
            max_length = recommended_length
            print(f"根据控制点数量自动确定节点合理长度: {max_length}")
    
    # 截取节点数据（如果指定了最大长度）
    if max_length and len(knots) > max_length:
        knots = knots[:max_length]
        print(f"节点数据已截取至{max_length}个元素")
    
    # 生成格式字符串
    result = "knot_u = np.array([\n"
    # 每行显示5个元素（可根据需要调整）
    elements_per_line = 5
    total = len(knots)
    
    for i, val in enumerate(knots):
        # 计算是否为行末
        line_end = ",\n" if (i + 1) % elements_per_line == 0 and i != total - 1 else ", "
        # 最后一个元素不加逗号
        if i == total - 1:
            line_end = "\n"
        # 保留4位小数，右对齐
        result += f"    {val:>8.4f}{line_end}"
    
    result += "])"
    
    return result

def convert_points(npz_file_path, output_file=None, keep_zero_rows=False):
    # 读取npz文件
    data = np.load(npz_file_path)
    
    # 提取points数据
    points = data['points']
    
    # 筛选非零数据点（除非指定保留全零行）
    if not keep_zero_rows:
        non_zero_mask = ~np.all(points == 0, axis=1)
        points = points[non_zero_mask]
        print(f"已过滤全零数据点，保留 {len(points)} 个有效点")
    
    # 生成格式字符串
    result = "points = np.array([\n"
    for i, pt in enumerate(points):
        x, y = pt
        # 处理行尾逗号
        line_end = ",\n" if i != len(points) - 1 else "\n"
        # 保留4位小数，右对齐，保持格式一致
        result += f"    [{x:>8.4f}, {y:>8.4f}]{line_end}"
    
    result += "])"
    
    return result
# 使用示例
if __name__ == "__main__":
    # 检查文件是否存在
    npz_file = "spline_data_1754277229.npz"
    if not os.path.exists(npz_file):
        print(f"错误: 文件 {npz_file} 不存在")
        exit(1)
    
    try:
        # 读取数据获取控制点数量（只需加载一次）
        data = np.load(npz_file)
        
        # 检查必要的键是否存在
        required_keys = ['ctrl_pts', 'knots', 'points']
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"错误: NPZ文件中缺少以下键: {missing_keys}")
            exit(1)
            
        ctrl_pts = data['ctrl_pts']
        non_zero_mask = ~np.all(ctrl_pts == 0, axis=1)
        len_ctrl_pts = np.sum(non_zero_mask)
        max_length = len_ctrl_pts + 4
        
        output_file = "all_processed_data.txt"
        
        # 获取控制点
        ctrl_pts = data['ctrl_pts']
        non_zero_mask = ~np.all(ctrl_pts == 0, axis=1)
        ctrl_pts = ctrl_pts[non_zero_mask]
        
        # B样条阶数（提前定义）
        p = 3
        
        # 获取knots - 更智能的截取方式
        all_knots = data['knots']
        
        # 如果节点数量超过推荐值，尝试保持正确的B样条结构
        if len(all_knots) > max_length:
            print(f"原始节点数量 {len(all_knots)} 超过推荐值 {max_length}")
            
            # 检查是否可以安全截取
            # 确保截取后仍然有正确的边界重复度
            if len(all_knots) >= len_ctrl_pts + 2 * (p + 1):
                # 保持前后各p+1个重复节点
                mid_knots_needed = len_ctrl_pts - (p + 1)
                if mid_knots_needed > 0:
                    # 从中间部分选择节点
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
        
        # 获取原始数据点
        points = data['points']
        non_zero_points_mask = ~np.all(points == 0, axis=1)
        points = points[non_zero_points_mask]
        
        print(f"控制点数量: {len(ctrl_pts)}")
        print(f"knots数量: {len(knots)}")
        print(f"原始数据点数量: {len(points)}")
        
        # 验证B样条参数
        n = len(ctrl_pts)  # 控制点数量
        m = len(knots)     # 节点数量
        if m != n + p + 1:
            print(f"警告: B样条参数不匹配。控制点数={n}, 节点数={m}, 阶数={p}")
            print(f"标准关系应为: 节点数 = 控制点数 + 阶数 + 1 = {n + p + 1}")
        
        # 验证节点向量单调性
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
        x = ctrl_pts[:, 0]
        y = ctrl_pts[:, 1]
        spl_x = BSpline(knots, x, p)
        spl_y = BSpline(knots, y, p)
        
        # 正确的参数域：从 knots[p] 到 knots[-(p+1)]
        u_min = knots[p]
        u_max = knots[-(p+1)]
        print(f"B样条有效参数域: [{u_min:.6f}, {u_max:.6f}]")
        
        u_vals = np.linspace(u_min, u_max, 200)
        curve = np.stack([spl_x(u_vals), spl_y(u_vals)], axis=1)

        # 绘图
        plt.figure(figsize=(8, 6))
        plt.plot(curve[:, 0], curve[:, 1], 'b-', label='B-spline Curve')
        plt.plot(x, y, 'ro--', label='Control Points')
        plt.scatter(points[:, 0], points[:, 1], color='green', s=20, label='Original Points')
        plt.title('B-spline Curve with Original Points')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        
        # 先保存再显示
        plt.savefig('bspline_curve.png', dpi=300, bbox_inches='tight')
        print("图片已保存为 bspline_curve.png")
        # plt.show()
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        exit(1)
