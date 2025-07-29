import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline
import numpy as np


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
    # 读取数据获取控制点数量
    data = np.load("spline_data.npz")
    ctrl_pts = data['ctrl_pts']
    non_zero_mask = ~np.all(ctrl_pts == 0, axis=1)
    len_ctrl_pts = np.sum(non_zero_mask)
    max_length = len_ctrl_pts + 4
    
    output_file = "all_processed_data.txt"
    
    # 依次获取实际数据（不是字符串格式）
    data = np.load("spline_data.npz")
    
    # 获取控制点
    ctrl_pts = data['ctrl_pts']
    non_zero_mask = ~np.all(ctrl_pts == 0, axis=1)
    ctrl_pts = ctrl_pts[non_zero_mask]
    
    # 获取knots
    knots = data['knots'][:max_length]
    
    # 获取原始数据点
    points = data['points']
    non_zero_points_mask = ~np.all(points == 0, axis=1)
    points = points[non_zero_points_mask]
    
    print(f"控制点数量: {len(ctrl_pts)}")
    print(f"knots数量: {len(knots)}")
    print(f"原始数据点数量: {len(points)}")

    # B样条阶数
    p = 3

    # 构造B样条曲线
    x = ctrl_pts[:, 0]
    y = ctrl_pts[:, 1]
    spl_x = BSpline(knots, x, p)
    spl_y = BSpline(knots, y, p)
    u_vals = np.linspace(0, 1, 200)
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
    plt.show()
    plt.savefig('bspline_curve.png')  # 保存为 PNG 图片，会生成在代码同目录下
