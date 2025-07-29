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
    
    # 输出结果
    if output_file:
        with open(output_file, 'a') as f:  # 改为追加模式
            f.write(result + "\n\n")  # 添加额外的换行分隔
        print(f"已将处理后的控制点追加到 {output_file}")
    else:
        print(result)
      


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
    
    # 输出结果
    if output_file:
        with open(output_file, 'a') as f:  # 改为追加模式
            f.write(result + "\n\n")  # 添加额外的换行分隔
        print(f"已将处理后的节点数据追加到 {output_file}")
    else:
        print(result)

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
    
    # 输出结果
    if output_file:
        with open(output_file, 'a') as f:  # 改为追加模式
            f.write(result + "\n\n")  # 添加额外的换行分隔
        print(f"已将处理后的点数据追加到 {output_file}")
    else:
        print(result)
# 使用示例
if __name__ == "__main__":
    # 读取数据获取控制点数量
    data = np.load("spline_data.npz")
    ctrl_pts = data['ctrl_pts']
    non_zero_mask = ~np.all(ctrl_pts == 0, axis=1)
    len_ctrl_pts = np.sum(non_zero_mask)
    max_length = len_ctrl_pts + 4
    
    output_file = "all_processed_data.txt"
    
    # 创建文件并写入头部注释
    with open(output_file, 'w') as f:
        f.write("# 从spline_data.npz提取的所有数据\n")
        f.write(f"# 控制点数量: {len_ctrl_pts}\n")
        f.write(f"# knots最大长度: {max_length}\n\n")
    
    # 依次追加各种数据到同一个文件
    convert_ctrl_pts("spline_data.npz", output_file)
    convert_knots("spline_data.npz", max_length=max_length, output_file=output_file)
    convert_points("spline_data.npz", output_file=output_file)
    
    print(f"\n所有数据已写入到 {output_file}")
