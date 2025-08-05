import numpy as np

def explore_npz_file(input_path, output_path):
    """
    读取NPZ文件内容并输出到文本文件
    
    参数:
    input_path (str): 输入NPZ文件路径
    output_path (str): 输出文本文件路径
    """
    try:
        with np.load(input_path, allow_pickle=True) as data, open(output_path, 'w', encoding='utf-8') as f:
            # 1. 查看文件中包含的所有数组名称
            f.write("文件中包含的数组名称：" + str(data.files) + "\n")
            
            # 2. 遍历每个数组，查看形状和前几行内容
            for arr_name in data.files:
                # 获取数组
                arr = data[arr_name]
                f.write(f"\n===== 数组名称：{arr_name} =====\n")
                # 打印形状
                f.write(f"形状（shape）：{arr.shape}\n")
                # 打印维度
                f.write(f"维度（ndim）：{arr.ndim}\n")
                # 打印前5行（根据数组维度调整显示方式）
                # f.write("前5行内容：\n")
                if arr.ndim == 1:
                    # 1维数组：显示前5个元素
                    f.write(str(arr) + "\n")
                elif arr.ndim == 2:
                    # 2维数组：显示前5行
                    f.write(str(arr) + "\n")
                elif arr.ndim >= 3:
                    # 高维数组（如3D及以上）：显示第一个维度的前5个元素
                    f.write(str(arr) + "\n")  # 例如3D数组shape为(n, h, w)，显示前5个(h,w)的矩阵
        
        print(f"成功将NPZ文件内容输出到：{output_path}")
        
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_path}")
    except Exception as e:
        print(f"发生未知错误：{e}")

if __name__ == "__main__":

    # file_name="2d_eval"
    # file_name="2d_train"
    # file_name="3d_eval"
    # file_name="3d_train"
    file_name="spline_data_1754277229"
    
    input_file = file_name + ".npz"  # 字符串拼接
    output_file = file_name + ".txt"  # 输出文件路径
    
    # 执行程序
    explore_npz_file(input_file, output_file) 