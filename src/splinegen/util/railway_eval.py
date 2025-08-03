import torch

def getRailwayElementFit_vectorized(degree, input):
    """
    铁路线元拟合的向量化损失函数
    替代原有的getCtrlPts_vectorized函数
    
    Args:
        degree: 线形次数（保持兼容性）
        input: [params, points, points_mask, railway_elements, element_mask]
               - params: 弧长参数化 [batch_size, max_points]
               - points: 观测点坐标 [batch_size, max_points, 2]
               - points_mask: 点有效性掩码
               - railway_elements: 线元参数 [batch_size, max_elements, 4]
               - element_mask: 线元有效性掩码
    
    Returns:
        loss: 铁路线元拟合损失
        railway_elements: 优化后的线元参数
    """
    params, points, points_mask, railway_elements, element_mask = input
    device = points.device
    batch_size = points.shape[0]
    
    # 1. 基于线元参数计算理论曲线点
    theoretical_points = evaluate_railway_curve_batch(railway_elements, params, element_mask, points.shape[1])
    
    # 2. 计算拟合损失
    loss = compute_railway_fitting_loss(theoretical_points, points, points_mask)
    
    return loss, railway_elements


def evaluate_railway_curve_batch(elements, arc_params, element_mask, num_points):
    """
    批量计算铁路线元理论曲线
    """
    batch_size, max_elements, _ = elements.shape
    device = elements.device
    
    # 输出：理论点坐标
    theoretical_points = torch.zeros(batch_size, num_points, 2, device=device)
    
    for b in range(batch_size):
        theoretical_points[b] = evaluate_railway_curve_single(
            elements[b], arc_params[b], element_mask[b], num_points
        )
    
    return theoretical_points


def evaluate_railway_curve_single(elements, arc_params, element_mask, num_points):
    """
    计算单个批次的铁路线元曲线
    
    基于线元通用数学模型：
    1. 曲率：k(l) = k_s + (k_e - k_s) * l / L
    2. 方位角：α(l) = α_0 + S_ccw * ∫k(l)dl
    3. 坐标：X(l) = x_0 + ∫cos(α(l))dl, Y(l) = y_0 + ∫sin(α(l))dl
    """
    device = elements.device
    valid_elements = element_mask.sum().item()
    
    if valid_elements == 0:
        return torch.zeros(num_points, 2, device=device)
    
    # 初始状态
    current_pos = torch.zeros(2, device=device)      # [x_0, y_0]
    current_angle = torch.tensor(0.0, device=device) # α_0
    
    # 分配点到各线元
    points_per_element = num_points // valid_elements
    remainder_points = num_points % valid_elements
    
    theoretical_points = torch.zeros(num_points, 2, device=device)
    
    point_idx = 0
    for i in range(valid_elements):
        element = elements[i]  # [k_start, k_end, length, direction]
        k_start, k_end, length, s_ccw = element
        
        # 当前线元的点数
        current_element_points = points_per_element
        if i < remainder_points:
            current_element_points += 1
        
        if current_element_points == 0:
            continue
        
        # 线元内的局部弧长参数
        local_arc = torch.linspace(0, length.item(), current_element_points, device=device)
        
        # 计算线元类型并评估
        element_points = evaluate_single_railway_element_vectorized(
            k_start, k_end, length, s_ccw, local_arc, current_pos, current_angle
        )
        
        # 存储结果
        end_idx = point_idx + current_element_points
        theoretical_points[point_idx:end_idx] = element_points
        
        # 更新当前状态到线元终点
        if current_element_points > 0:
            current_pos, current_angle = update_railway_element_state(
                k_start, k_end, length, s_ccw, current_pos, current_angle
            )
        
        point_idx = end_idx
    
    return theoretical_points


def evaluate_single_railway_element_vectorized(k_start, k_end, length, s_ccw, local_arc, start_pos, start_angle):
    """
    向量化计算单个铁路线元
    """
    num_points = len(local_arc)
    device = local_arc.device
    
    points = torch.zeros(num_points, 2, device=device)
    
    # 判断线元类型
    is_straight = (torch.abs(k_start) < 1e-6) and (torch.abs(k_end) < 1e-6)
    is_circular = torch.abs(k_start - k_end) < 1e-6
    
    if is_straight:
        # 直线段：k(l) = 0
        # 坐标公式：X(l) = x_0 + l*cos(α_0), Y(l) = y_0 + l*sin(α_0)
        points[:, 0] = start_pos[0] + local_arc * torch.cos(start_angle)
        points[:, 1] = start_pos[1] + local_arc * torch.sin(start_angle)
        
    elif is_circular and torch.abs(k_start) > 1e-6:
        # 圆曲线段：k(l) = k_start (常数)
        # 使用圆曲线积分公式
        radius = 1.0 / torch.abs(k_start)
        
        # 向量化计算方位角
        angles = start_angle + s_ccw * k_start * local_arc
        
        # 圆曲线坐标积分公式
        points[:, 0] = start_pos[0] + s_ccw * radius * (torch.sin(angles) - torch.sin(start_angle))
        points[:, 1] = start_pos[1] - s_ccw * radius * (torch.cos(angles) - torch.cos(start_angle))
        
    else:
        # 缓和曲线段：k(l) = k_start + (k_end - k_start) * l / length
        # 需要数值积分
        points = evaluate_transition_curve_numerical(
            k_start, k_end, length, s_ccw, local_arc, start_pos, start_angle
        )
    
    return points


def evaluate_transition_curve_numerical(k_start, k_end, length, s_ccw, local_arc, start_pos, start_angle):
    """
    数值积分计算缓和曲线
    """
    num_points = len(local_arc)
    device = local_arc.device
    
    points = torch.zeros(num_points, 2, device=device)
    points[0] = start_pos
    
    current_pos = start_pos.clone()
    current_angle = start_angle
    
    for i in range(1, num_points):
        dl = local_arc[i] - local_arc[i-1]
        l_mid = (local_arc[i] + local_arc[i-1]) / 2
        
        # 中点曲率：k(l) = k_s + (k_e - k_s) * l / L
        if length > 1e-6:
            k_mid = k_start + (k_end - k_start) * l_mid / length
        else:
            k_mid = k_start
        
        # 更新方位角：dα = S_ccw * k(l) * dl
        current_angle += s_ccw * k_mid * dl
        
        # 更新坐标：dx = cos(α) * dl, dy = sin(α) * dl
        current_pos[0] += dl * torch.cos(current_angle)
        current_pos[1] += dl * torch.sin(current_angle)
        
        points[i] = current_pos.clone()
    
    return points


def update_railway_element_state(k_start, k_end, length, s_ccw, current_pos, current_angle):
    """
    更新线元结束后的位置和方位角
    """
    device = current_pos.device
    
    # 计算方位角变化
    if torch.abs(k_start - k_end) < 1e-6:
        # 直线或圆曲线
        angle_change = s_ccw * k_start * length
    else:
        # 缓和曲线：∫k(l)dl = ∫[k_s + (k_e - k_s)*l/L]dl = k_s*L + (k_e - k_s)*L/2
        angle_change = s_ccw * (k_start + k_end) * length / 2
    
    new_angle = current_angle + angle_change
    
    # 计算位置变化（简化方法：用平均方位角）
    avg_angle = current_angle + angle_change / 2
    new_pos = current_pos + length * torch.tensor([torch.cos(avg_angle), torch.sin(avg_angle)], device=device)
    
    return new_pos, new_angle


def compute_railway_fitting_loss(theoretical_points, observed_points, points_mask):
    """
    计算铁路线元拟合损失
    
    Args:
        theoretical_points: 理论曲线点 [batch_size, num_points, 2]
        observed_points: 观测点 [batch_size, num_points, 2]
        points_mask: 点有效性掩码 [batch_size, num_points]
    
    Returns:
        loss: 总拟合损失
    """
    # 计算点之间的距离误差
    point_errors = torch.norm(theoretical_points - observed_points, dim=-1)  # [batch_size, num_points]
    
    # 应用掩码，只计算有效点的损失
    masked_errors = point_errors * points_mask.float()
    
    # 计算平均损失
    total_valid_points = points_mask.sum()
    if total_valid_points > 0:
        loss = masked_errors.sum() / total_valid_points
    else:
        loss = torch.tensor(0.0, device=theoretical_points.device)
    
    # 添加平顺性约束损失
    smoothness_loss = compute_smoothness_loss(theoretical_points, points_mask)
    
    # 总损失
    total_loss = loss + 0.1 * smoothness_loss  # 平顺性权重
    
    return total_loss


def compute_smoothness_loss(points, mask):
    """
    计算曲线平顺性损失
    基于二阶差分（曲率变化率）
    """
    if points.shape[1] < 3:
        return torch.tensor(0.0, device=points.device)
    
    # 计算一阶差分（速度）
    first_diff = torch.diff(points, dim=1)  # [batch_size, num_points-1, 2]
    
    # 计算二阶差分（加速度/曲率变化）
    second_diff = torch.diff(first_diff, dim=1)  # [batch_size, num_points-2, 2]
    
    # 曲率变化的模长
    curvature_change = torch.norm(second_diff, dim=-1)  # [batch_size, num_points-2]
    
    # 应用掩码（注意维度调整）
    valid_mask = mask[:, 2:].float()  # 对应二阶差分的有效位置
    
    # 平均曲率变化率
    total_valid = valid_mask.sum()
    if total_valid > 0:
        smoothness_loss = (curvature_change * valid_mask).sum() / total_valid
    else:
        smoothness_loss = torch.tensor(0.0, device=points.device)
    
    return smoothness_loss
