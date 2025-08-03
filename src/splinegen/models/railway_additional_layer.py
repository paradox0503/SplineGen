from torch import nn
import math
import torch

class RailwayAdditionalLayer(nn.Module):
    """
    铁路线元精调层 - 基于统一线元数学模型
    支持直线、圆曲线、缓和曲线的统一表示
    """

    def __init__(self, degree=3, dimension=2):  # 铁路一般为2D平面
        super().__init__()
        self.degree = degree
        self.dimension = dimension
        
        # 铁路设计参数
        self.min_radius = 300.0  # 最小曲线半径(m)
        self.max_curvature = 1.0 / self.min_radius
        self.max_transition_length = 150.0  # 最大缓和曲线长度(m)
        
        # 精调网络：预测线元参数
        self.element_finetune = TunableLayer(
            input_dim=dimension+1,  # 点坐标+弧长参数
            d_model=512, nhead=4, 
            num_decoder_layers=3, num_encoder_layers=3,
            dim_feedforward=2048, dropout=0.01,
            tgt_input_dim=4,  # [起点曲率, 终点曲率, 长度, 偏向系数]
            tgt_out_dim=4,
            p=3, max_knots_len=30, max_seq_len=30
        )
        
        # 参数精调网络
        self.param_finetune = TunableLayer(
            input_dim=4,  # 线元参数
            tgt_input_dim=dimension+1,
            d_model=512, nhead=4,
            num_decoder_layers=3, num_encoder_layers=3, 
            dim_feedforward=2048, dropout=0.01,
            p=3, max_knots_len=30, max_seq_len=30
        )
        
    def forward(self, points, params, points_mask, knots, knots_mask):
        """
        前向传播：从NURBS参数转换为铁路线元参数
        
        Args:
            points: 观测点 [batch_size, max_points, 2]
            params: 参数化坐标 [batch_size, max_points] 
            points_mask: 点有效性掩码
            knots: 节点向量（转换为线元分段点）
            knots_mask: 节点掩码
            
        Returns:
            new_params: 优化后的弧长参数化
            railway_elements: 铁路线元参数
        """
        
        # 1. 计算弧长参数化
        arc_length_params = self.compute_arc_length_parameterization(points, points_mask)
        
        # 2. 从观测点估计初始线元参数
        initial_elements = self.estimate_railway_elements(points, arc_length_params, points_mask, knots, knots_mask)
        
        # 3. 构建输入特征
        src_input = torch.cat([arc_length_params.unsqueeze(-1), points], dim=-1)
        
        # 4. 精调线元参数
        refined_elements = self.element_finetune(
            src_input, points_mask, 
            initial_elements, knots_mask
        )
        
        # 5. 应用铁路线元约束
        constrained_elements = self.apply_railway_constraints(refined_elements, knots_mask)
        
        # 6. 基于线元参数优化弧长参数化
        refined_arc_params = self.param_finetune(
            constrained_elements, knots_mask,
            src_input, points_mask
        )
        
        return refined_arc_params.squeeze(-1), constrained_elements
    
    def compute_arc_length_parameterization(self, points, points_mask):
        """
        计算弧长参数化
        """
        batch_size = points.shape[0]
        device = points.device
        
        # 计算相邻点间距离
        point_diffs = torch.diff(points, dim=1)  # [batch_size, max_points-1, 2]
        distances = torch.norm(point_diffs, dim=-1)  # [batch_size, max_points-1]
        
        # 处理掩码
        valid_distances = distances * points_mask[:, 1:].float()
        
        # 累积弧长
        cumulative_length = torch.zeros(batch_size, points.shape[1], device=device)
        cumulative_length[:, 1:] = torch.cumsum(valid_distances, dim=1)
        
        # 归一化到[0,1]区间（保持与原始参数化兼容）
        max_length = cumulative_length.max(dim=1, keepdim=True)[0]
        max_length = torch.where(max_length > 0, max_length, torch.ones_like(max_length))
        normalized_arc_length = cumulative_length / max_length
        
        return normalized_arc_length
    
    def estimate_railway_elements(self, points, arc_params, points_mask, knots, knots_mask):
        """
        从观测点估计初始铁路线元参数
        
        Returns:
            elements: [batch_size, max_elements, 4] 
                     [起点曲率, 终点曲率, 长度, 偏向系数]
        """
        batch_size, max_elements = knots.shape
        device = points.device
        
        elements = torch.zeros(batch_size, max_elements, 4, device=device)
        
        for b in range(batch_size):
            valid_points = points[b][points_mask[b]]
            valid_arc = arc_params[b][points_mask[b]]
            
            if len(valid_points) < 3:
                continue
                
            # 估计曲率
            curvatures = self.estimate_curvature_from_points(valid_points)
            
            # 分段处理
            num_valid_elements = knots_mask[b].sum().item()
            if num_valid_elements > 0:
                segment_length = len(curvatures) // num_valid_elements
                
                for i in range(num_valid_elements):
                    start_idx = i * segment_length
                    end_idx = min((i + 1) * segment_length, len(curvatures))
                    
                    if start_idx < end_idx:
                        segment_curvatures = curvatures[start_idx:end_idx]
                        
                        # 线元参数
                        k_start = segment_curvatures[0]
                        k_end = segment_curvatures[-1] if len(segment_curvatures) > 1 else k_start
                        length = valid_arc[end_idx-1] - valid_arc[start_idx] if end_idx > start_idx else 0.1
                        direction = 1.0 if k_start >= 0 else -1.0  # 偏向系数
                        
                        elements[b, i] = torch.tensor([k_start, k_end, length, direction], device=device)
        
        return elements
    
    def estimate_curvature_from_points(self, points):
        """
        从点序列估计曲率
        使用三点法计算离散曲率
        """
        if len(points) < 3:
            return torch.zeros(len(points), device=points.device)
        
        curvatures = torch.zeros(len(points), device=points.device)
        
        for i in range(1, len(points) - 1):
            p1, p2, p3 = points[i-1], points[i], points[i+1]
            
            # 向量
            v1 = p2 - p1
            v2 = p3 - p2
            
            # 叉积计算曲率
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            
            # 边长
            l1 = torch.norm(v1)
            l2 = torch.norm(v2)
            
            if l1 > 1e-6 and l2 > 1e-6:
                # 曲率公式：κ = 2 * sin(θ) / |chord|
                curvature = 2 * cross_product / (l1 * l2 * (l1 + l2))
                curvatures[i] = curvature
        
        # 边界处理
        curvatures[0] = curvatures[1] if len(curvatures) > 1 else 0
        curvatures[-1] = curvatures[-2] if len(curvatures) > 1 else 0
        
        return curvatures
    
    def apply_railway_constraints(self, elements, mask):
        """
        应用铁路线元约束
        
        Args:
            elements: [batch_size, max_elements, 4] [k_start, k_end, length, direction]
        """
        batch_size, max_elements, _ = elements.shape
        constrained = elements.clone()
        
        # 1. 曲率限制
        constrained[:, :, 0] = torch.clamp(constrained[:, :, 0], -self.max_curvature, self.max_curvature)  # k_start
        constrained[:, :, 1] = torch.clamp(constrained[:, :, 1], -self.max_curvature, self.max_curvature)  # k_end
        
        # 2. 长度限制
        constrained[:, :, 2] = torch.clamp(constrained[:, :, 2], 0.1, 500.0)  # length
        
        # 3. 偏向系数约束（只能为±1）
        constrained[:, :, 3] = torch.sign(constrained[:, :, 3])
        constrained[:, :, 3] = torch.where(constrained[:, :, 3] == 0, 
                                         torch.ones_like(constrained[:, :, 3]), 
                                         constrained[:, :, 3])
        
        # 4. 连续性约束：相邻线元的曲率连续
        for b in range(batch_size):
            valid_elements = mask[b].sum().item()
            for i in range(1, valid_elements):
                # 前一线元的终点曲率 = 当前线元的起点曲率
                constrained[b, i, 0] = constrained[b, i-1, 1]
        
        # 5. 应用掩码
        mask_expanded = mask.unsqueeze(-1).expand_as(constrained)
        constrained = constrained * mask_expanded.float()
        
        return constrained
    
    def evaluate_railway_curve(self, elements, arc_params, mask):
        """
        基于铁路线元参数计算理论曲线点
        
        使用线元通用数学模型：
        - 曲率：k(l) = k_s + (k_e - k_s) * l / L
        - 方位角：α(l) = α_0 + S_ccw * ∫k(l)dl  
        - 坐标：X(l) = x_0 + ∫cos(α(l))dl, Y(l) = y_0 + ∫sin(α(l))dl
        """
        batch_size, num_points = arc_params.shape
        device = elements.device
        
        # 初始化输出
        theoretical_points = torch.zeros(batch_size, num_points, 2, device=device)
        
        # 初始状态
        current_pos = torch.zeros(batch_size, 2, device=device)  # [x_0, y_0]
        current_angle = torch.zeros(batch_size, device=device)   # α_0
        
        for b in range(batch_size):
            valid_elements = mask[b].sum().item()
            points_per_element = num_points // max(valid_elements, 1)
            
            for i in range(valid_elements):
                element = elements[b, i]  # [k_start, k_end, length, direction]
                k_start, k_end, length, s_ccw = element
                
                # 当前线元的弧长参数范围
                start_idx = i * points_per_element
                end_idx = min((i + 1) * points_per_element, num_points)
                
                if start_idx >= end_idx:
                    continue
                
                # 线元内的局部弧长参数
                local_arc = torch.linspace(0, length.item(), end_idx - start_idx, device=device)
                
                # 计算线元内各点
                element_points = self.evaluate_single_railway_element(
                    k_start, k_end, length, s_ccw, local_arc, 
                    current_pos[b], current_angle[b]
                )
                
                theoretical_points[b, start_idx:end_idx] = element_points
                
                # 更新当前状态到线元终点
                if len(local_arc) > 0:
                    # 终点方位角
                    if torch.abs(k_start - k_end) < 1e-6:  # 圆曲线或直线
                        angle_change = s_ccw * k_start * length
                    else:  # 缓和曲线
                        angle_change = s_ccw * (k_start + k_end) * length / 2
                    
                    current_angle[b] += angle_change
                    
                    # 终点坐标（需要积分计算，这里简化）
                    if torch.abs(k_start) < 1e-6 and torch.abs(k_end) < 1e-6:  # 直线
                        dx = length * torch.cos(current_angle[b] - angle_change)
                        dy = length * torch.sin(current_angle[b] - angle_change)
                    else:  # 曲线（简化处理）
                        dx = element_points[-1, 0] - current_pos[b, 0]
                        dy = element_points[-1, 1] - current_pos[b, 1]
                    
                    current_pos[b] += torch.tensor([dx, dy], device=device)
        
        return theoretical_points
    
    def evaluate_single_railway_element(self, k_start, k_end, length, s_ccw, local_arc, start_pos, start_angle):
        """
        计算单个铁路线元内的点坐标
        """
        num_points = len(local_arc)
        device = local_arc.device
        
        points = torch.zeros(num_points, 2, device=device)
        
        if torch.abs(k_start) < 1e-6 and torch.abs(k_end) < 1e-6:
            # 直线段：k(l) = 0
            # X(l) = x_0 + l*cos(α_0), Y(l) = y_0 + l*sin(α_0)
            points[:, 0] = start_pos[0] + local_arc * torch.cos(start_angle)
            points[:, 1] = start_pos[1] + local_arc * torch.sin(start_angle)
            
        elif torch.abs(k_start - k_end) < 1e-6:
            # 圆曲线段：k(l) = k_start (常数)
            # 简化的圆弧公式
            radius = 1.0 / torch.abs(k_start) if torch.abs(k_start) > 1e-6 else 1e6
            
            for i, l in enumerate(local_arc):
                angle = start_angle + s_ccw * k_start * l
                if torch.abs(k_start) > 1e-6:
                    # 圆弧积分结果
                    points[i, 0] = start_pos[0] + s_ccw * radius * (torch.sin(angle) - torch.sin(start_angle))
                    points[i, 1] = start_pos[1] - s_ccw * radius * (torch.cos(angle) - torch.cos(start_angle))
                else:
                    # 退化为直线
                    points[i, 0] = start_pos[0] + l * torch.cos(start_angle)
                    points[i, 1] = start_pos[1] + l * torch.sin(start_angle)
        
        else:
            # 缓和曲线段：k(l) = k_start + (k_end - k_start) * l / length
            # 需要数值积分计算坐标
            current_pos = start_pos.clone()
            current_angle = start_angle
            
            points[0] = current_pos
            
            for i in range(1, num_points):
                dl = local_arc[i] - local_arc[i-1]
                l_mid = (local_arc[i] + local_arc[i-1]) / 2
                
                # 中点曲率
                k_mid = k_start + (k_end - k_start) * l_mid / length
                
                # 更新方位角
                current_angle += s_ccw * k_mid * dl
                
                # 更新坐标
                current_pos[0] += dl * torch.cos(current_angle)
                current_pos[1] += dl * torch.sin(current_angle)
                
                points[i] = current_pos.clone()
        
        return points


# 复用原有的TunableLayer和PositionalEncoding类
class TunableLayer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, dropout, p, max_knots_len, max_seq_len, 
                 tgt_input_dim=1, tgt_out_dim=1, head=True):
        
        super(TunableLayer, self).__init__()
        self.input_dim = input_dim
        self.p = p
        self.embedding = nn.Linear(input_dim, d_model)
        self.embedding_tgt = nn.Linear(tgt_input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, 
                                        num_decoder_layers, dim_feedforward, dropout,
                                        batch_first=True)
        self.fully_connected = nn.Linear(d_model, tgt_out_dim)
        self.head = head

    def forward(self, src, src_mask_position, tgt, tgt_mask_position):
        src_key_padding_mask = ~src_mask_position
        tgt_key_padding_mask = ~tgt_mask_position
        
        src = self.embedding(src)
        tgt = self.embedding_tgt(tgt)
        
        transformer_output = self.transformer(src, tgt, 
                                            src_key_padding_mask=src_key_padding_mask, 
                                            tgt_key_padding_mask=tgt_key_padding_mask)
        
        fc_output = self.fully_connected(transformer_output)
        
        # 处理掩码
        neg_inf_mask = torch.zeros_like(fc_output)
        tgt_key_padding_mask = tgt_key_padding_mask.unsqueeze(-1).expand_as(fc_output)
        neg_inf_mask.masked_fill_(tgt_key_padding_mask, float('-inf'))
        
        output = fc_output + neg_inf_mask
        
        if self.head:
            output = torch.sigmoid(output)
        
        # 应用位置掩码
        tgt_mask_position = tgt_mask_position.unsqueeze(-1).expand_as(output)
        output = output * tgt_mask_position.float()
        
        return output.squeeze(-1) if output.shape[-1] == 1 else output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
