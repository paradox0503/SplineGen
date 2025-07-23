import torch
import time

'''
Modified from NURBS-Diff
(https://github.com/idealab-isu/NURBSDiff)
'''
class CurveEval(torch.nn.Module):
    def __init__(self, m, dimension=3, p=3, out_dim=128, method='tc', dvc='cuda'):
        super(CurveEval, self).__init__()
        self.m = m
        self._dimension = dimension
        self.p=p
        self.u = torch.linspace(1e-5, 1.0-1e-5, steps=out_dim, dtype=torch.float32,device=dvc)
        self.method = method
        self.dvc = dvc

    def forward(self,input):
        # input will be of dimension (batch_size, m+1, n+1, dimension)
        ctrl_pts, knot_u= input

        U=knot_u
        # U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
        # U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

        if torch.isnan(U).any():
            # print(U_c)
            print(knot_u)

        u = self.u.unsqueeze(0)
        uspan_uv = torch.stack([torch.min(torch.where((u - U[s,self.p:-self.p].unsqueeze(1))>1e-8, u - U[s,self.p:-self.p].unsqueeze(1), (u - U[s,self.p:-self.p].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.p for s in range(U.size(0))])

        u = u.squeeze(0)
        Ni = [u*0 for i in range(self.p+1)]
        Ni[0] = u*0 + 1
        for k in range(1,self.p+1):
            saved = (u)*0.0
            for r in range(k):
                UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
                UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])
                temp = Ni[r]/((UList1 - u) + (u - UList2))
                temp = torch.where(((UList1 - u) + (u - UList2))==0.0, u*0+1e-4, temp)
                Ni[r] = saved + (UList1 - u)*temp
                saved = (u - UList2)*temp
            Ni[k] = saved

        Nu_uv = torch.stack(Ni).permute(1,0,2).unsqueeze(-1)

        pts = torch.stack([torch.stack([ctrl_pts[s,(uspan_uv[s,:]-self.p+l),:] for l in range(self.p+1)]) for s in range(U.size(0))])


        curve = torch.sum((Nu_uv*pts), dim=1)

        return curve

    def Max_MSE_Error(self,input,data_points:torch.Tensor):
        pred_points:torch.Tensor=self(input)

        diff=torch.linalg.norm(data_points-pred_points,dim=-1)

        max=torch.max(diff,dim=1)[0]
        mse=torch.mean(diff,dim=1)

        return max,mse

class CurveEval2(torch.nn.Module):
    def __init__(self, m, dimension=3, p=3, out_dim=128, method='tc', dvc='cuda'):
        super(CurveEval2, self).__init__()
        self.m = m
        self._dimension = dimension
        self.p=p
        self.u = torch.linspace(1e-5, 1.0-1e-5, steps=out_dim, dtype=torch.float32)
        self.method = method
        self.dvc = dvc

    def forward(self,input):
        # input will be of dimension (batch_size, m+1, n+1, dimension)
        ctrl_pts, knot_u= input

        U=knot_u
        # U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
        # U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

        if torch.isnan(U).any():
            # print(U_c)
            print(knot_u)

        u = self.u.unsqueeze(0)
        p=self.p
        # uspan_uv = torch.stack([torch.min(torch.where((u - U[s,self.p:-self.p].unsqueeze(1))>1e-8, u - U[s,self.p:-self.p].unsqueeze(1), (u - U[s,self.p:-self.p].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+self.p for s in range(U.size(0))])
        uspan_uv = torch.stack([torch.min(torch.where((u[s] - U[s,p:].unsqueeze(1))>1e-8, u[s] - U[s,p:].unsqueeze(1), (u[s] - U[s,p:].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+p for s in range(U.size(0))])

        u = self.u
        Ni = [u*0 for i in range(p+1)]
        Ni[0] = u*0 + 1
        for k in range(1,p+1):
            saved = (u)*0.0
            for r in range(k):
                UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
                UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])

                dU=((UList1 - u) + (u - UList2))
                dU_=torch.where(dU==0.0, u*0+1e-4, dU)

                temp = Ni[r]/dU_
                temp = torch.where(dU==0.0, u*0+1e-4, temp)
                Ni[r] = saved + (UList1 - u)*temp
                saved = (u - UList2)*temp
            Ni[k] = saved

        Nu_uv = torch.stack(Ni).permute(1,2,0)


        pts = torch.stack([torch.stack([ctrl_pts[s,(uspan_uv[s,:]-self.p+l),:] for l in range(self.p+1)]) for s in range(U.size(0))])

        curve = torch.sum((Nu_uv*pts), dim=1)

        return curve

class CurveEval3(torch.nn.Module):
    def __init__(self, m, dimension=3, p=3, out_dim=128, method='tc', dvc='cuda'):
        super(CurveEval3, self).__init__()
        self.m = m
        self._dimension = dimension
        self.p=p
        self.u = torch.linspace(1e-5, 1.0-1e-5, steps=out_dim, dtype=torch.float32)
        self.method = method
        self.dvc = dvc

    def forward(self,input):
        # input will be of dimension (batch_size, m+1, n+1, dimension)
        ctrl_pts, knot_u= input
        p=self.p

        # params,points,points_mask,knot_u,knots_len= input

        params=self.u.unsqueeze(0)

        device=params.device

        U=knot_u
        # U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
        # U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

        if torch.isnan(U).any():
            # print(U_c)
            print(knot_u)

        u = params.unsqueeze(1)
        uspan_uv = torch.stack([torch.min(torch.where((u[s] - U[s,p:].unsqueeze(1))>1e-8, u[s] - U[s,p:].unsqueeze(1), (u[s] - U[s,p:].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+p for s in range(U.size(0))])

        # torch._assert(torch.all(uspan_uv<25),"uspan out of bound")
        # print('uspan_uv:',uspan_uv)
        # print((u[0] - U[0,p:].unsqueeze(1))[...,35])

        u = u.squeeze(1)
        Ni = [u*0 for i in range(p+1)]
        Ni[0] = u*0 + 1
        for k in range(1,p+1):
            saved = (u)*0.0
            for r in range(k):
                UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
                UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])

                dU=((UList1 - u) + (u - UList2))
                dU_=torch.where(dU==0.0, u*0+1e-4, dU)

                temp = Ni[r]/dU_
                temp = torch.where(dU==0.0, u*0+1e-4, temp)
                Ni[r] = saved + (UList1 - u)*temp
                saved = (u - UList2)*temp
            Ni[k] = saved

        Nu_uv = torch.stack(Ni).permute(1,2,0)
        # torch.linalg.lstsq()
        scatter_index=torch.arange(-p,1,device=device).unsqueeze(0).unsqueeze(0).expand(uspan_uv.shape[0],uspan_uv.shape[1],-1)
        scatter_index=uspan_uv.unsqueeze(-1)+scatter_index

        N_all=torch.zeros((Nu_uv.size(0),Nu_uv.size(1),ctrl_pts.shape[1]),device=device)
        # print(143,scatter_index.dtype,Nu_uv.dtype,N_all.dtype)
        N_all=N_all.scatter_(2,scatter_index,Nu_uv)
        # N_mask=points_mask.unsqueeze(-1).expand(-1,-1,30)

        # N_all=torch.masked_fill(N_all,~N_mask.bool(),0)
            # N_all[s,uspan_uv[s,:]]

        solution=ctrl_pts.to(dtype=torch.float32)
        results=N_all@solution


        return results

def getCtrlPts(p,input):
    # input will be of dimension (batch_size, m+1, n+1, dimension)
    # 1 batch
    start_time = time.time()
    
    
    params,points,points_mask,knot_u,knots_len= input
    device=params.device
    import pdb;pdb.set_trace()
    
    # 数据预处理时间
    preprocess_start = time.time()
    params=torch.masked_fill(params,~points_mask.bool(),0)
    points=torch.masked_fill(points,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)
    preprocess_time = time.time() - preprocess_start

    U=knot_u
    # U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
    # U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

    if torch.isnan(U).any():
        # print(U_c)
        print(knot_u)

    # 参数计算时间
    param_calc_start = time.time()
    u = params.unsqueeze(1)
    uspan_uv = torch.stack([torch.min(torch.where((u[s] - U[s,p:].unsqueeze(1))>1e-8, u[s] - U[s,p:].unsqueeze(1), (u[s] - U[s,p:].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+p for s in range(U.size(0))])

    # torch._assert(torch.all(uspan_uv<25),"uspan out of bound")
    # print('uspan_uv:',uspan_uv)
    # print((u[0] - U[0,p:].unsqueeze(1))[...,35])

    u = u.squeeze(1)
    param_calc_time = time.time() - param_calc_start
    
    # 基函数计算时间
    basis_start = time.time()
    Ni = [u*0 for i in range(p+1)]
    Ni[0] = u*0 + 1
    for k in range(1,p+1):
        saved = (u)*0.0
        for r in range(k):
            UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
            UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])

            dU=((UList1 - u) + (u - UList2))
            dU_=torch.where(dU==0.0, u*0+1e-4, dU)

            temp = Ni[r]/dU_
            temp = torch.where(dU==0.0, u*0+1e-4, temp)
            Ni[r] = saved + (UList1 - u)*temp
            saved = (u - UList2)*temp
        Ni[k] = saved

    Nu_uv = torch.stack(Ni).permute(1,2,0)
    basis_time = time.time() - basis_start
    
    # 矩阵组装时间
    matrix_start = time.time()
    # torch.linalg.lstsq()
    scatter_index=torch.arange(-p,1,device=device).unsqueeze(0).unsqueeze(0).expand(uspan_uv.shape[0],uspan_uv.shape[1],-1)
    scatter_index=uspan_uv.unsqueeze(-1)+scatter_index

    N_all=torch.zeros((Nu_uv.size(0),Nu_uv.size(1),30),device=device)
    # print(143,scatter_index.dtype,Nu_uv.dtype,N_all.dtype)
    N_all=N_all.scatter_(2,scatter_index,Nu_uv)
    N_mask=points_mask.unsqueeze(-1).expand(-1,-1,30)

    N_all=torch.masked_fill(N_all,~N_mask.bool(),0)
    matrix_time = time.time() - matrix_start
    
    # 求解时间
    solve_start = time.time()
        # N_all[s,uspan_uv[s,:]]
    # solution=torch.linalg.pinv(N_all)@points
    try:
        solution = torch.linalg.lstsq(N_all, points).solution
    except:
        # 如果 lstsq 失败，回退到 pinv
        solution = torch.linalg.pinv(N_all) @ points
    
    solve_time = time.time() - solve_start
    
    # 残差计算时间
    residual_start = time.time()
    residuals=points-N_all@solution

    residuals=torch.masked_fill(residuals,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)

    loss=torch.linalg.norm(residuals,dim=-1).max(dim=-1)[0].sum()
    # loss=torch.linalg.norm(residuals,dim=-1).max(dim=-1)[0].sum()
    residual_time = time.time() - residual_start

    #### pinv method end #####
    
    total_time = time.time() - start_time
    
    # 输出时间统计
    print(f"NURBS计算时间统计:")
    print(f"  预处理时间: {preprocess_time*1000:.2f}ms")
    print(f"  参数计算时间: {param_calc_time*1000:.2f}ms") 
    print(f"  基函数计算时间: {basis_time*1000:.2f}ms")
    print(f"  矩阵组装时间: {matrix_time*1000:.2f}ms")
    print(f"  线性求解时间: {solve_time*1000:.2f}ms")
    print(f"  残差计算时间: {residual_time*1000:.2f}ms")
    print(f"  总计算时间: {total_time*1000:.2f}ms")
    print("-" * 40)
    total_time = time.time() - start_time

    return loss,solution
    # return torch.functional.F.mse_loss((N_all@solution),points)


def getCtrlPts_vectorized(p, input):
    params, points, points_mask, knot_u, knots_len = input
    device = params.device
    B, M = params.shape[:2]  # batch_size, max_points
    
    # 向量化预处理
    params = torch.masked_fill(params, ~points_mask.bool(), 0)
    points = torch.masked_fill(points, ~(points_mask.unsqueeze(-1).expand_as(points)).bool(), 0)
    
    U = knot_u
    u = params.unsqueeze(2)  # [B, M, 1]
    
    # 向量化 uspan 计算
    U_expanded = U[:, p:].unsqueeze(1)  # [B, 1, K-p]
    diff = u - U_expanded  # [B, M, K-p]
    mask = diff > 1e-8
    
    # 使用 torch.argmax 替代循环
    indices = torch.arange(U.shape[1] - p, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, K-p]
    valid_indices = torch.where(mask, indices, torch.full_like(indices, U.shape[1]))
    uspan_uv = torch.argmin(valid_indices, dim=-1) + p  # [B, M]
    
    # 向量化基函数计算
    u = u.squeeze(2)  # [B, M]
    
    # 使用更高效的基函数计算 - 避免原地操作
    Ni = torch.zeros(B, M, p+1, device=device)
    Ni[:, :, 0] = 1.0
    
    for k in range(1, p+1):
        saved = torch.zeros_like(u)
        # 创建新的 Ni 来避免原地操作
        Ni_new = Ni.clone()
        
        for r in range(k):
            # 向量化索引操作
            idx1 = uspan_uv + r + 1
            idx2 = uspan_uv + 1 - k + r
            
            # 确保索引在有效范围内
            idx1 = torch.clamp(idx1, 0, U.shape[1]-1)
            idx2 = torch.clamp(idx2, 0, U.shape[1]-1)
            
            UList1 = torch.gather(U, 1, idx1)
            UList2 = torch.gather(U, 1, idx2)
            
            dU = (UList1 - u) + (u - UList2)
            dU_safe = torch.where(dU == 0.0, torch.full_like(dU, 1e-4), dU)
            
            temp = Ni[:, :, r] / dU_safe
            temp = torch.where(dU == 0.0, torch.full_like(temp, 0.0), temp)
            
            saved_new = (u - UList2) * temp
            # 避免原地操作，创建新张量
            Ni_new[:, :, r] = saved + (UList1 - u) * temp
            saved = saved_new
        
        # 设置新的基函数值
        Ni_new[:, :, k] = saved
        Ni = Ni_new
    
    Nu_uv = Ni  # [B, M, p+1]
    
    # 向量化矩阵组装
    scatter_index = uspan_uv.unsqueeze(-1) + torch.arange(-p, 1, device=device).view(1, 1, -1)
    scatter_index = torch.clamp(scatter_index, 0, 29)  # 确保索引有效
    
    N_all = torch.zeros(B, M, 30, device=device)
    N_all.scatter_(2, scatter_index, Nu_uv)
    
    # 向量化mask处理
    N_mask = points_mask.unsqueeze(-1).expand(-1, -1, 30)
    N_all = torch.masked_fill(N_all, ~N_mask.bool(), 0)
    
    # 批量求解线性系统
    solution = torch.linalg.pinv(N_all) @ points
    
    # 向量化损失计算
    residuals = points - N_all @ solution
    residuals = torch.masked_fill(residuals, ~points_mask.unsqueeze(-1).expand_as(residuals).bool(), 0)
    
    loss = torch.linalg.norm(residuals, dim=-1).max(dim=-1)[0].sum()
    
    return loss, solution

def getCtrlPts_fast(p, input, reduce=True):
    params, points, points_mask, knot_u, knots_len = input
    device = params.device
    B, M = params.shape[:2]
    
    start_time = time.time()

    # 数据预处理
    preprocess_start = time.time()
    valid_mask = points_mask.bool()
    params = torch.where(valid_mask, params, torch.zeros_like(params))
    points = torch.where(valid_mask.unsqueeze(-1), points, torch.zeros_like(points))
    preprocess_time = time.time() - preprocess_start

    U = knot_u

    # 向量化参数计算
    param_calc_start = time.time()
    u = params.unsqueeze(2)  # [B, M, 1]
    U_expanded = U[:, p:].unsqueeze(1)  # [B, 1, K-p]
    diff = u - U_expanded  # [B, M, K-p]
    mask = diff > 1e-8
    
    indices = torch.arange(U.shape[1] - p, device=device).view(1, 1, -1)
    valid_indices = torch.where(mask, indices, torch.full_like(indices, U.shape[1]))
    uspan_uv = torch.argmin(valid_indices, dim=-1) + p
    u = u.squeeze(2)
    param_calc_time = time.time() - param_calc_start
    
    # 向量化基函数计算
    basis_start = time.time()
    Ni = torch.zeros(B, M, p+1, device=device)
    Ni[:, :, 0] = 1.0
    
    for k in range(1, p+1):
        saved = torch.zeros_like(u)
        new_Ni = torch.zeros(B, M, k+1, device=device)
        new_Ni[:, :, :k] = Ni[:, :, :k]
        
        for r in range(k):
            idx1 = torch.clamp(uspan_uv + r + 1, 0, U.shape[1]-1)
            idx2 = torch.clamp(uspan_uv + 1 - k + r, 0, U.shape[1]-1)
            
            UList1 = torch.gather(U, 1, idx1)
            UList2 = torch.gather(U, 1, idx2)
            
            dU = (UList1 - u) + (u - UList2)
            dU_safe = torch.where(torch.abs(dU) < 1e-8, 
                                torch.full_like(dU, 1e-4), dU)
            
            temp = new_Ni[:, :, r] / dU_safe
            temp = torch.where(torch.abs(dU) < 1e-8, 
                             torch.zeros_like(temp), temp)
            
            new_Ni[:, :, r] = saved + (UList1 - u) * temp
            saved = (u - UList2) * temp
        
        new_Ni[:, :, k] = saved
        Ni = new_Ni
    
    Nu_uv = Ni[:, :, :p+1]
    basis_time = time.time() - basis_start
    
    # 矩阵组装
    matrix_start = time.time()
    scatter_index = uspan_uv.unsqueeze(-1) + torch.arange(-p, 1, device=device).view(1, 1, -1)
    scatter_index = torch.clamp(scatter_index, 0, 29)
    
    N_all = torch.zeros(B, M, 30, device=device)
    N_all.scatter_(2, scatter_index, Nu_uv)
    
    # 应用mask
    N_mask = valid_mask.unsqueeze(-1).expand(-1, -1, 30)
    N_all = torch.where(N_mask, N_all, torch.zeros_like(N_all))
    matrix_time = time.time() - matrix_start

    # 高效求解
    solve_start = time.time()
    
    # 选择最优求解策略
    if M > 40:  # 超定系统，使用QR分解
        Q, R = torch.linalg.qr(N_all)
        Qt_points = torch.bmm(Q.transpose(-2, -1), points)
        solution = torch.linalg.solve_triangular(R, Qt_points, upper=True)
    elif M > 30:  # 中等规模，使用正规方程 + Cholesky
        NtN = torch.bmm(N_all.transpose(-2, -1), N_all)
        Ntp = torch.bmm(N_all.transpose(-2, -1), points)
        
        # 正则化
        reg = 1e-6 * torch.eye(30, device=device).unsqueeze(0).expand(B, -1, -1)
        NtN_reg = NtN + reg
        
        try:
            L = torch.linalg.cholesky(NtN_reg)
            y = torch.linalg.solve_triangular(L, Ntp, upper=False)
            solution = torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
        except:
            solution = torch.linalg.solve(NtN_reg, Ntp)
    else:  # 小规模，直接求解
        solution = torch.linalg.lstsq(N_all, points).solution
    
    pred_points = torch.bmm(N_all, solution)
    solve_time = time.time() - solve_start
    
    # 损失计算
    loss_calc_start = time.time()
    residuals = points - pred_points
    residuals = torch.where(valid_mask.unsqueeze(-1), residuals, torch.zeros_like(residuals))
    
    loss = torch.linalg.norm(residuals, dim=-1)
    
    if reduce:
        loss1 = loss.max(dim=-1)[0].sum()
        loss2 = (loss.sum(dim=-1) / valid_mask.sum(dim=-1)).sum()
    else:
        loss1 = loss.max(dim=-1)[0]
        loss2 = loss.sum(dim=-1) / valid_mask.sum(dim=-1)
    
    loss_calc_time = time.time() - loss_calc_start
    
    total_time = time.time() - start_time
    
    print(f"优化版NURBS计算时间统计:")
    print(f"  预处理时间: {preprocess_time*1000:.2f}ms")
    print(f"  参数计算时间: {param_calc_time*1000:.2f}ms") 
    print(f"  基函数计算时间: {basis_time*1000:.2f}ms")
    print(f"  矩阵组装时间: {matrix_time*1000:.2f}ms")
    print(f"  线性求解时间: {solve_time*1000:.2f}ms")
    print(f"  损失计算时间: {loss_calc_time*1000:.2f}ms")
    print(f"  总计算时间: {total_time*1000:.2f}ms")
    print("-" * 40)

    return loss1, loss2, solution


def getCtrlPts2(p,input,reduce=True):
    # input will be of dimension (batch_size, m+1, n+1, dimension)
    # 1 batch
    start_time = time.time()
    
    params,points,points_mask,knot_u,knots_len= input
    device=params.device

    # 数据预处理时间
    preprocess_start = time.time()
    params=torch.masked_fill(params,~points_mask.bool(),0)
    points=torch.masked_fill(points,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)
    preprocess_time = time.time() - preprocess_start

    U=knot_u
    # U_c = torch.cumsum(torch.where(knot_u<0.0, knot_u*0+1e-4, knot_u), dim=1)
    # U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

    if torch.isnan(U).any():
        # print(U_c)
        print(knot_u)

    # 参数计算时间
    param_calc_start = time.time()
    u = params.unsqueeze(1)
    uspan_uv = torch.stack([torch.min(torch.where((u[s] - U[s,p:].unsqueeze(1))>1e-8, u[s] - U[s,p:].unsqueeze(1), (u[s] - U[s,p:].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+p for s in range(U.size(0))])

    # torch._assert(torch.all(uspan_uv<25),"uspan out of bound")
    # print('uspan_uv:',uspan_uv)
    # print((u[0] - U[0,p:].unsqueeze(1))[...,35])

    u = u.squeeze(1)
    param_calc_time = time.time() - param_calc_start
    
    # 基函数计算时间
    basis_start = time.time()
    Ni = [u*0 for i in range(p+1)]
    Ni[0] = u*0 + 1
    for k in range(1,p+1):
        saved = (u)*0.0
        for r in range(k):
            UList1 = torch.stack([U[s,uspan_uv[s,:] + r + 1] for s in range(U.size(0))])
            UList2 = torch.stack([U[s,uspan_uv[s,:] + 1 - k + r] for s in range(U.size(0))])

            dU=((UList1 - u) + (u - UList2))
            dU_=torch.where(dU==0.0, u*0+1e-4, dU)

            temp = Ni[r]/dU_
            temp = torch.where(dU==0.0, u*0+1e-4, temp)
            Ni[r] = saved + (UList1 - u)*temp
            saved = (u - UList2)*temp
        Ni[k] = saved

    Nu_uv = torch.stack(Ni).permute(1,2,0)
    basis_time = time.time() - basis_start
    
    # 矩阵组装时间
    matrix_start = time.time()
    # torch.linalg.lstsq()
    scatter_index=torch.arange(-p,1,device=device).unsqueeze(0).unsqueeze(0).expand(uspan_uv.shape[0],uspan_uv.shape[1],-1)
    scatter_index=uspan_uv.unsqueeze(-1)+scatter_index

    N_all=torch.zeros((Nu_uv.size(0),Nu_uv.size(1),30),device=device)
    # print(143,scatter_index.dtype,Nu_uv.dtype,N_all.dtype)
    N_all=N_all.scatter_(2,scatter_index,Nu_uv)
    N_mask=points_mask.unsqueeze(-1).expand(-1,-1,30)

    N_all=torch.masked_fill(N_all,~N_mask.bool(),0)
    matrix_time = time.time() - matrix_start

    # 求解时间
    solve_start = time.time()
    solution=torch.linalg.pinv(N_all)@points
    pred_points=N_all@solution
    solve_time = time.time() - solve_start
    
    # 损失计算时间
    loss_calc_start = time.time()
    residuals=points-pred_points

    residuals=torch.masked_fill(residuals,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)

    # loss=torch.sum(torch.square(residuals))
    # loss=torch.linalg.norm(residuals)
    # loss=torch.abs(residuals).sum()
    # loss=torch.pow(residuals,2).max(dim=-1)[0].sum()
    loss=torch.linalg.norm(residuals,dim=-1)

    h_loss=Hausdorff_distance_batch_mask(pred_points,points,points_mask,points_mask)

    if reduce:
        loss1=loss.max(dim=-1)[0].sum()
        loss2=(loss.sum(dim=-1)/points_mask.sum(dim=-1)).sum()
        # loss2=loss.mean(dim=-1).sum()
        # loss=torch.linalg.norm(residuals,dim=-1).max(dim=-1)[0].sum()
        h_loss=h_loss.sum()
    else:
        loss1=loss.max(dim=-1)[0]
        # loss2=loss.mean(dim=-1)
        loss2=(loss.sum(dim=-1)/points_mask.sum(dim=-1))
    loss_calc_time = time.time() - loss_calc_start


    #### pinv method end #####
    
    total_time = time.time() - start_time
    
    # 输出时间统计
    print(f"NURBS2计算时间统计:")
    print(f"  预处理时间: {preprocess_time*1000:.2f}ms")
    print(f"  参数计算时间: {param_calc_time*1000:.2f}ms") 
    print(f"  基函数计算时间: {basis_time*1000:.2f}ms")
    print(f"  矩阵组装时间: {matrix_time*1000:.2f}ms")
    print(f"  线性求解时间: {solve_time*1000:.2f}ms")
    print(f"  损失计算时间: {loss_calc_time*1000:.2f}ms")
    print(f"  总计算时间: {total_time*1000:.2f}ms")
    print("-" * 40)

    return loss1,loss2,h_loss,solution
    # return torch.functional.F.mse_loss((N_all@solution),points)

def Hausdorff_distance_batch(a : torch.Tensor, b : torch.Tensor):
    """
    Params:
        a: (bs, sz1, 2), only bs = 1 used ? 
        b: (bs, sz2, 2)
    Returns:
        hdis: a scale, as bs = 1 is set 
    """
    # print("\nstart")
    # print(a[..., :10])
    # print(b[..., :10])
    # print("end\n")
    # assert a.shape == b.shape
    bs = a.size(0)
    sz1 = a.size(-2)
    sz2 = b.size(-2)

    expand_a = a.unsqueeze(dim=-2).expand(-1, -1, sz2,-1)
    expand_b = b.unsqueeze(dim=-3).expand(-1, sz1, -1, -1)
    delta = torch.linalg.norm(expand_a - expand_b, dim=-1)
    ret1 = delta.min(dim=-2)[0].max(dim=-1)[0]
    ret2 = delta.min(dim=-1)[0].max(dim=-1)[0]
    hdis = torch.where(ret1 > ret2, ret1, ret2)
    return hdis

def Hausdorff_distance_batch_mask(a : torch.Tensor, b : torch.Tensor,a_mask:torch.Tensor,b_mask:torch.Tensor):
    """
    Params:
        a: (bs, sz1, 2), only bs = 1 used ? 
        b: (bs, sz2, 2)
    Returns:
        hdis: a scale, as bs = 1 is set 
    """
    # print("\nstart")
    # print(a[..., :10])
    # print(b[..., :10])
    # print("end\n")
    # assert a.shape == b.shape
    bs = a.size(0)
    sz1 = a.size(-2)
    sz2 = b.size(-2)

    expand_a = a.unsqueeze(dim=-2).expand(-1, -1, sz2,-1)
    expand_b = b.unsqueeze(dim=-3).expand(-1, sz1, -1, -1)

    expand_a_mask=a_mask.unsqueeze(dim=-1).expand(-1,-1,sz2)
    expand_b_mask=b_mask.unsqueeze(dim=-2).expand(-1,sz1,-1)

    delta = torch.linalg.norm(expand_a - expand_b, dim=-1)
    delta_mask=torch.logical_and(expand_a_mask,expand_b_mask)
    delta=torch.masked_fill(delta,torch.logical_not(delta_mask),torch.finfo(torch.float32).max)
    ret1 = torch.masked_fill(delta.min(dim=-2)[0],torch.logical_not(b_mask),0).max(dim=-1)[0]
    ret2 = torch.masked_fill(delta.min(dim=-1)[0],torch.logical_not(a_mask),0).max(dim=-1)[0]
    hdis = torch.where(ret1 > ret2, ret1, ret2)
    return hdis
