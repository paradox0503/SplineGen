import torch

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
    params,points,points_mask,knot_u,knots_len= input
    device=params.device

    params=torch.masked_fill(params,~points_mask.bool(),0)
    points=torch.masked_fill(points,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)

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

    N_all=torch.zeros((Nu_uv.size(0),Nu_uv.size(1),30),device=device)
    # print(143,scatter_index.dtype,Nu_uv.dtype,N_all.dtype)
    N_all=N_all.scatter_(2,scatter_index,Nu_uv)
    N_mask=points_mask.unsqueeze(-1).expand(-1,-1,30)

    N_all=torch.masked_fill(N_all,~N_mask.bool(),0)
        # N_all[s,uspan_uv[s,:]]
    solution=torch.linalg.pinv(N_all)@points
    residuals=points-N_all@solution

    residuals=torch.masked_fill(residuals,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)

    loss=torch.linalg.norm(residuals,dim=-1).max(dim=-1)[0].sum()
    # loss=torch.linalg.norm(residuals,dim=-1).max(dim=-1)[0].sum()

    #### pinv method end #####

    return loss,solution
    # return torch.functional.F.mse_loss((N_all@solution),points)

import torch

def getCtrlPts3(p, input):
    """
    重写后的 B样条控制点求解函数（向量化，GPU优化）
    
    参数:
        p: 曲线阶数（int）
        input: tuple，包含
            params: (batch, seq_len)
            points: (batch, seq_len, dim)
            points_mask: (batch, seq_len)
            knot_u: (batch, knot_len)
            knots_len: (batch,)
    返回:
        loss: 标量，残差范数总和
        solution: (batch, 30, dim)，控制点解
    """
    params, points, points_mask, knot_u, knots_len = input
    device = params.device

    # 掩码填充
    params = torch.masked_fill(params, ~points_mask.bool(), 0)
    points = torch.masked_fill(points, ~points_mask.unsqueeze(-1).bool(), 0)

    U = knot_u  # (batch, knot_len)

    u = params  # (batch, seq_len)

    # 1. 向量化计算 uspan_uv （每个 u 对应的区间）
    # U区间是从p开始的，因此只看 U[:, p:] 部分
    U_p = U[:, p:]  # (batch, knot_len-p)
    # 计算 u_i 在 U_j 和 U_{j+1} 之间的索引 j
    # mask shape: (batch, seq_len, knot_len-p)
    mask = (u.unsqueeze(-1) >= U_p.unsqueeze(1))  # True表示 u >= U_j

    # uspan_uv = 每个 u 在 U 中最大满足 u>=U_j 的 j 的索引（相对于 U[:, p:]）
    uspan_uv = mask.sum(dim=-1) - 1 + p  # (batch, seq_len), -1是因为索引从0开始

    # 2. 计算基函数Ni，大小 (batch, seq_len, p+1)
    batch_size, seq_len = u.shape
    dim = points.shape[-1]

    Ni = torch.zeros(batch_size, seq_len, p + 1, device=device, dtype=u.dtype)
    Ni[..., 0] = 1.0

    for k in range(1, p + 1):
        left = u - torch.gather(U, 1, (uspan_uv - k).clamp(min=0))  # (batch, seq_len)
        right = torch.gather(U, 1, (uspan_uv + 1 - k).clamp(max=U.shape[1] - 1)) - u  # (batch, seq_len)

        left_div = left / torch.clamp(left + right, min=1e-8)
        right_div = right / torch.clamp(left + right, min=1e-8)

        Ni[..., k] = left_div * Ni[..., k - 1] + right_div * Ni[..., k - 1]

    # 3. 构造N_all矩阵，shape (batch, seq_len, 30)
    max_ctrl_pts = 30
    N_all = torch.zeros(batch_size, seq_len, max_ctrl_pts, device=device, dtype=u.dtype)

    # scatter基函数到对应的控制点索引
    # scatter indices shape (batch, seq_len, p+1)
    scatter_indices = uspan_uv.unsqueeze(-1) + torch.arange(-p, 1, device=device)
    scatter_indices = scatter_indices.clamp(min=0, max=max_ctrl_pts - 1)

    # scatter操作
    N_all.scatter_(-1, scatter_indices, Ni)

    # 4. 掩码处理
    N_all = torch.masked_fill(N_all, ~points_mask.unsqueeze(-1).bool(), 0)

    # 5. 解线性最小二乘问题 N_all @ solution = points
    # 使用 torch.linalg.lstsq 替代 pinv
    solution = torch.linalg.lstsq(N_all, points).solution  # shape (batch, 30, dim)

    # 6. 计算残差
    residuals = points - torch.bmm(N_all, solution)
    residuals = torch.masked_fill(residuals, ~points_mask.unsqueeze(-1).bool(), 0)

    # 7. 计算 loss: 选用最大范数+求和
    loss = torch.linalg.norm(residuals, dim=-1).max(dim=-1)[0].sum()

    return loss, solution


def getCtrlPts2(p,input,reduce=True):
    # input will be of dimension (batch_size, m+1, n+1, dimension)
    # 1 batch
    params,points,points_mask,knot_u,knots_len= input
    device=params.device

    params=torch.masked_fill(params,~points_mask.bool(),0)
    points=torch.masked_fill(points,~(points_mask.unsqueeze(-1).expand(-1,-1,points.shape[-1])).bool(),0)

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

    N_all=torch.zeros((Nu_uv.size(0),Nu_uv.size(1),30),device=device)
    # print(143,scatter_index.dtype,Nu_uv.dtype,N_all.dtype)
    N_all=N_all.scatter_(2,scatter_index,Nu_uv)
    N_mask=points_mask.unsqueeze(-1).expand(-1,-1,30)

    N_all=torch.masked_fill(N_all,~N_mask.bool(),0)

    solution=torch.linalg.pinv(N_all)@points
    pred_points=N_all@solution
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


    #### pinv method end #####

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
