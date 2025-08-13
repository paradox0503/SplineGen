import os
import datetime
import re
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader,random_split

from torch.utils.tensorboard import SummaryWriter
from dataset.curveDataset import CurveDataset
import train.getModel as getModel
from util import AverageMeter,masked_accuracy

TOKENS = {
  '<eos>': 0
}

def train(ifsave,data_path,model_save_path,log_path,knot_model_load_path,train_weights=[0.1,0.9],use_cuda=True,n_epochs=500,batch_size=512,lr=1e-5,save_epoch=5,n_workers = 4,resume_from=None):
    print('epoch:',n_epochs,'base_batch_size',batch_size)
    torch.random.manual_seed(231)

    use_cuda = True
    # num_gpus = torch.cuda.device_count() if use_cuda else 0
    num_gpus = 1
    print(f"Found {num_gpus} available GPUs. Using {'multi-GPU' if num_gpus > 1 else 'single-GPU'} training.")
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    if num_gpus == 0 and use_cuda:
        print("Warning: No GPU available, falling back to CPU.")

    log_path=log_path
    model_save_path = model_save_path

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if not os.path.exists(log_path):
      os.makedirs(log_path)
    writer = SummaryWriter(log_dir=log_path)
    model_save_path=model_save_path
    batch_size = batch_size * num_gpus
    
    dataset=CurveDataset(data_path,use_points_params=True,use_knots=True,use_orders=True, random_select_rate=None)
    
    input_dim=dataset.dimension

    train_dataset,val_dataset=random_split(dataset=dataset,lengths=(0.8,0.2))

    print(f'# train: {len(train_dataset):7d}')
    print(f'# val:   {len(val_dataset):7d}')


    train_loader = DataLoader(train_dataset, batch_size=batch_size,
      num_workers=n_workers,shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
      num_workers=n_workers,shuffle=False,pin_memory=True)

    model=getModel.getModel_SimpleEncoder_Knots(device=device,knot_load_path=knot_model_load_path,input_dim=input_dim)
    # model = getModel.getModel_SimpleEncoder_Knots(
    #     device=device if num_gpus > 0 else 'cpu',  # 先加载到主GPU
    #     knot_load_path=knot_model_load_path,
    #     input_dim=input_dim
    # )
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        model = model.to(device)  # 移动到主GPU
        print(f"Model wrapped with DataParallel, using {num_gpus} GPUs")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 添加断点续训逻辑
    start_epoch = 0
    if resume_from is not None:
        # 如果 resume_from 不是绝对路径，则在 model_save_path 目录中查找
        if not os.path.isabs(resume_from):
            resume_path = os.path.join(os.path.dirname(model_save_path), resume_from)
        else:
            resume_path = resume_from
        
        print(f"Looking for checkpoint at: {resume_path}")
        print(f"Model directory: {os.path.dirname(model_save_path)}")
        print(f"Resume from: {resume_from}")
            
        if os.path.exists(resume_path):
            print(f"Loading checkpoint from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # 加载模型状态
            if num_gpus > 1:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载训练轮次
            start_epoch = checkpoint['epoch']
            
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"Checkpoint file {resume_path} not found. Starting from scratch.")

    def params_loss_fn(params_label,params,params_mask):
        loss = torch.nn.functional.mse_loss(params, params_label, reduction='none')
        
        loss = loss.masked_fill(params_mask == 0, 0)
        loss = loss.sum(dim=-1)
        valid_len = params_mask.sum(dim = -1)

        loss = loss/valid_len
        loss= torch.mean(loss)

        return loss
    
    def criterion(score,label,params,params_label,params_mask,w):
        # 由于不再需要排序，order loss设为0
        # loss1 = torch.tensor(0.0, device=params.device)  # 虚拟的排序损失
        loss2 = params_loss_fn(params,params_label,params_mask)

        # return loss1,loss2,w[1]*loss2  # 只使用参数损失
        return loss2

    train_loss = AverageMeter()
    train_loss_order = AverageMeter()
    train_loss_param = AverageMeter()
    train_accuracy = AverageMeter()
    val_loss = AverageMeter()
    val_loss_order=AverageMeter()
    val_loss_param=AverageMeter()
    val_accuracy = AverageMeter()

# begin training
    for epoch in range(start_epoch, n_epochs):
        model.train()
        print(f'Epoch {epoch} training...')
        for bat, input in enumerate(tqdm(train_loader)):
        # for bat, (batch_data, batch_labels, batch_lengths,batch_mask,batch_params,batch_points) in enumerate(tqdm(train_loader)):
            batch_labels = input['targets'].to(device)
            batch_lengths = input['length'].to(device)
            batch_points_mask=input['points_mask'].to(device)
            batch_params_mask=input['params_mask'].to(device)
            batch_params=input['params'].to(device=device,dtype=torch.float32)
            batch_points=input['points'].to(device)
            batch_knots=input['knots_expanded'].to(device)
            batch_knots_mask=input['knots_mask_expanded'].to(device)

            optimizer.zero_grad()
            # knots,knots_mask,log_pointer_scores, pointer_argmaxs,params = model(
            #     batch_points,batch_mask, batch_lengths,
            #     batch_labels,batch_knots[:,:-1],batch_knots_mask[:,:-1],half_eval=True
            # )
            knots,knots_mask,log_pointer_scores, pointer_argmaxs,params = model(
                batch_points,batch_params,batch_params_mask,batch_points_mask, batch_lengths,
                batch_labels,batch_knots[:,:-1],batch_knots_mask[:,:-1]
            )

            loss = criterion(
               log_pointer_scores.view(-1, log_pointer_scores.shape[-1]),
               batch_labels.reshape(-1),
               params=params,params_label=batch_params,params_mask=batch_params_mask,w=train_weights
               )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss.update(loss.item(), batch_knots.size(0))
            print(f'Epoch {epoch}: train\tLoss: {train_loss.avg:.6f}')
            # 由于不再需要排序，设置虚拟的准确率为1.0
            # train_accuracy.update(1.0, batch_params_mask.int().sum().item())
            # train_loss_order.update(order_loss.item(), batch_knots.size(0))
            # train_loss_param.update(param_loss.item(), batch_params_mask.int().sum().item())
            # train_loss_knots.update(knots_loss.item(), batch_knots_mask[:,1:].int().sum().item())

        # if bat % log_interval == 0:
        #   print(f'Epoch {epoch}: '
        #         f'Train [{bat * len(batch_data):9d}/{len(train_dataset):9d} '
        #         f'Loss: {train_loss.avg:.6f}\tAccuracy: {train_accuracy.avg:3.4%}')
                  
        writer.add_scalar('Loss/train',train_loss.avg,epoch)
        # writer.add_scalar('Order Loss/train',train_loss_order.avg,epoch)
        # writer.add_scalar('Param Loss/train',train_loss_param.avg,epoch)
        # writer.add_scalar('Accuracy/train',train_accuracy.avg,epoch)

        print(f'Epoch {epoch} validating...')
        model.eval()

        with torch.no_grad():
          for bat, input in enumerate(tqdm(val_loader)):
              batch_labels = input['targets'].to(device)
              batch_lengths = input['length'].to(device)
              batch_points_mask=input['points_mask'].to(device)
              batch_params_mask=input['params_mask'].to(device)
              batch_params=input['params'].to(device=device,dtype=torch.float32)
              batch_points=input['points'].to(device)
              batch_knots=input['knots_expanded'].to(device)
              batch_knots_mask=input['knots_mask_expanded'].to(device)

              knots,knots_mask,log_pointer_scores, pointer_argmaxs,params = model(
                  batch_points,batch_params,batch_params_mask,batch_points_mask, batch_lengths,
                  batch_labels,batch_knots[:,:-1],batch_knots_mask[:,:-1]
              )

              loss = criterion(
                log_pointer_scores.view(-1, log_pointer_scores.shape[-1]),
                batch_labels.reshape(-1),
                params=params,params_label=batch_params,params_mask=batch_params_mask,w=train_weights)

              val_loss.update(loss.item(), batch_knots.size(0))
              # 由于不再需要排序，设置虚拟的准确率为1.0
            #   val_accuracy.update(1.0, batch_params_mask.int().sum().item())
            #   val_loss_order.update(order_loss.item(), batch_knots.size(0))
            #   val_loss_param.update(param_loss.item(), batch_params_mask.int().sum().item())

        # if bat % log_interval == 0:
        #   print(f'Epoch {epoch}: '
        #         f'Train [{bat * len(batch_data):9d}/{len(train_dataset):9d} '
        #         f'Loss: {train_loss.avg:.6f}\tAccuracy: {train_accuracy.avg:3.4%}')
                  
        writer.add_scalar('Loss/val',val_loss.avg,epoch)
        # writer.add_scalar('Order Loss/val',val_loss_order.avg,epoch)
        # writer.add_scalar('Param Loss/val',val_loss_param.avg,epoch)
        # writer.add_scalar('Accuracy/val',val_accuracy.avg,epoch)
        # writer.add_scalar('Accuracy/val',train_accuracy.avg,epoch)

        print(f'Epoch {epoch}: Val\tLoss: {val_loss.avg:.6f}')
        
        # 保存 checkpoint
        if (epoch + 1) % save_epoch == 0:
            writer.flush()
            # save model every save_epoch epochs
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss.avg,
                'val_loss': val_loss.avg,
                'train_loss_order': train_loss_order.avg,
                'train_loss_param': train_loss_param.avg,
                'val_loss_order': val_loss_order.avg,
                'val_loss_param': val_loss_param.avg,
                'train_accuracy': train_accuracy.avg,
                'val_accuracy': val_accuracy.avg,
            }
            torch.save(checkpoint, model_save_path+f'epoch_{epoch+1}'+'.pth')
            print(f"Checkpoint saved at epoch {epoch + 1}")
            
        if ifsave:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss.avg,
                'val_loss': val_loss.avg,
                'train_loss_order': train_loss_order.avg,
                'train_loss_param': train_loss_param.avg,
                'val_loss_order': val_loss_order.avg,
                'val_loss_param': val_loss_param.avg,
                'train_accuracy': train_accuracy.avg,
                'val_accuracy': val_accuracy.avg,
            }
            torch.save(checkpoint, model_save_path+f'epoch_{epoch+1}'+'.pth')
            print(f"Checkpoint saved at epoch {epoch + 1}")
        #         )
        train_loss.reset()
        train_accuracy.reset()
        train_loss_param.reset()
        train_loss_order.reset()
        val_loss.reset()
        val_accuracy.reset()
        val_loss_order.reset()
        val_loss_param.reset()
                  

if __name__=='__main__':
    train()