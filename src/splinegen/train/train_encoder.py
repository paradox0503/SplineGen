from util.statistic import AverageMeter
import torch
import models.encoder_decoder
import models.pointsEncoder
import models.kkn
from torch.utils.data import DataLoader
import os
import re
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot
import datetime
from dataset.curveDataset_for_encoder import CurveDataset_for_encoder
from torch.utils.data import random_split

def train(data_path,log_dir,model_save_dir,epochs,base_batch_size,ifsave,resume_from=None):
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} available GPUs. Using multi-GPU training.")
    if num_gpus == 0:
        raise ValueError("No GPU available. Please check CUDA configuration.")
    
    dataset = CurveDataset_for_encoder(
        data_path=data_path)
    log_dir=log_dir
    model_dir = model_save_dir
    # if 'h100' in torch.cuda.get_device_name(0).lower():
    #     n_workers = max(8, os.cpu_count() // 2)  # H100可以使用更多worker
    # else:
    #     n_workers = max(8, os.cpu_count() // 2)  # 4090适当减少
    # print("--------------------------------",torch.cuda.get_device_name(0).lower())
    n_workers = 4
                                         
    device='cuda'
    KNOT_LOSS_WEIGHT=0.3
    p = 3
    d_model = 512  # Embedding dimension
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 2048
    dropout = 0.05
    learning_rate = 0.0001 #This is very good: learning_rate = 0.0001
    # learning_rate = 0.00001 # have a test
    batch_size = base_batch_size * num_gpus

    # Create DataLoader for training data
    print('# Create DataLoader for training data')

    train_dataset,val_dataset=random_split(dataset,[int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)],generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=True,num_workers=n_workers, pin_memory=True)

    input_dim=dataset.dimension

    points_encoder=models.pointsEncoder.PointsEncoder(
         input_dim, 
         hidden_dim=d_model, 
         num_layers=num_encoder_layers, 
         num_head=nhead, 
         dim_feedforward=dim_feedforward,
         dropout=dropout)

    decoder1=models.kkn.KPN(
         d_model,
         num_decoder_layers=num_decoder_layers,
         nhead=nhead,
         dim_feedforward=dim_feedforward,
         dropout=dropout)
         
    decoder2=models.kkn.KPN(
         d_model,
         num_decoder_layers=num_decoder_layers,
         nhead=nhead,
         dim_feedforward=dim_feedforward,
         dropout=dropout)

    model=models.encoder_decoder.PointsEncoderDecoder2(points_encoder,decoder1,decoder2,masking_rate=None).to(device)

    # 多GPU包装：自动将模型分发到所有可用GPU
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {num_gpus} GPUs for training.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # 添加断点续训逻辑
    start_epoch = 0
    if resume_from is not None:
        # 如果 resume_from 不是绝对路径，则在 model_dir 中查找
        if not os.path.isabs(resume_from):
            resume_path = os.path.join(model_dir, resume_from)
        else:
            resume_path = resume_from
        
        print(f"Looking for checkpoint at: {resume_path}")
        print(f"Model directory: {model_dir}")
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
    
    def loss_fn(knots_pred, knots, knots_mask):
        loss = torch.nn.functional.mse_loss(knots_pred, knots, reduction='none')
        
        loss = loss.sum(dim=-1)
        loss = loss.masked_fill(knots_mask == 0, 0)
        loss = loss.sum(dim=-1)
        knots_len = knots_mask.sum(dim = -1)

        loss = loss/knots_len
        return torch.mean(loss)
    criterion = loss_fn  # Mean Squared Error Loss

    # Define the directory where the models are saved
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Get a list of all existing model files

    # TensorBoard setup
    torch.backends.cuda.matmul.allow_tf32 = True
    writer = SummaryWriter(log_dir=log_dir)
    scaler = torch.cuda.amp.GradScaler()
    def train_step(model,
                   batch,loss_avg:AverageMeter,
                   loss1_avg:AverageMeter,
                   loss2_avg:AverageMeter,
                   criterion,train=True):
        batch_points = batch['points'].to(device)
        batch_points_mask = batch['points_mask'].to(device)
        batch_knots = batch['knots_expanded'].to(device)
        batch_knots_mask = batch['knots_mask_expanded'].to(device)
        batch_params = batch['params_expanded'].to(device)
        batch_params_mask = batch['params_mask_expanded'].to(device)

        with torch.cuda.amp.autocast():
            output1,output2=model(
                batch_points,batch_points_mask,
                batch_knots[:,:-1],batch_knots_mask[:,:-1],
                batch_params[:,:-1],batch_params_mask[:,:-1])

            loss_knot=criterion(output1[0],batch_knots[:,1:],batch_knots_mask[:,:-1])
            loss_params=criterion(output2[0],batch_params[:,1:], batch_params_mask[:,:-1])
            # loss = criterion(train_output, batch_knots_left_shifted)
            loss=loss_knot*KNOT_LOSS_WEIGHT+loss_params

        loss1_avg.update(loss_knot.item()*batch_points.size(0),batch_points.size(0))
        loss2_avg.update(loss_params.item()*batch_points.size(0),batch_points.size(0))
        loss_avg.update(loss.item()*batch_points.size(0),batch_points.size(0))

        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # 缩放梯度并反向传播
            scaler.step(optimizer)         # 更新优化器
            scaler.update()                # 更新缩放器

    # Training loop
    for epoch in range(start_epoch,epochs):

        train_loss1_recoder=AverageMeter()
        val_loss1_recoder=AverageMeter()

        train_loss2_recoder=AverageMeter()
        val_loss2_recoder=AverageMeter()
        
        train_loss_recoder=AverageMeter()
        val_loss_recoder=AverageMeter()

        model.train()
        train_loss_total = 0
        for batch in tqdm(train_loader):
            train_step(model,batch,train_loss_recoder,
                       train_loss1_recoder,
                       train_loss2_recoder,
                       criterion,train=True)

        train_loss=train_loss_recoder.acc()
        train_loss1=train_loss1_recoder.acc()
        train_loss2=train_loss2_recoder.acc()
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}")
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss1: {train_loss1}")
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss2: {train_loss2}")
        writer.add_scalar('Training/Total/Loss', train_loss, epoch)
        writer.add_scalar('Training/Knot/Loss', train_loss1, epoch)
        writer.add_scalar('Training/Param/Loss', train_loss2, epoch)

        # Validation
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch in tqdm(val_loader):
                    train_step(model,batch,val_loss_recoder,
                            val_loss1_recoder,
                            val_loss2_recoder,
                            criterion,train=False)

        val_loss=val_loss_recoder.acc()
        val_loss1=val_loss1_recoder.acc()
        val_loss2=val_loss2_recoder.acc()
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}")
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss1: {val_loss1}")
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss2: {val_loss2}")
        writer.add_scalar('Validation/Total/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Knot/Loss', val_loss1, epoch)
        writer.add_scalar('Validation/Param/Loss', val_loss2, epoch)
        
        # 保存 checkpoint
        if ifsave:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, f'{model_dir}/checkpoint_epoch_{epoch + 1}.pth')
            print(f"Checkpoint saved at epoch {epoch + 1}")
            
        # Save model every 10 epochs (额外保存)
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, f'{model_dir}/checkpoint_epoch_{epoch + 1}.pth')
            print(f"Checkpoint saved at epoch {epoch + 1}")
    

    print("Training complete.")
    writer.close()

def find_latest_checkpoint(model_dir):
    """查找最新的checkpoint文件"""
    if not os.path.exists(model_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(model_dir):
        if file.startswith('checkpoint_epoch_') and file.endswith('.pth'):
            # 提取epoch数字
            match = re.search(r'checkpoint_epoch_(\d+)\.pth', file)
            if match:
                epoch_num = int(match.group(1))
                checkpoint_files.append((epoch_num, os.path.join(model_dir, file)))
    
    if checkpoint_files:
        # 返回最新的checkpoint文件路径
        latest_checkpoint = max(checkpoint_files, key=lambda x: x[0])
        return latest_checkpoint[1]
    
    return None