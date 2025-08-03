"""
铁路线元通用数学模型训练脚本
基于现有的编码器-解码器架构，使用新的铁路线元数学模型
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from util.statistic import AverageMeter
import torch
import models.encoder_decoder
import models.pointsEncoder
import models.kkn
from torch.utils.data import DataLoader
import re
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import random_split
import datetime
from dataset.curveDataset_for_encoder_railway import CurveDataset_for_encoder as RailwayDataset_for_encoder
from config.railway_config import MODEL_CONFIG, TRAINING_CONFIG, PATH_CONFIG, print_config


def train_railway_model(data_path=None, log_dir=None, model_save_dir=None, 
                       epochs=None, batch_size=None, ifsave=False, resume_from=None):
    """
    训练铁路线元模型
    
    Args:
        data_path: 数据路径，如果为None则使用配置文件中的路径
        log_dir: 日志目录
        model_save_dir: 模型保存目录
        epochs: 训练轮数
        batch_size: 批量大小
        ifsave: 是否保存模型
        resume_from: 断点续训路径
    """
    
    # 打印配置信息
    print_config()
    
    # 使用配置文件中的参数（如果参数未指定）
    epochs = epochs or TRAINING_CONFIG['epochs']
    batch_size = batch_size or TRAINING_CONFIG['batch_size']
    learning_rate = TRAINING_CONFIG['learning_rate']
    
    # GPU配置
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} available GPUs. Using multi-GPU training.")
    if num_gpus == 0:
        raise ValueError("No GPU available. Please check CUDA configuration.")
    
    # 数据集加载
    print("Loading Railway Element Dataset...")
    if data_path is None:
        # 如果没有指定数据路径，尝试生成新数据
        dataset = RailwayDataset_for_encoder(
            generate_new=True,
            num_samples=MODEL_CONFIG['railway_model']['num_train_samples'] + 
                       MODEL_CONFIG['railway_model']['num_val_samples'],
            max_points=MODEL_CONFIG['railway_model']['max_points']
        )
    else:
        dataset = RailwayDataset_for_encoder(data_path=data_path)
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Dataset dimension: {dataset.dimension}D")
    
    # 目录设置
    if log_dir is None:
        log_dir = os.path.join(PATH_CONFIG['log_dir'], 'train_railway_encoder')
    if model_save_dir is None:
        model_save_dir = os.path.join(PATH_CONFIG['model_dir'], 'train_railway_encoder', 'models')
    
    log_dir = log_dir + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = model_save_dir + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    
    # 创建目录
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # 数据分割
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # 数据加载器
    n_workers = TRAINING_CONFIG['num_workers']
    batch_size = batch_size * num_gpus
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=TRAINING_CONFIG['shuffle'],
        num_workers=n_workers, 
        pin_memory=TRAINING_CONFIG['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers, 
        pin_memory=TRAINING_CONFIG['pin_memory']
    )
    
    # 模型配置
    device = 'cuda'
    KNOT_LOSS_WEIGHT = 0.3
    input_dim = dataset.dimension
    
    # 模型超参数（适配铁路线元模型）
    d_model = 512
    nhead = 4
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 2048
    dropout = 0.05
    
    print("Building Railway Element Model...")
    
    # 点编码器
    points_encoder = models.pointsEncoder.PointsEncoder(
        input_dim, 
        hidden_dim=d_model, 
        num_layers=num_encoder_layers, 
        num_head=nhead, 
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
    # 解码器1 - 线元参数解码器
    decoder1 = models.kkn.KPN(
        d_model,
        num_decoder_layers=num_decoder_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
    # 解码器2 - 几何参数解码器
    decoder2 = models.kkn.KPN(
        d_model,
        num_decoder_layers=num_decoder_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
    # 编码器-解码器模型
    model = models.encoder_decoder.PointsEncoderDecoder2(
        points_encoder, decoder1, decoder2, masking_rate=None
    ).to(device)
    
    # 多GPU包装
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {num_gpus} GPUs for training.")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 断点续训逻辑
    start_epoch = 0
    if resume_from is not None:
        if not os.path.isabs(resume_from):
            resume_path = os.path.join(model_dir, resume_from)
        else:
            resume_path = resume_from
        
        print(f"Looking for checkpoint at: {resume_path}")
        
        if os.path.exists(resume_path):
            print(f"Loading checkpoint from {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            
            if num_gpus > 1:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"Checkpoint file {resume_path} not found. Starting from scratch.")
    
    # 损失函数
    def loss_fn(knots_pred, knots, knots_mask):
        loss = torch.nn.functional.mse_loss(knots_pred, knots, reduction='none')
        loss = loss.sum(dim=-1)
        loss = loss.masked_fill(knots_mask == 0, 0)
        loss = loss.sum(dim=-1)
        knots_len = knots_mask.sum(dim=-1)
        
        eps = 1e-8
        knots_len = torch.clamp(knots_len, min=eps)
        loss = loss / knots_len
        return torch.mean(loss)
    
    criterion = loss_fn
    
    # TensorBoard记录器
    writer = SummaryWriter(log_dir=log_dir)
    
    # 训练步骤函数
    def train_step(model, batch, loss_avg: AverageMeter,
                   loss1_avg: AverageMeter, loss2_avg: AverageMeter,
                   criterion, train=True):
        batch_points = batch['points'].to(device)
        batch_points_mask = batch['points_mask'].to(device)
        batch_knots = batch['knots_expanded'].to(device)
        batch_knots_mask = batch['knots_mask_expanded'].to(device)
        batch_params = batch['params_expanded'].to(device)
        batch_params_mask = batch['params_mask_expanded'].to(device)
        
        with torch.cuda.amp.autocast():
            output1, output2 = model(
                batch_points, batch_points_mask,
                batch_knots[:, :-1], batch_knots_mask[:, :-1],
                batch_params[:, :-1], batch_params_mask[:, :-1]
            )
            
            loss_knot = criterion(output1[0], batch_knots[:, 1:], batch_knots_mask[:, :-1])
            loss_params = criterion(output2[0], batch_params[:, 1:], batch_params_mask[:, :-1])
            loss = loss_knot * KNOT_LOSS_WEIGHT + loss_params
        
        loss1_avg.update(loss_knot.item() * batch_points.size(0), batch_points.size(0))
        loss2_avg.update(loss_params.item() * batch_points.size(0), batch_points.size(0))
        loss_avg.update(loss.item() * batch_points.size(0), batch_points.size(0))
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # 模型权重检查函数
    def check_model_weights(model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Warning: Found NaN/Inf in parameter {name}")
                return False
        return True
    
    print("Starting Railway Element Model Training...")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    
    # 训练循环
    for epoch in range(start_epoch, epochs):
        
        # 检查模型权重
        if not check_model_weights(model):
            print(f"Model weights contain NaN/Inf at epoch {epoch}. Stopping training.")
            break
        
        # 初始化记录器
        train_loss1_recorder = AverageMeter()
        val_loss1_recorder = AverageMeter()
        train_loss2_recorder = AverageMeter()
        val_loss2_recorder = AverageMeter()
        train_loss_recorder = AverageMeter()
        val_loss_recorder = AverageMeter()
        
        # 训练阶段
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            train_step(model, batch, train_loss_recorder,
                      train_loss1_recorder, train_loss2_recorder,
                      criterion, train=True)
        
        train_loss = train_loss_recorder.acc()
        train_loss1 = train_loss1_recorder.acc()
        train_loss2 = train_loss2_recorder.acc()
        
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.6f}")
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss1: {train_loss1:.6f}")
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss2: {train_loss2:.6f}")
        
        writer.add_scalar('Training/Total/Loss', train_loss, epoch)
        writer.add_scalar('Training/Element/Loss', train_loss1, epoch)
        writer.add_scalar('Training/Geom/Loss', train_loss2, epoch)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                train_step(model, batch, val_loss_recorder,
                          val_loss1_recorder, val_loss2_recorder,
                          criterion, train=False)
        
        val_loss = val_loss_recorder.acc()
        val_loss1 = val_loss1_recorder.acc()
        val_loss2 = val_loss2_recorder.acc()
        
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.6f}")
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss1: {val_loss1:.6f}")
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss2: {val_loss2:.6f}")
        
        writer.add_scalar('Validation/Total/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Element/Loss', val_loss1, epoch)
        writer.add_scalar('Validation/Geom/Loss', val_loss2, epoch)
        
        # 保存检查点
        if ifsave and (epoch + 1) % TRAINING_CONFIG['save_every'] == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': MODEL_CONFIG
            }
            torch.save(checkpoint, f'{model_dir}railway_epoch_{epoch + 1}.pth')
            print(f"Railway model checkpoint saved at epoch {epoch + 1}")
    
    print("Railway Element Model Training completed.")
    writer.close()
    
    return model, model_dir


def find_latest_checkpoint(model_dir):
    """查找最新的checkpoint文件"""
    if not os.path.exists(model_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(model_dir):
        if file.startswith('railway_epoch_') and file.endswith('.pth'):
            match = re.search(r'railway_epoch_(\d+)\.pth', file)
            if match:
                epoch_num = int(match.group(1))
                checkpoint_files.append((epoch_num, os.path.join(model_dir, file)))
    
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: x[0])
        return latest_checkpoint[1]
    
    return None


if __name__ == "__main__":
    print("Railway Element Universal Mathematical Model Training")
    print("="*60)
    
    # 训练铁路线元模型
    model, save_dir = train_railway_model(
        epochs=100,  # 减少轮数用于测试
        batch_size=256,  # 减少批量大小
        ifsave=True
    )
    
    print(f"Model saved to: {save_dir}")
    print("Training completed successfully!")
