from util.statistic import AverageMeter
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import re
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.nn.functional import one_hot
import datetime
from dataset.curveDataset import CurveDataset
from torch.utils.data import random_split
import train.getModel as getModel

# if __name__ == '__main__':
# 
def train(data_path,log_path,encoder_path,knot_model_save_dir,epochs=1000,base_batch_size=512,ifsave=False,resume_from=None):
    print('epoch:',epochs,'base_batch_size',base_batch_size)

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} available GPUs. Using multi-GPU training.")
    if num_gpus == 0:
        raise ValueError("No GPU available. Please check CUDA configuration.")
    # n_workers = max(8, os.cpu_count() // 2) 
    n_workers = 4
    dataset = CurveDataset(
        data_path,
        random_select_rate=None
        )
    # log_dir=log_path+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # model_dir = knot_model_save_dir+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'/'
    log_dir=log_path
    model_dir = knot_model_save_dir
    device='cuda'
    p = 3
    input_dim=dataset.dimension
    learning_rate = 0.0001 #This is very good: learning_rate = 0.0001
    # learning_rate = 0.00001 # have a test
    batch_size = base_batch_size * num_gpus  # 总batch size = 单GPU batch size × GPU数量
    # Create DataLoader for training data
    print('# Create DataLoader for training data')

    train_dataset,val_dataset=random_split(dataset,[int(len(dataset)*0.8),len(dataset)-int(len(dataset)*0.8)],generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=n_workers, pin_memory=True)

    # model=getModel.getModel_Simple(device=device,encoder_load_path=encoder_path,input_dim=input_dim)
    model = getModel.getModel_Simple(device='cuda:0', encoder_load_path=encoder_path, input_dim=input_dim)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)  # 自动分发到所有GPU
        model = model.to(device)  # 移动到主GPU
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

    # TensorBoard setup
    writer = SummaryWriter(log_dir=log_dir)

    def train_step(model,batch,loss_avg:AverageMeter,criterion,train=True):
        batch_points = batch['points'].to(device)
        batch_points_mask = batch['points_mask'].to(device)
        batch_knots = batch['knots_expanded'].to(device)
        batch_knots_mask = batch['knots_mask_expanded'].to(device)

        output,_,_=model(batch_points,batch_points_mask,batch_knots[:,:-1],batch_knots_mask[:,:-1])
        loss = criterion(output,batch_knots[:,1:],batch_knots_mask[:,:-1])

        loss_avg.update(loss.item()*batch_points.size(0),batch_points.size(0))

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        return loss_avg

    # Training loop
    for epoch in range(start_epoch, epochs):

        train_loss_recoder=AverageMeter()
        val_loss_recoder=AverageMeter()

        model.train()
        train_loss_total = 0
        for batch in tqdm(train_loader):
            train_step(model,batch,train_loss_recoder,criterion,train=True)

        train_loss=train_loss_recoder.acc()
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss}")
        writer.add_scalar('Training/Loss', train_loss, epoch)

        # Validation
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch in tqdm(val_loader):
                    train_step(model,batch,val_loss_recoder,criterion,train=False)

        val_loss=val_loss_recoder.acc()

        print(f"Epoch {epoch + 1}/{epochs},Validation Loss: {val_loss}")
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        
        # 保存 checkpoint
        if ifsave:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, f'{model_dir}epoch_{epoch + 1}.pth')
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
            torch.save(checkpoint, f'{model_dir}epoch_{epoch + 1}.pth')
            print(f"Checkpoint saved at epoch {epoch + 1}")

    print("Training complete.")
    writer.close()

def find_latest_checkpoint(model_dir):
    """查找最新的checkpoint文件"""
    if not os.path.exists(model_dir):
        return None
    
    checkpoint_files = []
    for file in os.listdir(model_dir):
        if file.startswith('epoch_') and file.endswith('.pth'):
            # 提取epoch数字
            match = re.search(r'epoch_(\d+)\.pth', file)
            if match:
                epoch_num = int(match.group(1))
                checkpoint_files.append((epoch_num, os.path.join(model_dir, file)))
    
    if checkpoint_files:
        # 返回最新的checkpoint文件路径
        latest_checkpoint = max(checkpoint_files, key=lambda x: x[0])
        return latest_checkpoint[1]
    
    return None