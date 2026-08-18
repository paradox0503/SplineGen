import datetime
import os

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.curveDataset import CurveDataset
import train.getModel as getModel
from util import AverageMeter, nurbs_eval


def _masked_mean(loss, mask):
    mask = mask.to(loss.dtype)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    weighted = loss * mask
    dimensions = tuple(range(1, weighted.ndim))
    denominator = mask.expand_as(weighted).sum(dim=dimensions).clamp_min(1.0)
    return (weighted.sum(dim=dimensions) / denominator).mean()


def parameter_loss(predicted, target, mask):
    elementwise = torch.nn.functional.mse_loss(predicted, target, reduction='none')
    return _masked_mean(elementwise, mask)


def knot_loss(predicted, target, mask):
    elementwise = torch.nn.functional.mse_loss(predicted, target, reduction='none')
    return _masked_mean(elementwise, mask)


def _loaders(data_path, batch_size, n_workers):
    dataset = CurveDataset(
        data_path,
        use_points_params=True,
        use_knots=True,
        use_orders=False,
        random_select_rate=None,
    )
    train_set, validation_set = random_split(dataset, lengths=(0.8, 0.2))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=n_workers, shuffle=True
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, num_workers=n_workers, shuffle=False
    )
    return dataset, train_loader, validation_loader


def train_parameters(
    data_path,
    model_save_path,
    log_path,
    knot_model_load_path=None,
    train_weights=(0.2, 0.8),
    use_cuda=True,
    n_workers=4,
    n_epochs=1000,
    batch_size=256,
    lr=1e-4,
    save_epoch=5,
):
    """Jointly train automatic knots and monotonic parameters for ordered points."""
    torch.random.manual_seed(231)
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    run_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(model_save_path, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_path, run_name))

    dataset, train_loader, validation_loader = _loaders(
        data_path, batch_size, n_workers
    )
    model = getModel.getOrderedSplineGen(
        device=device,
        knot_load_path=knot_model_load_path,
        input_dim=dataset.dimension,
        lock_encoder=False,
        lock_knots=False,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        meters = {
            'train': [AverageMeter(), AverageMeter(), AverageMeter()],
            'validation': [AverageMeter(), AverageMeter(), AverageMeter()],
        }
        for split, loader, training in (
            ('train', train_loader, True),
            ('validation', validation_loader, False),
        ):
            model.train(training)
            context = torch.enable_grad() if training else torch.no_grad()
            with context:
                for batch in tqdm(loader, desc=f'Epoch {epoch + 1} {split}'):
                    points = batch['points'].to(device)
                    point_mask = batch['points_mask'].to(device)
                    target_params = batch['params'].to(device=device, dtype=torch.float32)
                    knot_tokens = batch['knots_expanded'].to(device)
                    knot_token_mask = batch['knots_mask_expanded'].to(device)
                    if training:
                        optimizer.zero_grad()

                    predicted_knots, predicted_params = model(
                        points, point_mask,
                        knot_tokens[:, :-1], knot_token_mask[:, :-1],
                    )
                    loss_k = knot_loss(
                        predicted_knots, knot_tokens[:, 1:], knot_token_mask[:, :-1]
                    )
                    loss_p = parameter_loss(predicted_params, target_params, point_mask)
                    loss = train_weights[0] * loss_k + train_weights[1] * loss_p
                    if training:
                        loss.backward()
                        optimizer.step()

                    meters[split][0].update(loss.item(), points.size(0))
                    meters[split][1].update(loss_p.item(), points.size(0))
                    meters[split][2].update(loss_k.item(), points.size(0))

        for split in ('train', 'validation'):
            writer.add_scalar(f'Loss/{split}', meters[split][0].avg, epoch)
            writer.add_scalar(f'Parameter Loss/{split}', meters[split][1].avg, epoch)
            writer.add_scalar(f'Knot Loss/{split}', meters[split][2].avg, epoch)
        if (epoch + 1) % save_epoch == 0:
            writer.flush()
            torch.save(model.state_dict(), os.path.join(output_dir, f'epoch_{epoch + 1}.pth'))

    writer.close()


def train_geometry(
    data_path,
    log_path,
    model_save_path,
    base_model_load_path,
    use_cuda=True,
    n_workers=4,
    n_epochs=1000,
    batch_size=256,
    lr=1e-6,
):
    """Fine-tune ordered parameters and knots through differentiable fitting loss."""
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    run_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(model_save_path, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_path, run_name))
    dataset, train_loader, validation_loader = _loaders(data_path, batch_size, n_workers)
    model = getModel.getOrderedSplineGen(
        device=device,
        model_load_path=base_model_load_path,
        input_dim=dataset.dimension,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        meters = {'train': AverageMeter(), 'validation': AverageMeter()}
        for split, loader, training in (
            ('train', train_loader, True),
            ('validation', validation_loader, False),
        ):
            model.train(training)
            context = torch.enable_grad() if training else torch.no_grad()
            with context:
                for batch in tqdm(loader, desc=f'Epoch {epoch + 1} {split}'):
                    points = batch['points'].to(device)
                    point_mask = batch['points_mask'].to(device)
                    if training:
                        optimizer.zero_grad()
                    knots, knot_mask, params = model.generate(points, point_mask)
                    knot_length = knot_mask.sum(dim=-1)
                    loss, _ = nurbs_eval.getCtrlPts(
                        3, [params, points, point_mask, knots, knot_length]
                    )
                    if training:
                        loss.backward()
                        optimizer.step()
                    meters[split].update(loss.item() / points.size(0), points.size(0))

        writer.add_scalar('Loss/train', meters['train'].avg, epoch)
        writer.add_scalar('Loss/validation', meters['validation'].avg, epoch)
        if (epoch + 1) % 5 == 0:
            writer.flush()
            torch.save(model.state_dict(), os.path.join(output_dir, f'epoch_{epoch + 1}.pth'))

    writer.close()


def evaluate(
    data_path,
    model_load_path,
    use_cuda=True,
    n_workers=4,
    batch_size=256,
):
    """Evaluate ordered inputs directly, without pointer-based point gathering."""
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    dataset = CurveDataset(
        data_path, use_points_params=True, use_knots=True,
        use_orders=False, random_select_rate=None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False)
    model = getModel.getOrderedSplineGen(
        device=device, model_load_path=model_load_path, input_dim=dataset.dimension
    )
    model.eval()
    mean_loss, sum_loss, hausdorff_loss = AverageMeter(), AverageMeter(), AverageMeter()

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating ordered SplineGen'):
            points = batch['points'].to(device)
            point_mask = batch['points_mask'].to(device)
            knots, knot_mask, params = model.generate(points, point_mask)
            knot_length = knot_mask.sum(dim=-1)
            loss1, loss2, loss3, _ = nurbs_eval.getCtrlPts2(
                3, [params, points, point_mask, knots, knot_length]
            )
            mean_loss.update(loss1.item() / points.size(0), points.size(0))
            sum_loss.update(loss2.item() / points.size(0), points.size(0))
            hausdorff_loss.update(loss3.item() / points.size(0), points.size(0))

    return {
        'mean_loss': mean_loss.avg,
        'sum_loss': sum_loss.avg,
        'hausdorff_loss': hausdorff_loss.avg,
    }
