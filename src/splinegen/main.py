import argparse
import pathlib
import train

import train.train_encoder
import train.train_knots
import train.train_params
import train.train_differientiable_approximation_layer
import train.eval
import datetime

parser = argparse.ArgumentParser("SplineGen")
parser.add_argument(
    "task", choices=("train_encoder","train_knot_decoder","train_param_decoder","train_diff_approximation", "test"), help="Choose train/test task"
)
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--encoder_path",
    type=str,
    default=None,
    help="Weights file of pretrained encoder for train knot decoder",
)

parser.add_argument(
    "--knot_path",
    type=str,
    default=None,
    help="Weights file of pretrained knot model for train param decoder",
)

parser.add_argument(
    "--base_model_path",
    type=str,
    default=None,
    help="Weights file of pretrained model for train differientiable approximation layer",
)

parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="Weights file of pretrained model for test",
)

parser.add_argument(
    "--experiment_name",
    type=str,
    default=None,
    help="Experiment name (used to create folder inside ./results/ to save logs and models), default is task name",
)
parser.add_argument(
    "--epoch",
    type=int,
    default=1000,
)
parser.add_argument(
    "--save",
    type=bool,
    default=False,
)
parser.add_argument(
    "--reuse",
    type=str,
    default=None,
)
parser.add_argument(
    "--result_dir",
    type=str,
    default=None,
)

# parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

if not args.experiment_name:
    args.experiment_name = args.task
from pathlib import Path
if args.result_dir:
    results_path = Path(args.result_dir).joinpath(args.experiment_name)
else:
    results_path = pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
    # results_path = Path("/work") / "results" / args.experiment_name
log_path= str(results_path.joinpath('logs'))
model_save_path= str(results_path.joinpath('models'))

log_path=log_path+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_path = model_save_path+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'/'
# log_path=log_path+'/'+"1"
# model_save_path = model_save_path+'/'+"1"+'/'

if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

if args.task=='train_encoder':
    train.train_encoder.train(args.dataset_path,log_path,model_save_path,args.epoch,args.batch_size,args.save,resume_from=args.reuse)

elif args.task=='train_knot_decoder':
    train.train_knots.train(args.dataset_path,log_path,args.encoder_path,model_save_path,args.epoch,args.batch_size,args.save,resume_from=args.reuse)

elif args.task=='train_param_decoder':
    train.train_params.train(args.save,args.dataset_path,model_save_path,log_path,args.knot_path,n_epochs=args.epoch,batch_size=args.batch_size,resume_from=args.reuse)

elif args.task=='train_diff_approximation':
    train.train_differientiable_approximation_layer.train(args.save,args.dataset_path,log_path,model_save_path,args.base_model_path,n_epochs=args.epoch,batch_size=args.batch_size,resume_from=args.reuse)

elif args.task=='test':
    train.eval.eval(args.dataset_path,args.model_path)
