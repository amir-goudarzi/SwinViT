import numpy as np
import argparse
import torch.utils
import tqdm
import random
import logging
from tqdm import tqdm
import os

import torch
import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.optim.lr_scheduler import _LRScheduler
# import torchvision
# from torchvision import datasets, transforms
# from torch.optim import AdamW, Adam
from torch.cuda.amp import autocast, GradScaler

# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange

# from functools import partial

# from PIL import ImageFilter, ImageOps, Image

# from ignite.utils import convert_tensor

# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model

from utils.make_dataloader import get_loaders
from src.swin_vit import SwinTransformer
from utils.scheduler import build_scheduler  
from utils.optimizer import get_adam_optimizer
from utils.utils import clip_gradients
from utils.utils import save_checkpoint
# from utils.cutmix import CutMix

import warnings
warnings.filterwarnings("ignore")

import neptune.new as neptune


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, lr_scheduler, loss_fn, device, params, args):
        self.params = params
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.args = args
        self.scaler = GradScaler()
        # self.cutmix = CutMix(loss_fn)

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.run = neptune.init_run(
            project=args.neptune_project,
            api_token=args.neptune_api_token
        )
        self.run["parameters"] = params


    def train(self):

        print("\n--- Training Progress ---\n")

        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        best_accuracy = 0.0

        for epoch in range(self.args.epochs):
            epoch_progress_bar = tqdm(desc=f"Epoch {epoch + 1}/{self.args.epochs}", unit="batch", disable=os.environ.get("SSH_CLIENT"))

            # Training Phase
            self.model.train()
            total_train_loss, total_train_correct, total_train_samples = 0.0, 0, 0
            for batch in self.train_loader:
                # images, labels = self.cutmix.prepare_batch(batch, self.device, non_blocking=True)
                self.optimizer.zero_grad()
                x, _, sudoku_label = batch
                x, sudoku_label = x.to(self.device), sudoku_label.to(self.device)

                with autocast():
                    outputs = self.model(x)
                    loss = self.loss_fn(outputs, sudoku_label)

                self.scaler.scale(loss).backward()

                if self.args.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_gradients(self.model, self.args.clip_grad)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train_correct += (predicted == sudoku_label).sum().item()
                total_train_samples += x.shape[0]
                
                # Log training loss to Neptune
                self.run[f"train/loss"].log(loss.item())

                epoch_progress_bar.update(1)

            avg_train_loss = total_train_loss / total_train_samples
            train_accuracy = (total_train_correct / total_train_samples) * 100
            self.run[f"train/accuracy"].log(train_accuracy)

            # Validation Phase
            self.model.eval()
            total_val_loss, total_val_correct, total_val_samples = 0.0, 0, 0
            with torch.no_grad():
                for images, _, sudoku_label in self.val_loader:
                    images, sudoku_label = images.to(self.device), sudoku_label.to(self.device)
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, sudoku_label)
                    total_val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val_correct += (predicted == sudoku_label).sum().item()
                    total_val_samples += images.shape[0]
                    
                    # Log validation loss to Neptune
                    self.run[f"val/loss"].log(loss.item())

                    epoch_progress_bar.update(1)

            avg_val_loss = total_val_loss / total_val_samples
            val_accuracy = (total_val_correct / total_val_samples) * 100
            current_lr = self.optimizer.param_groups[0]['lr']

            epoch_progress_bar.set_postfix({"Train Loss": avg_train_loss, "Train Acc": train_accuracy, "Val Loss": avg_val_loss, "Val Acc": val_accuracy, "LR": current_lr})
            # Log validation accuracy to Neptune
            self.run[f"val/accuracy"].log(val_accuracy)
            
            epoch_progress_bar.close()

            # Logging and Checkpointing
            self.logger.info(f"Epoch {epoch + 1}/{self.args.epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}, LR: {current_lr:.4f}")
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                # save_checkpoint(self.args.checkpoint_dir, self.model, epoch)
                self.logger.info(f"New best accuracy: {best_accuracy:.4f}, Model saved as 'best_model.pth'")

            if self.args.scheduler:
                self.lr_scheduler.step()

            print(f"\n\n--- Done epoch number {epoch} ---\n\n")

        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def test(self):
        
        self.model.eval()  
        total_test_loss, total_test_correct, total_test_samples = 0.0, 0, 0

        self.logger.info("---Testing Phase---")

        with torch.no_grad():
            for images, _, sudoku_label in self.test_loader:
                images, sudoku_label = images.to(self.device), sudoku_label.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, sudoku_label)
                
                total_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test_correct += (predicted == sudoku_label).sum().item()
                total_test_samples += images.shape[0]

        avg_test_loss = total_test_loss / total_test_samples
        test_accuracy = total_test_correct / total_test_samples

        self.logger.info(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        # Log testing accuracy to Neptune
        self.run[f"test/accuracy"].log(test_accuracy)
        self.run.stop()

        return {
            "test_loss": avg_test_loss,
            "test_accuracy": test_accuracy
        }




def main():
    parser = argparse.ArgumentParser('SWIN ViT for VisualSudoku', add_help=False)
    parser.add_argument('--dir', type=str, default='/content/drive/MyDrive/Colab Notebooks/',
                    help='Data directory')

    # Model parameters
    parser.add_argument('--patch_size', default=4, type=int, help="""Size in pixels of input square patches - default 4 (for 4x4 patches) """)
    parser.add_argument('--window_size', default=4, type=int, help="""Window size. """)
    parser.add_argument('--out_dim', default=1024, type=int, help="""Dimensionality of the SSL MLP head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--grid_size', default=4, type=int, help="""Size of the puzzle (a row/column or a subgrid)""")
    parser.add_argument('--norm_last_layer', default=False, type=bool,
        help="""Whether or not to weight normalize the last layer of the MLP head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool,
        help="Whether to use batch normalizations in projection head (Default: False)")

    parser.add_argument('--image_size', default=112, type=int, help=""" Size of input image. """)
    parser.add_argument('--in_channels',default=3, type=int, help=""" input image channels. """)
    parser.add_argument('--embed_dim',default=192, type=int, help=""" dimensions of vit """)
    parser.add_argument('--num_layers',default="2, 6, 4", type=str, help=""" No. of layers of ViT in each stage""")
    parser.add_argument('--num_heads',default="3, 6, 12", type=str, help=""" No. of heads in attention layer
                                                                                 in ViT """)
    parser.add_argument('--vit_mlp_ratio',default=2, type=int, help=""" MLP hidden dim """)
    parser.add_argument('--qkv_bias',default=True, type=bool, help=""" Bias in Q K and V values """)
    parser.add_argument('--mlp_dropout',default=0.1, type=float, help=""" MLP dropout """)
    parser.add_argument('--attn_dropout',default=0., type=float, help=""" attention dropout """)
    parser.add_argument('--positional_encoding', default='learned', type=str,
        choices=['learned', 'freq', 'abs'], help="""Type of positional encoding.""")

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=1e-1, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--batch_size', default=10, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--optimizer', default='Adam', type=str,
        choices=['Adam', 'SGD'], help="""Type of optimizer. Recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                    help='Label smoothing for optimizer')
    parser.add_argument("--scheduler", type=bool, required=False, default=True, help="True to use scheduler")
    parser.add_argument('--gamma', type=float, default=1.0,
                    help='Gamma value for Cosine LR schedule')
    

    # Misc
    parser.add_argument('--dataset_path', default='VisualSudoku', type=str, help='Please specify path to the training data.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--mlp_head_in", default=192, type=int, help="input dimension going inside MLP projection head")
    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="directory to save checkpoints")
    parser.add_argument("--neptune_project", type=str, help="Neptune project directory")
    parser.add_argument("--neptune_api_token", type=str, help="Neptune api token")
    

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("\n--- GPU Information ---\n")

    if torch.cuda.is_available():
        print(f"Model is using device: {device}")
        print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 2} MB")
    else:
        print("Model is using CPU")

    params = {
        "PATCH_SIZE": args.patch_size,
        "WINDOW_SIZE": args.window_size,
        "POSITIONAL_ENCODING": args.positional_encoding,
        "EMBEDDING_DIM": args.embed_dim,
        "NUM_TRANSFORMER_LAYERS": args.num_layers,
        "MLP_DROPOUT": args.mlp_dropout,
        "ATTN_DROPOUT": args.attn_dropout,
        "MLP_SIZE": args.embed_dim * args.vit_mlp_ratio,
        "NUM_HEADS": args.num_heads,
        "BATCH_SIZE": args.batch_size,
        "ADAM_OPTIMIZER": args.optimizer,
        "LEARNING_RATE": args.lr,
        "NUM_EPOCHS": args.epochs,
        "SCHEDULER_USED": args.scheduler,
        "NUM_WORKERS": args.num_workers
    }


    train_loader, val_loader, test_loader, n_classes = get_loaders(batch_size= params['BATCH_SIZE'], num_workers=params['NUM_WORKERS'], path= os.path.join(args.dir, args.dataset_path), return_whole_puzzle=True)
    print("\n ---Dataloaders succusfully created--- \n")

    num_layers = [int(item) for item in args.num_layers.split(',')]
    num_heads = [int(item) for item in args.num_heads.split(',')]
    model = SwinTransformer(img_size=args.image_size,
                        num_classes=n_classes,
                        window_size=params['WINDOW_SIZE'], 
                        patch_size=params['PATCH_SIZE'], 
                        in_chans=args.in_channels,
                        embed_dim=params['EMBEDDING_DIM'], 
                        depths=num_layers, 
                        num_heads=num_heads,
                        mlp_ratio=args.vit_mlp_ratio, 
                        qkv_bias=True, 
                        drop_rate=params['MLP_DROPOUT'],
                        attn_drop_rate=params['ATTN_DROPOUT'],
                        drop_path_rate=args.drop_path_rate).to(device)
    

    loss = nn.CrossEntropyLoss()
    if args.optimizer == "Adam":
        optimizer = get_adam_optimizer(model.parameters(), lr=args.lr, wd=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    lr_scheduler = build_scheduler(args, optimizer)

    

    trainer = Trainer(model, train_loader, val_loader, test_loader, optimizer, lr_scheduler, loss, device, params, args)
    trainer.train()
    trainer.test()
    


if __name__ == "__main__":
    main()
