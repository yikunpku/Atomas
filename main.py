import os
from torch.utils.data import DataLoader
from dataset.dataset import ChEBI_20_data_Dataset, PubChem_Dataset
from models.atomas import Atomas
import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin, DDPPlugin
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
import argparse
import yaml

mol_data_directory = "./data"
model_data_directory = "./model_data"
prediction_directory = "./output_data"
seed_everything(42)


def train(args):
    
    if args.dataset == "ChEBI-20_data":
        train_data = ChEBI_20_data_Dataset(
            args.data_dir,
            args.dataset,
            args.train_split,
        )
        
        valid_data = ChEBI_20_data_Dataset(
            args.data_dir,
            args.dataset,
            args.valid_split,
        )
        
        test_data = ChEBI_20_data_Dataset(
            args.data_dir,
            args.dataset,
            args.test_split,
        )

    elif args.dataset == "pubchemstm":
        train_data = PubChem_Dataset(
            args.data_dir,
            "pubchemstm",
            args.split,
        )
        
        valid_data = None
        test_data = None
    else:
        raise Exception("choose pubchemstm or ChEBI_20_data")
    
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    if not valid_data == None:
        valid_loader = DataLoader(
            valid_data,
            batch_size=1,
            num_workers=args.num_workers,
        )
    else:
        valid_loader=None
        
    if not test_data == None:
        test_loader = DataLoader(
            test_data,
            batch_size=1,
            num_workers=args.num_workers,
        )
    else:
        test_loader=None

    model = Atomas(
        args=args,
    )
    
    model_save_path = os.path.join(str(model_data_directory) + "ckpt", args.version)
    os.makedirs(model_save_path, exist_ok=True)
    
    if valid_loader is not None:
        if args.task=="genmol":
            monitor = "valid_bleu_score"
            filename = args.version + "-{epoch:02d}-{step:02d}-{train_loss_tol:.4f}" + ("-{valid_bleu_score:.3f}-{valid_Exact:.3f}-{valid_levenshtein_score:.3f}")
        else:
            monitor = "valid_BLEU2"
            filename = args.version + "-{epoch:02d}-{step:02d}-{train_loss_tol:.4f}" + ("-{valid_BLEU2:.3f}-{valid_BLEU4:.3f}-{valid_ROUGE1:.3f}")
        mode = "max"
        save_top_k = 1
    else:
        monitor = "train_loss_tol"
        mode = "min"
        filename = args.version + "-{epoch:02d}-{step:02d}-{train_loss_tol:.4f}"
        save_top_k = -1
        
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_save_path,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_last=True,
        save_top_k=save_top_k,
    )

    wandb_save_path = os.path.join(str(model_data_directory) + "/wandb",  args.version)
    os.makedirs(wandb_save_path, exist_ok=True)
    wandb_logger = WandbLogger(project=args.project,
                               name=args.version,
                               save_dir=wandb_save_path,
                               config=args,
                            )
    lr_logger = pl.callbacks.LearningRateMonitor()
    
    trainer = pl.Trainer(
        default_root_dir=model_data_directory,
        logger=wandb_logger,
        callbacks=[lr_logger, checkpoint_callback],
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator=args.accelerator,
        strategy=DDPPlugin(find_unused_parameters=True),
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=args.log_every_n_steps,
        max_epochs=args.max_epochs,
        precision=args.precision,
        track_grad_norm=args.track_grad_norm,
    )
    if valid_loader is not None:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader)
    if test_loader is not None:
        torch.distributed.destroy_process_group()
        if trainer.global_rank == 0:
            model_load_pths = [os.path.join(model_save_path, model_ckpt) for model_ckpt in os.listdir(model_save_path)]
            print(f"=======testing:{model_load_pths[0]}=======")   
            model.load_state_dict(torch.load(
                    model_load_pths[0],
                    map_location="cpu")['state_dict'], strict=True)
            model.eval()
            trainer = pl.Trainer(gpus=1)
            trainer.test(model, dataloaders=test_loader)   
    
def main():
    
    with open('_yamls/Pretrain_Atomas.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="Atomas")
    parser.add_argument("--version", type=str, default=config["version"])
    ########## for dataset ##########
    parser.add_argument("--data_dir", type=str, default=mol_data_directory)
    parser.add_argument("--dataset", type=str, default=str(config["dataset"]), choices=["pubchemstm", "ChEBI-20_data"])
    parser.add_argument("--split", type=str, default="distilled")
    parser.add_argument("--train_split", type=str, default=config["train_split"])
    parser.add_argument("--valid_split", type=str, default=config["valid_split"])
    parser.add_argument("--test_split", type=str, default=config["test_split"])
    parser.add_argument("--batch_size", type=int, default=config["batch_size"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_lenth", type=int, default=config["max_lenth"])
    ########## for model ##########
    parser.add_argument("--model_size", type=str, default=config["model_size"])
    parser.add_argument("--queue_size", type=int, default=config["queue_size"])
    parser.add_argument("--task", type=str, default=config["task"], choices=["genmol","gentext"])
    parser.add_argument("--momentum", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--tsclosswt", type=float, default=config["tsclosswt"])
    parser.add_argument("--lmlosswt", type=float, default=config["lmlosswt"])
    parser.add_argument("--wtilosswt", type=float, default=config["wtilosswt"])
    parser.add_argument("--textencoder", type=str, default="molt5")
    parser.add_argument("--encode_text_lr", type=float, default=config["encode_text_lr"])
    parser.add_argument("--encode_smiles_lr", type=float, default=config["encode_smiles_lr"])
    parser.add_argument("--molt5_lr", type=float, default=config["molt5_lr"])
    parser.add_argument("--text_lr_scale", type=float, default=config["text_lr_scale"])
    parser.add_argument("--smiles_lr_scale", type=float, default=config["smiles_lr_scale"])
    parser.add_argument("--decay", type=float, default=config["decay"])
    ########## for train ##########
    parser.add_argument("--precision", default=config["precision"])
    parser.add_argument("--accumulate_grad_batches", type=int, default=config["accumulate_grad_batches"])
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=config["max_epochs"])
    parser.add_argument("--start_valid_epoch", type=int, default=config["start_valid_epoch"])
    parser.add_argument("--track_grad_norm", type=int, default=-1)
    parser.add_argument("--temp_dir", type=str, default=os.path.join(prediction_directory, parser.parse_known_args()[0].version))
    parser.add_argument("--resume_from_checkpoint", type=str, default=config["resume_from_checkpoint"])

    args = parser.parse_args()
    print(args)
    ########## start train ##########
    train(args)

        
if __name__ == "__main__":
    main()
            
    
    

