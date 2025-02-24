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


def evaluation(args):
    
    test_data = ChEBI_20_data_Dataset(
        args.data_dir,
        args.dataset,
        args.test_split,
    )


    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


    model = Atomas(
        args=args,
    )

    model.load_state_dict(torch.load(
            args.resume_from_checkpoint,
            map_location="cpu")['state_dict'], strict=True)
    model.eval()
    
    trainer = pl.Trainer(gpus=1)
    trainer.test(model, dataloaders=test_loader)   
    
def main():
    
    with open('_yamls/Eval_Atomas.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="Atomas")
    parser.add_argument("--mode", type=str, default="eval")
    parser.add_argument("--version", type=str, default=config["version"])
    ########## for dataset ##########
    parser.add_argument("--data_dir", type=str, default=mol_data_directory)
    parser.add_argument("--dataset", type=str, default=str(config["dataset"]), choices=["pubchemstm", "ChEBI-20_data"])
    parser.add_argument("--split", type=str, default="distilled")
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
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--track_grad_norm", type=int, default=-1)
    parser.add_argument("--temp_dir", type=str, default=os.path.join(prediction_directory, parser.parse_known_args()[0].version))
    parser.add_argument("--resume_from_checkpoint", type=str, default=config["resume_from_checkpoint"])

    args = parser.parse_args()
    print(args)
    ########## start train ##########
    evaluation(args)

        
if __name__ == "__main__":
    main()
            
    
    

