import shutup
shutup.please()
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dataset import get_dataloaders
from dataset import get_local_dataloaders
from utils import set_seed, get_config, train, save_config
from model import CrossAttentionTransformerEncoder, MyTransformerEncoder, BidirectionalCrossAttentionTransformerEncoder, ElementWiseFusionEncoder
import torch
os.environ["WANDB_MODE"] = "offline"
import wandb
import sys
import torch.nn as nn
from transformers import get_scheduler
from torch.optim import AdamW
import argparse
import subprocess
from subprocess import Popen, PIPE, STDOUT
import time
import sys
import os
from utils import load_config, extract_best_results, generate_tsv_summary
##

def set_up(config, train_dataloader, device, fold=0):
    """Set up model, optimizer, loss function, and scheduler."""
    set_seed(3407)

    # if config.model.multimodality:
    if not config.model.single_modal:
        if 'bicross' in config.model.fusion:
            model = BidirectionalCrossAttentionTransformerEncoder(config.model).to(device)
        elif 'cross' in config.model.fusion:
            model = CrossAttentionTransformerEncoder(config.model).to(device)
        else:
            model = ElementWiseFusionEncoder(config.model).to(device)
    else:
         model = MyTransformerEncoder(config.model).to(device)



    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    lossfn = nn.BCEWithLogitsLoss()
    
    num_training_steps = config.train.num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="cosine", optimizer=optimizer, num_warmup_steps=20, num_training_steps=num_training_steps
    )

    
    wandb.init(
        project="WordLevelFusion",
        name=f"{config.model_name}_{fold}" if config.train.cross_validation else config.model_name,
        config={
            "learning_rate": config.train.learning_rate,
            "architecture": config.model_name,
            "dataset": "ADReSSo",
            "epochs": config.train.num_epochs,
            "batch_size": config.train.batch_size,
        }
    )
    
    wandb.watch(model)
    return model, optimizer, lossfn, lr_scheduler

def cl_main(config):
    wandb.login()
    """Main function to train and save model, supporting cross-validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = config.log_path
    os.makedirs(log_path, exist_ok=True)
    print(log_path)


    if config.train.learning_paradigm == 'cl':
        log_file = os.path.join(log_path, 'cross_fold_summary.txt')
        with open(log_file, "w") as log:
            for fold in range(config.train.cross_validation_folds):
                train_dataloader, validation_dataloader = get_dataloaders(config, kfold_number=fold)
                model, optimizer, lossfn, lr_scheduler = set_up(config, train_dataloader, device, fold)
                model, best_value, rest_best_values = train(
                    model, train_dataloader, validation_dataloader, lossfn, optimizer, lr_scheduler,
                    config.train.num_epochs, config.path_name, config.train.early_stopping, 
                    config.train.early_stopping_patience, config.train.cross_validation, fold
                )
                
                log.write(f'Fold {fold}: Best Value = {best_value}\n')
                log.write(f'Best F1: {rest_best_values[0]}\nBest Recall: {rest_best_values[1]}\nBest Precision: {rest_best_values[2]}\n')
                
                torch.save(model.state_dict(), os.path.join(log_path, f'model_fold_{fold}.pth'))
                print(f'Model for fold {fold} saved')
                wandb.log({
                    "best_value": best_value,
                    "best_f1": rest_best_values[0],
                    "best_recall": rest_best_values[1],
                    "best_precision": rest_best_values[2],
                })
                wandb.run.summary["best_value"] = best_value
                wandb.run.summary["best_f1"] = rest_best_values[0]
                wandb.run.summary["best_recall"] = rest_best_values[1]
                wandb.run.summary["best_precision"] = rest_best_values[2]
                wandb.finish()

        generate_tsv_summary(log_path)

    elif config.train.learning_paradigm == 'll':
        local_dataloaders, val_loader = get_local_dataloaders(config, num_clients=3)

        for client_id, train_loader in enumerate(local_dataloaders):
            print(f"Training client {client_id}")
            model, optimizer, lossfn, lr_scheduler = set_up(config, train_loader, device, fold=client_id)

            model, best_value, rest_best_values = train(
                model, train_loader, val_loader, lossfn, optimizer, lr_scheduler,
                config.train.num_epochs, config.path_name,    # ✅ 不拼 client_id
                config.train.early_stopping, config.train.early_stopping_patience
            )

            model_save_path = os.path.join(log_path, f"model_client_{client_id}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved for client {client_id}")

            wandb.log({
                "client": client_id,
                "best_value": best_value,
                "best_f1": rest_best_values[0],
                "best_recall": rest_best_values[1],
                "best_precision": rest_best_values[2],
            })
            wandb.finish()


    else:
        train_dataloader, validation_dataloader = get_dataloaders(config)
        
        model, optimizer, lossfn, lr_scheduler = set_up(config, train_dataloader, device)
        model, best_value, rest_best_values = train(
            model, train_dataloader, validation_dataloader, lossfn, optimizer, lr_scheduler, 
            config.train.num_epochs, config.path_name, config.train.early_stopping, 
            config.train.early_stopping_patience
        )
        
        model_save_path = os.path.join(log_path, 'model.pt')
        torch.save(model.state_dict(), model_save_path)
        print('Model saved')
        wandb.finish()


def fl_main(config_path, config, num_folds=5):
    # 参数可自定义
    NUM_CLIENTS = 3
    # 修改SERVER_CMD以包含日志参数
    SERVER_CMD = ["python", "server.py", "--config", config_path]
    CLIENT_CMD_TEMPLATE = ["python", "client.py", "--client_id", "{cid}", "--config", config_path, "--kfold_number", "{fold}"]

    base_log_path = config.log_path
    log_path_server = os.path.join(base_log_path, 'server')
    log_path_client = os.path.join(base_log_path, 'client')
    os.makedirs(log_path_server, exist_ok=True)
    os.makedirs(log_path_client, exist_ok=True)

    print(config.train.strategy)
    for fold in range(num_folds):
        print(f"\n=== Starting Cross Validation Fold {fold} ===")

        server_cmd = [x.format(fold) if '{}' in x else x for x in SERVER_CMD]

        server_log_file = open(f"{log_path_server}/server_fold_{fold}.log", "w")
        server_proc = subprocess.Popen(server_cmd, stdout=server_log_file, stderr=server_log_file)
        print(f"Started Flower server for fold {fold}.")

        time.sleep(2)

        client_procs = []
        for cid in range(NUM_CLIENTS):
            env = os.environ.copy()
            cvd = config.train.cvd if config.train.cvd else 0
            env["CUDA_VISIBLE_DEVICES"] = str(cid % torch.cuda.device_count() + cvd)

            cmd = [x.replace('{cid}', str(cid)).replace('{fold}', str(fold)) for x in CLIENT_CMD_TEMPLATE]
            log_file_path = f"{log_path_client}/fold_{fold}_client_{cid}.log"
            out_file = open(log_file_path, "w")
            proc = subprocess.Popen(cmd, env=env, stdout=out_file, stderr=STDOUT)
            client_procs.append((proc, out_file))
            print(f"Started client {cid} for fold {fold}.")

        for proc, out_file in client_procs:
            proc.wait()
            out_file.close()

        server_log_file.close()
        server_proc.wait()

        print(f"Fold {fold} FL processes finished!")

    print("All cross-validation folds completed!")
    extract_best_results(log_path_server, 'Local')
    if config.train.learning_paradigm == 'fl':
        extract_best_results(log_path_server, 'Federal')
        extract_best_results(log_path_server, 'Global')




if __name__ == '__main__':
    config_path = sys.argv[sys.argv.index('--config') + 1]
    config = get_config(config_path)

    save_config(config)

    learning_paradigm = getattr(config.train, 'learning_paradigm', None)
    if learning_paradigm in ['fl', 'll'] :
        fl_main(config_path, config)
    else:
        cl_main(config)