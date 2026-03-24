import shutup
shutup.please()
import flwr as fl
import os
os.environ["WANDB_MODE"] = "offline"
import torch
from torch.nn.utils import vector_to_parameters
from dataset import get_local_dataloaders, get_local_dataloaders_cv
from utils import train, set_seed, load_config, evaluation
import argparse
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from collections import OrderedDict
import tqdm
from transformers import get_scheduler
from utils import get_model
tqdm.tqdm = tqdm.tqdm_notebook




class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, config):

        set_seed(42)
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        loader_list, self.val_loader = get_local_dataloaders_cv(config, num_clients=3, kfold_number=self.config.kfold_number)
        self.train_loader = loader_list[client_id]

        if hasattr(self.train_loader, 'num_workers') and self.train_loader.num_workers > 0:
            print(f"Client {client_id}: Setting multiprocessing context")

        self.model = get_model(config, self.device)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.best_f1_across_round = 0


    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]


    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.from_numpy(new_param).to(self.device)


    def fit(self, parameters, config=None):
        self.set_parameters(parameters)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.train.num_epochs)

        log_path = os.path.join(self.config.log_path, 'train_stats')
        os.makedirs(log_path, exist_ok=True)
        log_path_train_stats = f'{log_path}/client_{self.client_id}_fold_{self.config.kfold_number}_train_stats.txt'

        model, best_value, rest_best_values = train(
            self.model,
            self.train_loader,
            self.val_loader,
            self.loss_fn,
            optimizer,
            scheduler,
            self.config.train.num_epochs,
            model_name=self.config.model_name,
            early_stopping=self.config.train.early_stopping,
            early_stopping_patience=self.config.train.early_stopping_patience,
            log_path=log_path_train_stats,
        )

        return self.get_parameters(), len(self.train_loader.dataset), {"accuracy": best_value, "f1": rest_best_values[0], "client_id": self.client_id}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()
        validation_value, rest_values = evaluation(self.model, self.val_loader, self.loss_fn, log=None, nolog=True)
        total = len(self.val_loader)

        return float(0.0), total, {"accuracy": validation_value, "f1": rest_values[0]}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--kfold_number", type=int, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config.kfold_number = args.kfold_number

    client = FlowerClient(args.client_id, config)
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client(),
        max_retries=5,
        max_wait_time=120,
    )
