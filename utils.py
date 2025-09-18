import torch
import numpy as np
import random
import yaml
import os
from dotmap import DotMap
import wandb
from tqdm import tqdm
import time
import copy
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from model import (
    MyTransformerEncoder,
    CrossAttentionTransformerEncoder,
    BidirectionalCrossAttentionTransformerEncoder,
    ElementWiseFusionEncoder,
)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_config(config_file):
    """Load configuration from a YAML file and ensure log directories exist."""

    with open(config_file, 'r') as f:
        config_yaml = yaml.safe_load(f)
    config = DotMap(config_yaml)
    return config


def save_config(config):
    """Save the configuration to a YAML file, ensuring log directories exist."""
    config.model.multimodality = config.model.textual_model != '' and config.model.audio_model != ''
    textual_data = config.model.textual_model + '_' if config.model.textual_model != '' else ''
    audio_data = config.model.audio_model + '_' if config.model.audio_model != '' else ''
    pauses_data = 'P_' if config.model.pauses else ''
    config.model_name = f"{textual_data}{audio_data}{pauses_data}{config.model.fusion}"
    config.model.model_name = config.model_name

    config.path_name = f"{config.model_name}_{config.model.pooling}"


    # log_path = os.path.join('logs', config.path_name)
    log_path = os.path.join('logs', config.train.learning_paradigm, config.path_name)
    os.makedirs(log_path, exist_ok=True)
    config.log_path = log_path

    config_file_path = os.path.join(log_path, 'config.yaml')
    # Convert DotMap to a standard dictionary
    config_dict = config.toDict()
    with open(config_file_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    return config


def get_metrics_classification(true_labels, pred_labels):
    """Compute classification metrics safely."""
    zero_div = 1 if len(set(true_labels)) == 1 else 0  # Avoid zero division warnings

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=zero_div)
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=zero_div)
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=zero_div)

    return accuracy, f1, recall, precision



def train(model, train_dataloader, valid_dataloader, lossfn, optimizer, lr_scheduler, num_epochs, model_name,
          early_stopping, early_stopping_patience, cross_val=False, num_cross_val=0, log_path=None):
    """Train the model with early stopping."""
    wandb.init(project="WordLevelFusion", config={"epochs": num_epochs})
    wandb.watch(model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if not log_path:
        log_path = f'logs/cl/{model_name}/train_stats_{num_cross_val}.txt' if cross_val else f'logs/cl/{model_name}/train_stats.txt'

    best_value, patience = 0, 0
    best_epoch, best_weights, rest_best_values = 0, None, []

    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log:
        for epoch in range(num_epochs):
            model.train()
            total_true, total_pred, total_loss = [], [], 0

            progress_bar.set_description(f"Epoch {epoch + 1}")
            log.write(f'Epoch {epoch + 1}:\n')

            for features, labels in train_dataloader:
                features[0].to(device)
                features[1].to(device)
                labels = labels.to(device)  # ✅ 迁移到 GPU

                outputs = model(features)

                if isinstance(outputs, tuple) and len(outputs) == 2:
                    outputs, all_vq_loss = outputs
                    loss = lossfn(outputs.squeeze(-1), labels) + all_vq_loss
                else:
                    loss = lossfn(outputs.squeeze(-1), labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()

                probs = torch.sigmoid(outputs)
                predictions = torch.round(probs)

                if torch.isnan(predictions).any():
                    print("⚠️ Warning: NaN detected in predictions! Skipping batch.")
                    continue

                predictions = predictions.detach().cpu().numpy().astype(int)
                labels = labels.detach().cpu().numpy().astype(int)

                total_true.extend(labels)
                total_pred.extend(predictions)
                progress_bar.update(1)

            accuracy, f1, recall, precision = get_metrics_classification(total_true, total_pred)
            avg_loss = total_loss / len(train_dataloader)

            log.write(f'Training completed in: {time.time()} seconds\n')
            log.write(
                f'Loss: {avg_loss}\nAccuracy: {accuracy}\nF1 Score: {f1}\nRecall: {recall}\nPrecision: {precision}\n')
            wandb.log({"train_loss": avg_loss, "train_ACC": accuracy, "train_F1": f1})

            validation_value, rest_values = evaluation(model, valid_dataloader, lossfn, log)

            if validation_value > best_value:
                best_epoch, best_weights = epoch + 1, copy.deepcopy(model.state_dict())
                best_value, rest_best_values = validation_value, rest_values
                patience = 0
            else:
                patience += 1

            if patience == early_stopping_patience and early_stopping:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        if not rest_best_values:
            rest_best_values = [0, 0, 0]

        log.write(f'Best validation accuracy: {best_value}\n')
        log.write(
            f'Best validation F1: {rest_best_values[0]}\nBest validation Recall: {rest_best_values[1]}\nBest validation Precision: {rest_best_values[2]}\n')
        log.write(f'Best epoch: {best_epoch}\n')

    model.load_state_dict(best_weights)
    return model, best_value, rest_best_values


def evaluation(model, dataloader, lossfn, log, test=False, nolog=False):
    """Evaluate the model on a given dataset."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_true, total_pred, total_loss = [], [], 0

    with torch.no_grad():
        for features, labels in dataloader:
            features[0].to(device)
            features[1].to(device)
            labels = labels.to(device)  # ✅ 每个 batch 的 labels 搬到 GPU

            outputs = model(features)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                outputs, all_vq_loss = outputs
                loss = lossfn(outputs.squeeze(-1), labels) + all_vq_loss
            else:
                loss = lossfn(outputs.squeeze(-1), labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            predictions = torch.round(probs)

            if torch.isnan(predictions).any():
                print("⚠️ Warning: NaN detected in predictions! Skipping batch.")
                continue

            predictions = predictions.detach().cpu().numpy().astype(int)
            labels = labels.detach().cpu().numpy().astype(int)

            total_true.extend(labels)
            total_pred.extend(predictions)

    accuracy, f1, recall, precision = get_metrics_classification(total_true, total_pred)
    avg_loss = total_loss / len(dataloader)

    if not nolog:
        log.write(f'Loss: {avg_loss}\nAccuracy: {accuracy}\nF1 Score: {f1}\nRecall: {recall}\nPrecision: {precision}\n')
        wandb.log(
            {"test_loss": avg_loss, "test_acc": accuracy, "test_F1": f1} if test else {"validation_loss": avg_loss,
                                                                                       "validation_ACC": accuracy,
                                                                                       "validation_F1": f1})

    return accuracy, [f1, recall, precision]


def get_model_statistics(model='all'):
    directory = 'logs/'
    folder_names = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

    # Ordered structure
    grouped_results = {}
    models_used = set()

    for folder_name in folder_names:
        try:
            model_name, pooling = folder_name.split('_')  # Expected: "distilbert_base_cls"
        except ValueError:
            print(f"Warning: Unexpected folder name format {folder_name}, skipping.")
            continue

        if model != 'all' and model not in model_name:
            continue

        file_path = os.path.join(directory, folder_name, 'cross_fold_summary.txt')

        if not os.path.exists(file_path):
            print(f"Warning: Missing file {file_path}")
            continue

        try:
            with open(file_path, "r") as result_file:
                lines = result_file.readlines()

            if not lines:
                print(f"Warning: Empty file {file_path}")
                continue

            metrics = {'acc': [], 'f1': [], 'recall': [], 'precision': []}

            for i in range(0, len(lines), 4):
                try:
                    metrics['acc'].append(float(lines[i].split()[-1]) * 100)
                    metrics['f1'].append(float(lines[i + 1].split()[-1]) * 100)
                    metrics['recall'].append(float(lines[i + 2].split()[-1]) * 100)
                    metrics['precision'].append(float(lines[i + 3].split()[-1]) * 100)
                except (IndexError, ValueError) as e:
                    print(f"Warning: Malformed line in {file_path} - {e}")
                    continue

            if not all(metrics[key] for key in metrics):
                print(f"Warning: Incomplete statistics in {file_path}")
                continue

            means = np.array([np.mean(metrics[key]) for key in metrics])
            stds = np.array([np.std(metrics[key]) for key in metrics])

            if model_name not in grouped_results:
                grouped_results[model_name] = {}

            grouped_results[model_name][pooling] = (
                round(means[0], 2), round(stds[0], 1),
                round(means[1], 2), round(stds[1], 1),
                round(means[2], 2), round(stds[2], 1),
                round(means[3], 2), round(stds[3], 1)
            )
            models_used.add(model_name)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Print LaTeX formatted table
    for model_name, poolings in grouped_results.items():
        print("\n\n\\begin{table}[H]")
        print("\\centering")
        print("\\begin{tabular}{l|cccc}")
        print("\\hline")
        print("Pooling & Acc & F1 & Recall & Precision \\\\")
        print("\\Xhline{1pt}")

        for pooling in sorted(poolings.keys()):  # Ensure consistent order
            values = poolings[pooling]
            print(
                f"{pooling}  &  {values[0]}  $\\pm$  {values[1]}  &  {values[2]}  $\\pm$  {values[3]}  &  {values[4]}  $\\pm$  {values[5]}  &  {values[6]}  $\\pm$  {values[7]} \\\\")
        print("\\hline")

        print("\\end{tabular}")
        print(f"\\caption{{{model_name}}}")
        print("\\end{table}")


def load_config(path):
    return save_config(get_config(path))


def extract_best_results(log_path_server, keyword):
    """从服务器日志文件中提取最佳结果并汇总"""

    if not os.path.exists(log_path_server):
        print("Server log directory not found")
        return

    print(f"\n============= Cross-Validation {keyword} Summary =============")

    # 存储所有fold的结果
    all_fold_results = []

    # 遍历所有服务器日志文件
    server_log_files = [f for f in os.listdir(log_path_server) if f.startswith('server_fold_') and f.endswith('.log')]
    server_log_files.sort(key=lambda x: int(x.replace('server_fold_', '').replace('.log', '')))
    for filename in server_log_files:
        if filename.startswith('server_fold_') and filename.endswith('.log'):
            fold_num = int(filename.replace('server_fold_', '').replace('.log', ''))
            log_file_path = os.path.join(log_path_server, filename)

            if os.path.exists(log_file_path):
                best_f1 = None
                best_acc = None
                best_round = None

                with open(log_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # 查找Best Local Round行
                for i, line in enumerate(lines):
                    if keyword in line:
                        best_round = int(line.split()[-1])
                        # 读取下一行获取F1和Acc
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            # 解析 "F1=0.7710 Acc=0.7778" 格式
                            parts = next_line.split()
                            for part in parts:
                                if part.startswith('F1='):
                                    best_f1 = float(part.split('=')[1])
                                elif part.startswith('Acc='):
                                    best_acc = float(part.split('=')[1])
                                elif part.startswith('F1_test='):
                                    best_f1 = float(part.split('=')[1])
                                elif part.startswith('Acc_test='):
                                    best_acc = float(part.split('=')[1])

                            break

                if best_f1 is not None and best_acc is not None and best_round is not None:
                    all_fold_results.append({
                        'fold': fold_num, 'best_f1': best_f1, 'best_acc': best_acc, 'best_round': best_round
                    })
                    print(f"Fold {fold_num}: Best F1={best_f1:.4f}, Best Acc={best_acc:.4f}, Best Round={best_round}")
                else:
                    print(f"Fold {fold_num}: No best results found in log")

    # 计算平均值
    if all_fold_results:
        avg_f1 = sum(result['best_f1'] for result in all_fold_results) / len(all_fold_results)
        avg_acc = sum(result['best_acc'] for result in all_fold_results) / len(all_fold_results)
        avg_round = sum(result['best_round'] for result in all_fold_results) / len(all_fold_results)

        print(f"Average Best F1: {avg_f1:.4f}")
        print(f"Average Best Acc: {avg_acc:.4f}")
        print(f"Average Best Round: {avg_round:.2f}")
        # print(f"Results from {len(all_fold_results)} folds")

        # 保存汇总结果
        summary_file = os.path.join(log_path_server, f'../{keyword.lower()}_cross_fold_summary.tsv')
        with open(summary_file, 'w', encoding='utf-8') as f:
            # f.write("Fold\tBest_F1\tBest_Acc\tBest Round\n")
            f.write(
                "Fold 0 F1\tFold 0 ACC\tFold 1 F1\tFold 1 ACC\tFold 2 F1\tFold 2 ACC\tFold 3 F1\tFold 3 ACC\tFold 4 F1\tFold 4 ACC\tAVG F1\tAVG ACC\n")
            for result in all_fold_results:
                f.write(f"{result['best_f1']:.4f}\t{result['best_acc']:.4f}\t")
            f.write(f"{avg_f1:.4f}\t{avg_acc:.4f}\n")

            f.write("Best Round\t")
            for result in all_fold_results:
                f.write(f"{result['best_round']:.2f}\t")
            f.write(f"\nAVG Round\t{avg_round:.2f}\n")

        # 读取并打印保存的文件内容
        print(f"\nContent of {summary_file}:")
        with open(summary_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)

        print(f"Summary saved to {summary_file}")


def generate_tsv_summary(log_path):
    """
    从cross_fold_summary.txt生成cross_fold_summary.tsv，包含平均值，并将内容打印出来
    """
    log_file_path = os.path.join(log_path, 'cross_fold_summary.txt')
    if not os.path.exists(log_file_path):
        print(f"Warning: {log_file_path} not found.")
        return

    fold_results = {}

    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    current_fold = None
    for i, line in enumerate(lines):
        line = line.strip()

        # 查找Fold行
        if line.startswith('Fold'):
            # 提取fold编号
            current_fold = int(line.split()[1].rstrip(':'))
            fold_results[current_fold] = {'best_acc': None, 'best_f1': None}

        # 查找Best Value行 (改为best_acc) - 使用子串判断
        if 'Best Value =' in line and current_fold is not None:
            try:
                best_value_str = line.split('=')[1].strip()
                fold_results[current_fold]['best_acc'] = float(best_value_str)
            except (IndexError, ValueError) as e:
                print(f"Error parsing Best Value line: {line}, error: {e}")

        # 查找Best F1行
        if line.startswith('Best F1:') and current_fold is not None:
            try:
                best_f1_str = line.split(':')[1].strip()
                fold_results[current_fold]['best_f1'] = float(best_f1_str)
            except (IndexError, ValueError) as e:
                print(f"Error parsing Best F1 line: {line}, error: {e}")

    # 检查是否有缺失值并打印警告
    for fold, results in fold_results.items():
        if results['best_acc'] is None:
            print(f"Warning: best_acc is None for fold {fold}")
        if results['best_f1'] is None:
            print(f"Warning: best_f1 is None for fold {fold}")

    # 过滤掉有缺失值的fold
    valid_folds = {k: v for k, v in fold_results.items() if v['best_acc'] is not None and v['best_f1'] is not None}

    if not valid_folds:
        print("No valid fold results found!")
        return

    # 计算平均值
    f1_values = [valid_folds[fold]['best_f1'] for fold in sorted(valid_folds.keys())]
    acc_values = [valid_folds[fold]['best_acc'] for fold in sorted(valid_folds.keys())]

    avg_f1 = sum(f1_values) / len(f1_values)
    avg_acc = sum(acc_values) / len(acc_values)

    # 生成TSV内容
    headers = []
    values = []

    # 按照fold顺序排列
    for fold in sorted(valid_folds.keys()):
        result = valid_folds[fold]
        headers.append(f"fold {fold} f1")
        headers.append(f"fold {fold} acc")  # 改名
        # 保留四位小数
        values.append(f"{result['best_f1']:.4f}")
        values.append(f"{result['best_acc']:.4f}")

    # 添加平均值列
    headers.append("avg f1")
    headers.append("avg acc")
    # 保留四位小数
    values.append(f"{avg_f1:.4f}")
    values.append(f"{avg_acc:.4f}")

    # 生成TSV格式的字符串
    tsv_header = "\t".join(headers)
    tsv_values = "\t".join(values)
    tsv_content = f"{tsv_header}\n{tsv_values}"

    # 打印TSV内容
    print(tsv_content)

    # 保存TSV文件
    tsv_file_path = os.path.join(log_path, 'cross_fold_summary.tsv')
    with open(tsv_file_path, 'w') as f:
        f.write(tsv_content)

    print(f"TSV summary saved to {tsv_file_path}")
    return tsv_file_path


def get_model(config, device='cpu'):
    # if config.model.multimodality:
    if not config.model.single_modal:
        if "bicross" in config.model.fusion:
            model = BidirectionalCrossAttentionTransformerEncoder(config.model).to(device)
        elif "cross" in config.model.fusion:
            model = CrossAttentionTransformerEncoder(config.model).to(device)
        else:
            model = ElementWiseFusionEncoder(config.model).to(device)
    else:
        model = MyTransformerEncoder(config.model).to(device)

    return model

def get_model_parameters(model):
    """A getter method for the parameters of the model.

    Args:
        model (nn.Model): the model whose parameters we want to extract

    Returns:
        List[float]: a 1D representation of the model weights
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]