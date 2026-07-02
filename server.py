import shutup

shutup.please()
import argparse
import flwr as fl
import numpy as np
from collections import OrderedDict
import logging
from utils import load_config, get_model, get_model_parameters

logging.getLogger("flwr").setLevel(logging.INFO)


class MetricsLogger:
    """全局聚合每一轮指标，训练结束后汇总"""

    def __init__(self):
        self.all_metrics = []
        self.local_metrics = []
        self.round_counter = 1  # 记录当前轮次

    def aggregate_fit(self, metrics):
        """
        聚合每轮训练的指标（如 loss, accuracy）
        metrics: List[(num_examples, {metric_dict})]
        """
        print(f"\n[Server] 📈 Round {self.round_counter} Training Metrics from Clients:")

        # 存储客户端指标
        client_id_to_metrics = {}
        all_accuracies = []
        all_f1s = []

        # 收集每个客户端的指标
        for num_examples, metric_dict in metrics:
            acc = metric_dict.get("accuracy", None)
            f1 = metric_dict.get("f1", None)
            client_id = metric_dict.get("client_id", None)

            if client_id is None:
                continue  # 忽略没有 client_id 的数据

            # 直接保存原始 metric_dict
            client_id_to_metrics[client_id] = {"f1": f1, "accuracy": acc, }

            # 收集用于平均的指标
            if acc is not None:
                all_accuracies.append(acc)
            if f1 is not None:
                all_f1s.append(f1)

        # 按 client_id 排序输出
        for client_id in sorted(client_id_to_metrics.keys()):
            metrics_str = ", ".join(
                f"{k}: {v:.4f}" for k, v in client_id_to_metrics[client_id].items() if v is not None)
            print(f"  Client {client_id}: {{{metrics_str}}}")

        # 计算平均值
        mean_accuracy = float(np.mean(all_accuracies)) if all_accuracies else 0.0
        mean_f1 = float(np.mean(all_f1s)) if all_f1s else 0.0

        print(f"  Average:  {{f1: {mean_f1:.4f}, accuracy: {mean_accuracy:.4f}}}")
        # 保存本轮指标
        round_metric = {"client_f1s": [client_id_to_metrics[cid]["f1"] for cid in sorted(client_id_to_metrics.keys())],
                        "client_accs": [client_id_to_metrics[cid]["accuracy"] for cid in
                                        sorted(client_id_to_metrics.keys())], "mean_f1": mean_f1,
                        "mean_acc": mean_accuracy, }

        self.local_metrics.append(round_metric)
        # self.round_counter += 1  # 轮次递增

        # return round_metric
        return {}

    def aggregate_evaluate(self, metrics):
        # metrics: List[(num_examples, {metric_dict})]

        # 打印每个客户端的详细指标
        client_accuracies = []
        client_f1s = []
        client_losses = []

        for idx, (num_examples, metric_dict) in enumerate(metrics):
            acc = metric_dict.get("accuracy", None)
            f1 = metric_dict.get("f1", None)
            client_id = metric_dict.get("client_id", None)

            # loss = metric_dict.get("loss", None)
            # 构造有序字符串
            # 汇聚之后，每个客户端的模型参数是一致的，验证集也是一致的，因此三个客户端的指标是一样的
            if idx == 0:
                ordered_str = "{" + ", ".join(
                    f"'{k}': {v:.4f}" for k, v in [("f1", f1), ("accuracy", acc),  # ("loss", loss)
                                                   ] if v is not None) + "}"
                print(f"[Server] 📊 Round {self.round_counter} Evaluation Metrics from Clients: AVG: {ordered_str}")

            if "accuracy" in metric_dict:
                client_accuracies.append((num_examples, metric_dict["accuracy"]))
            if "f1" in metric_dict:
                client_f1s.append((num_examples, metric_dict[
                    "f1"]))  # if "loss" in metric_dict:  #     client_losses.append((num_examples, metric_dict["loss"]))

        # 简单平均（可选：加权平均）
        accuracies = [m["accuracy"] for _, m in metrics if "accuracy" in m]
        f1s = [m["f1"] for _, m in metrics if "f1" in m]
        # losses = [m["loss"] for _, m in metrics if "loss" in m]

        mean_accuracy = float(np.mean(accuracies)) if accuracies else 0
        mean_f1 = float(np.mean(f1s)) if f1s else 0
        # mean_loss = float(np.mean(losses)) if losses else 0

        # 保存本轮指标
        round_metric = {"mean_f1": mean_f1, "mean_acc": mean_accuracy, }

        self.all_metrics.append(round_metric)
        self.round_counter += 1  # 轮次递增

        # return round_metric
        return {}

    def summary(self):

        num_clients = len(self.local_metrics[0]['client_accs'])
        client_best_accs = []
        client_best_f1s = []

        for client_idx in range(num_clients):
            # 收集该客户端所有轮次的准确率和F1分数
            client_accs = [m['client_accs'][client_idx] for m in self.local_metrics]
            client_f1s = [m['client_f1s'][client_idx] for m in self.local_metrics]

            # 找到最佳准确率及其对应的F1分数（确保来自同一轮次）
            # best_idx = np.argmax(client_accs)
            best_idx = np.argmax(client_f1s)
            best_acc = client_accs[best_idx]
            best_f1 = client_f1s[best_idx]  # 同一轮次的F1分数

            client_best_accs.append(best_acc)
            client_best_f1s.append(best_f1)

        # 输出每个客户端的最佳性能
        print(f"\n🏆 每个客户端的最佳本地性能:")
        for client_idx in range(num_clients):
            print(f"  Client {client_idx}: "
                  f"{{Best F1: {client_best_f1s[client_idx]:.4f}, "
                  f"Best Acc: {client_best_accs[client_idx]:.4f}}}")

        avg_best_acc = np.mean(client_best_accs)
        avg_best_f1 = np.mean(client_best_f1s)

        print(f"\n🏆 Best Local Round(by F1): 0")
        print(f"F1={avg_best_f1:.4f} "  
              f"Acc={avg_best_acc:.4f}\n")

        if self.all_metrics:
            best_round_info = max([(round_idx, m) for round_idx, m in enumerate(self.all_metrics, 1)],
                                  key=lambda x: x[1]['mean_f1'])


            best_round_num = best_round_info[0]  # 最佳轮次编号
            best_round_metrics = best_round_info[1]  # 最佳轮次的指标
            print(f"🏆 Best Federal Round (by F1): {best_round_num}")
            print(f"F1={best_round_metrics['mean_f1']:.4f} "
                  f"Acc={best_round_metrics['mean_acc']:.4f} \n")

            # 上面两个指标分别是fit后（汇聚前，以local表示）的指标与evaluate（汇聚后，以federal表示）的指标，这里是要取每个客户端的最大值，然后计算三个客户端的平均值，以global表示
            best_f1_client_global = []
            best_acc_client_global = []
            for client_idx in range(num_clients):
                best_f1_client = max(client_best_f1s[client_idx], best_round_metrics['mean_f1'])
                best_acc_client = max(client_best_accs[client_idx], best_round_metrics['mean_acc'])
                best_f1_client_global.append(best_f1_client)
                best_acc_client_global.append(best_acc_client)

            best_f1_global = np.mean(best_f1_client_global)
            best_acc_global = np.mean(best_acc_client_global)

            print(f"\n🏆 Best Global Round(by F1): 0")
            print(f"F1={best_f1_global :.4f} "
                  f"Acc={best_acc_global:.4f}")



class LocalTrainingStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        print(f"\n[Server] 📈 Round {server_round} Training Metrics:")

        client_id_to_metrics = {}
        all_accuracies = []
        all_f1s = []

        for client, fit_res in results:
            metrics = fit_res.metrics
            acc = metrics.get("accuracy", None)
            f1 = metrics.get("f1", None)
            loss = metrics.get("loss", None)
            client_id = metrics.get("client_id", None)  # 👈 使用 client_id

            if acc is not None:
                all_accuracies.append(acc)
            if f1 is not None:
                all_f1s.append(f1)

            ordered_str = "{" + ", ".join(
                f"'{k}': {v:.4f}" for k, v in [("f1", f1), ("acc", acc), ] if v is not None) + "}"

            client_id_to_metrics[client_id] = ordered_str  # 👈 用 client_id 作为 key

        # 按照 client_id 排序输出
        for client_id in sorted(client_id_to_metrics.keys()):
            print(f"  Client {client_id}: {client_id_to_metrics[client_id]}")

        # 输出平均值
        mean_accuracy = float(np.mean(all_accuracies)) if all_accuracies else 0.0
        mean_f1 = float(np.mean(all_f1s)) if all_f1s else 0.0
        print(f"[Server] 📊 Round {server_round} Mean Metrics Across Clients:")
        print(f"  {{Mean F1: {mean_f1:.4f}, Mean ACC: {mean_accuracy:.4f}}}\n")

        print(f"🏆 Best Local Round (by F1): 1")
        print(f"F1={mean_f1:.4f} "
              f"Acc={mean_accuracy:.4f} \n")

        return None, {}  # 不更新全局参数


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    logger = MetricsLogger()
    num_clients = int(config.train.toDict().get('num_clients', 3))
    client_requirements = {
        "min_fit_clients": num_clients,
        "min_evaluate_clients": num_clients,
        "min_available_clients": num_clients,
    }
    if config.train.learning_paradigm == 'fl':
        initial_model = get_model(config)
        initial_params = get_model_parameters(initial_model)
        initial_params = fl.common.ndarrays_to_parameters(initial_params)
        strategy_name = config.train.strategy.lower()
        # FedAvg
        if strategy_name == 'fedavg':
            strategy = fl.server.strategy.FedAvg(fit_metrics_aggregation_fn=logger.aggregate_fit,
                                                 evaluate_metrics_aggregation_fn=logger.aggregate_evaluate,
                                                 **client_requirements,
                                                 initial_parameters=initial_params,
                                                 )
        elif strategy_name == 'fedadam':
            strategy = fl.server.strategy.FedAdam(fit_metrics_aggregation_fn=logger.aggregate_fit,
                                                  evaluate_metrics_aggregation_fn=logger.aggregate_evaluate,
                                                  initial_parameters=initial_params,  # 添加必需的初始参数
                                                  **client_requirements,
                                                  eta=0.0002,  # 学习率参数名为 eta 而不是 learning_rate
                                                  beta_1=0.9,  # Adam 优化器的 beta_1
                                                  beta_2=0.99,  # Adam 优化器的 beta_2
                                                  tau=1e-5  # 防止除零错误的小常数
                                                  )
        elif strategy_name == 'fedadagrad':
            strategy = fl.server.strategy.FedAdagrad(fit_metrics_aggregation_fn=logger.aggregate_fit,
                                                     evaluate_metrics_aggregation_fn=logger.aggregate_evaluate,
                                                     **client_requirements,
                                                     eta=0.0005, tau=1e-5, initial_parameters=initial_params,  # 添加必需的初始参数
                                                     )
        elif strategy_name == 'fedyogi':
            strategy = fl.server.strategy.FedYogi(fit_metrics_aggregation_fn=logger.aggregate_fit,
                                                  initial_parameters=initial_params,  # 添加必需的初始参数
                                                  evaluate_metrics_aggregation_fn=logger.aggregate_evaluate, eta=0.0008,
                                                  **client_requirements,
                                                  beta_1=0.9, beta_2=0.99, tau=1e-5)
        elif strategy_name == 'fedprox':
            strategy = fl.server.strategy.FedProx(fit_metrics_aggregation_fn=logger.aggregate_fit,
                                                  evaluate_metrics_aggregation_fn=logger.aggregate_evaluate,
                                                  initial_parameters=initial_params,  # 添加必需的初始参数
                                                  **client_requirements,
                                                  proximal_mu=0.3# 控制正则项强度的超参数
                                                  )
        fl.server.start_server(server_address="localhost:8080", config=fl.server.ServerConfig(num_rounds=30),
                               strategy=strategy)
        logger.summary()
    else:
        strategy = LocalTrainingStrategy(fit_metrics_aggregation_fn=logger.aggregate_fit, **client_requirements)
        fl.server.start_server(server_address="localhost:8080", config=fl.server.ServerConfig(num_rounds=1),
                               strategy=strategy)  # 配置 FedAvg，并接入我们自定义的聚合指标函数
