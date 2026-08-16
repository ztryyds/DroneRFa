import ast
import csv
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import glob
import joblib
from Dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import random
import numpy as np
import configparser
from Sup_TransResNet_model import TransResNet
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# ========= 配置 =========
config = configparser.ConfigParser()
config.read('config.ini')


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    print('Random seed :', seed)


my_seed = config.getint('general', 'my_seed')
set_seed(my_seed)

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
input_shape = config.get('general', 'input_shape')
input_shape = ast.literal_eval(input_shape)
path_to_files = config.get('general', 'path_to_files')
path_to_gan = f"{config.get('general', 'path_to_unknowns')}/*"
encoder = joblib.load(config.get('general', 'label_encoder_unknown'))
encoder_open = joblib.load(config.get('general', 'path_to_encoder_open'))
label_allow_list = encoder.classes_
open_list = encoder_open.classes_
batch_size = config.getint('general', 'batch_size_for_train')
num_classes = config.getint('general', 'num_classes')+1

max_epochs = config.getint('general', 'max_epochs')
max_lr = config.getfloat('general', 'max_lr')
model = TransResNet(input_shape,num_classes).to(device)

ckpt_path = config.get('general', 'path_to_model')
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model_state_dict"]

model_dict = model.state_dict()
filtered_dict = {}
for k, v in state_dict.items():
    if "fc" in k or "head" in k:
        continue  # 过滤后端层重新初始化
    if k in model_dict and model_dict[k].shape == v.shape:
        filtered_dict[k] = v

model_dict.update(filtered_dict)
model.load_state_dict(model_dict)
for name, p in model.named_parameters():
    if "fc" in name or "head" in name:
        p.requires_grad = True
    else:
        p.requires_grad = False
accumulation_steps = 1


# ========= DataLoader =========
def set_dataloader():
    train_file_paths, train_labels = [], []
    val_file_paths, val_labels = [], []
    unknown_file_paths, unknown_labels = [], []
    for parent_path in glob.glob(path_to_files):
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            # ===== unknown =====
            unknown_parent_path = os.path.join(parent_path, 'test/*')
            for files_path in glob.glob(unknown_parent_path):
                h5_file_path = os.path.join(files_path, '*.h5')
                for id, file_path in enumerate(glob.glob(h5_file_path)):
                    unknown_file_paths.append(file_path)
                    unknown_labels.append(label)
        else:
            train_parent_path = os.path.join(parent_path, 'train/*')
            for files_path in glob.glob(train_parent_path):
                h5_file_path = os.path.join(files_path, '*.h5')
                for id, file_path in enumerate(glob.glob(h5_file_path)):
                    train_file_paths.append(file_path)
                    train_labels.append(label)

            # ===== val =====
            val_parent_path = os.path.join(parent_path, 'test/*')
            for files_path in glob.glob(val_parent_path):
                h5_file_path = os.path.join(files_path, '*.h5')
                for id, file_path in enumerate(glob.glob(h5_file_path)):
                    val_file_paths.append(file_path)
                    val_labels.append(label)
    for parent_path in glob.glob(path_to_gan):
        # 判断类别是否参与训练
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue
        # 加载训练样本
        h5_file_path = os.path.join(parent_path, 'train/*/*.h5')
        for file_path in glob.glob(h5_file_path):
            train_file_paths.append(file_path)
            train_labels.append(label)
        # 加载验证样本
        # h5_file_path = os.path.join(parent_path, 'val/*/*.h5')
        # for file_path in glob.glob(h5_file_path):
        #     val_file_paths.append(file_path)
        #     val_labels.append(label)
    train_labels = encoder.transform(train_labels)
    val_labels = encoder.transform(val_labels)
    unknown_labels = encoder_open.transform(unknown_labels)
    print('train nums:', len(train_labels))
    print('val nums:', len(val_labels))
    print('unknown nums:', len(unknown_labels))

    train_set = Dataset(train_file_paths, train_labels)
    val_set = Dataset(val_file_paths, val_labels)
    unknown_set = Dataset(unknown_file_paths, unknown_labels)

    # 额外导出不乱序、不丢弃的训练推理流
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader_eval = DataLoader(train_set, batch_size=batch_size*2, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size*2, shuffle=False)
    unknown_loader = DataLoader(unknown_set, batch_size=batch_size*2, shuffle=False)
    return train_loader, train_loader_eval, val_loader, unknown_loader


# ========= 训练主循环 =========
if __name__ == '__main__':
    train_loader, train_loader_eval, val_loader, unknown_loader = set_dataloader()
    print('train batch nums:', len(train_loader))
    print('val batch nums:', len(val_loader))

    # 更换为正常的交叉熵损失
    criterion = nn.CrossEntropyLoss().to(device)

    # 优化器只留模型参数
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=max_lr,
        weight_decay=0
    )

    print("可训练参数列表：")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}")

    total_update_steps = (len(train_loader) // accumulation_steps) * max_epochs

    onecycle_scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_update_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=10
    )
    optimizer.zero_grad()

    # 日志落盘配置，增加自适应指标字段
    csv_file = open(config.get('general', 'path_to_log'), 'w', newline='')
    # 增加将要落盘的自适应拦截已知类指标 val_acc_under_thresh
    fieldnames = [
        'epoch',
        'train_loss',
        'train_accuracy',
        'val_loss',
        'val_accuracy',
        'val_acc_under_thresh',
        'total_unknown_accuracy',
        'auto_threshold',
        'open_set_auroc',
        'epoch_time',
        'lr'
    ]
    writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer_csv.writeheader()

    print(f"Train begin {accumulation_steps * batch_size}")

    for epoch in range(max_epochs):
        start_time = time.time()
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        train_loss, train_correct, train_total = 0, 0, 0
        lr = optimizer.param_groups[0]['lr']

        # ---- Train ----
        for i, (data, targets) in enumerate(train_loader):
            inputs = data.float().to(device)
            targets = targets.to(device)
            feature, feat, logits = model(inputs)
            loss = criterion(logits, targets)

            loss_scaled = loss / accumulation_steps
            loss_scaled.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                optimizer.step()
                onecycle_scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(logits.data, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)

        train_accuracy = train_correct / train_total
        train_loss = train_loss / train_total

        # 模型权重固化，切入静态数理评估模式
        model.eval()

        # ----------------------------------------------------------------
        # 2. 独立推理：用训练集在 FEAT 空间（64维）无损重构 MAV
        # ----------------------------------------------------------------
        train_feature_sums = torch.zeros(num_classes, 128).to(device)
        train_class_counts = torch.zeros(num_classes, 1).to(device)

        with torch.no_grad():
            for data, targets in train_loader_eval:
                inputs = data.float().to(device)
                targets = targets.to(device)
                feature, feat, logits = model(inputs)

                ones = torch.ones(targets.size(0), 1).to(device)
                train_class_counts.scatter_add_(0, targets.view(-1, 1), ones)
                target_expanded = targets.view(-1, 1).expand(-1, feat.size(1))
                train_feature_sums.scatter_add_(0, target_expanded, feat)

        train_feature_sums = train_feature_sums[:num_classes-1]
        train_class_counts = train_class_counts[:num_classes-1]
        MAV = train_feature_sums / (train_class_counts + 1e-10)
        norm_mav = F.normalize(MAV, p=2, dim=1, eps=1e-10)

        # ----------------------------------------------------------------
        # 3. 独立推理：Val 验证集多分类性能测试与得分收集 (收拢回 FEAT 维)
        # ----------------------------------------------------------------
        val_loss_total, val_correct, val_total = 0, 0, 0
        all_val_max_scores = []
        all_val_predictions_correct = []

        with torch.no_grad():
            for data, targets in val_loader:
                inputs = data.float().to(device)
                targets = targets.to(device)
                feature, feat, logits = model(inputs)
                v_loss = criterion(logits, targets)

                val_loss_total += v_loss.item() * targets.size(0)
                _, predicted = torch.max(logits.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

                norm_feat = F.normalize(feat, p=2, dim=1, eps=1e-10)
                sim_matrix = torch.mm(norm_feat, norm_mav.t())
                max_sims, max_idxs = torch.max(sim_matrix, dim=1)

                all_val_max_scores.extend(max_sims.cpu().numpy())
                all_val_predictions_correct.extend((max_idxs == targets).cpu().numpy())

        avg_val_loss = val_loss_total / val_total
        val_accuracy = val_correct / val_total

        # ----------------------------------------------------------------
        # 4. 独立推理：Unknown 未知类测试与得分收集 (收拢回 FEAT 维)
        # ----------------------------------------------------------------
        all_unknown_max_scores = []
        unknown_class_sims = defaultdict(list)
        with torch.no_grad():
            for data, targets in unknown_loader:
                inputs = data.float().to(device)

                feature, feat, logits = model(inputs)
                norm_feat = F.normalize(feat, p=2, dim=1, eps=1e-10)

                cos_sim_matrix = torch.mm(norm_feat, norm_mav.t())
                max_sims, _ = torch.max(cos_sim_matrix, dim=1)

                max_sims_cpu = max_sims.detach().cpu().numpy()
                targets_cpu = targets.detach().cpu().numpy()

                all_unknown_max_scores.extend(max_sims_cpu)
                for k in range(targets.size(0)):
                    unknown_class_sims[targets_cpu[k].item()].append(max_sims_cpu[k].item())

        # ----------------------------------------------------------------
        # 5. 数理推演：解算全局自适应黄金阈值与学术指标
        # ----------------------------------------------------------------
        val_scores_arr = np.array(all_val_max_scores)
        unk_scores_arr = np.array(all_unknown_max_scores)
        val_correct_mask = np.array(all_val_predictions_correct)
        threshold_grid = np.linspace(0.0, 1.0, 1000)
        best_threshold = 0.0
        min_diff = float('inf')

        eps = 1e-12
        total_unknown = len(unk_scores_arr) + eps
        total_known = len(val_scores_arr) + eps

        for th in threshold_grid:
            current_tur = np.sum(unk_scores_arr < th) / total_unknown
            current_tkr = np.sum((val_scores_arr >= th) & val_correct_mask) / total_known

            diff = np.abs(current_tkr - current_tur)
            if diff < min_diff:
                min_diff = diff
                best_threshold = th

        # 计算考虑黄金阈值拦截下的已知类最终实际准确率
        val_correct_under_thresh = np.sum((val_scores_arr >= best_threshold) & val_correct_mask)
        val_accuracy_under_thresh = val_correct_under_thresh / total_known

        # 💥 【核心修改】：利用最佳黄金阈值，解算总未知类拦截准确率 (Total U-Acc)
        unknown_rejected_mask = (
                unk_scores_arr < best_threshold
        )

        total_correct_unknown_rejections = np.sum(unknown_rejected_mask)
        total_unknown_accuracy = (
                total_correct_unknown_rejections / total_unknown
        )

        # 计算 Generalized Open-Set AUROC
        y_true_binary = np.concatenate([np.ones_like(val_scores_arr), np.zeros_like(unk_scores_arr)])
        y_true_binary[:len(val_scores_arr)] = np.where(val_correct_mask, 1, 0)
        all_scores_concat = np.concatenate([val_scores_arr, unk_scores_arr])

        try:
            generalized_open_set_auroc = roc_auc_score(y_true_binary, all_scores_concat)
        except ValueError:
            generalized_open_set_auroc = 0.5

        # 统计各个未知类拦截准确率 (U-Acc)
        unknown_info_list = []
        for class_id, sims_list in unknown_class_sims.items():
            sims_tensor = torch.tensor(sims_list)
            correct_preds = (sims_tensor < best_threshold).sum().item()
            acc_u = correct_preds / len(sims_list) if len(sims_list) > 0 else 0.0
            class_name = encoder_open.inverse_transform([class_id])[0]
            unknown_info_list.append(f"{class_name}: {acc_u * 100:.2f}%")

        unknown_info = ", ".join(unknown_info_list)
        epoch_time = time.time() - start_time

        # ----------------------------------------------------------------
        # 6. 日志输出与落盘 (已在 Val Acc(T) 后面紧跟 Total U-Acc 打印)
        # ----------------------------------------------------------------
        print(
            f"Epoch {epoch}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}, "
            f"Val Acc(T): {val_accuracy_under_thresh:.4f}, "
            f"Total U-Acc: {total_unknown_accuracy * 100:.2f}% | "
            f"Auto-Thresh: {best_threshold:.4f}, "
            f"Gen-AUROC: {generalized_open_set_auroc:.4f} | "
            f"U-Acc(<Thresh): [{unknown_info}], "
            f"Time: {epoch_time:.2f}s, "
            f"lr: {lr:.6f}"
        )

        writer_csv.writerow({
            'epoch': epoch, 'train_loss': train_loss,
            'train_accuracy': train_accuracy, 'val_loss': avg_val_loss, 'val_accuracy': val_accuracy,
            'val_acc_under_thresh': val_accuracy_under_thresh,
            'total_unknown_accuracy': total_unknown_accuracy,
            'auto_threshold': best_threshold, 'open_set_auroc': generalized_open_set_auroc,
            'epoch_time': epoch_time, 'lr': lr
        })
        csv_file.flush()

        # 模型存储逻辑
        save_dir = config.get('general', 'path_to_ce_gan_folder')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_filename = f"model_epoch_{epoch}.pth"
        save_path = os.path.join(save_dir, save_filename)

        checkpoint = {
            'model_state_dict': model.state_dict(),
        }
        torch.save(checkpoint, save_path)

    csv_file.close()