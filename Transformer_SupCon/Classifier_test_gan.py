import ast
import os
import torch
import glob
import joblib
import time
from Dataset import Dataset
from Sup_TransResNet_model import TransResNet, Classifier_gan
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns

import matplotlib

matplotlib.use('Agg')  # 使用Agg后端，这个后端适用于生成图像文件但不显示它们
import matplotlib.pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

# GPU configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

path_to_files = config.get('general', 'path_to_files')
path_to_gan = f"{config.get('general', 'path_to_unknowns')}/*"
encoder = joblib.load(config.get('general', 'path_to_encoder_gan'))
label_allow_list = encoder.classes_
batch_size = config.getint('general', 'batch_size_for_test')
num_classes = config.getint('general', 'num_classes') + 1
input_shape = config.get('general', 'input_shape')
input_shape = ast.literal_eval(input_shape)
conf = np.zeros((num_classes, num_classes))


def set_dataloader():
    """
    高效收集文件路径并构建 DataLoader
    """
    test_file_paths = []
    test_labels = []

    # 优化点：利用更深层的通配符直接提取文件，减少多重 Python 循环
    for parent_path in glob.glob(path_to_files):
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue
        for file_path in glob.glob(os.path.join(parent_path, 'test/*/*h5')):
            test_file_paths.append(file_path)
            test_labels.append(label)

    for parent_path in glob.glob(path_to_gan):
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue
        for file_path in glob.glob(os.path.join(parent_path, 'test/*/*h5')):
            test_file_paths.append(file_path)
            test_labels.append(label)

    test_labels = encoder.transform(test_labels)
    print('测试样本总数:', len(test_labels))

    test_set = Dataset(test_file_paths, test_labels)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return test_loader


if __name__ == '__main__':
    test_loader = set_dataloader()

    # 模型初始化与权重加载
    Classifier_model = Classifier_gan(num_classes).to(device)
    model_weights = torch.load(config.get('general', 'path_to_ce_gan'))
    Classifier_model.load_state_dict(model_weights)
    Classifier_model.eval()

    Sup_model = TransResNet(input_shape).to(device)
    model_weights = torch.load(config.get('general', 'path_to_model'))
    Sup_model.load_state_dict(model_weights)
    Sup_model.eval()

    # ========================================================
    # 🌟 核心技术点 1：GPU 算子冷启动预热（双模型联合拓扑对齐）
    # ========================================================
    print("正在进行 GPU 算子预热（双模型串联前向图加载）...")
    with torch.no_grad():
        # 从加载器中抽取一整批真实数据用来做底层 CUDNN 算子优化对齐
        for dummy_data, _ in test_loader:
            dummy_input = dummy_data.float().to(device)
            break
        # 连续迭代 10 次，使特征提取器与 GAN 分类器完全在 GPU 核心常驻
        for _ in range(10):
            _, dummy_feat = Sup_model(dummy_input)
            _ = Classifier_model(dummy_feat)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    print("预热完毕，正式进入流计算时延评测阶段...")

    # ========================================================
    # 🌟 核心技术点 2：高精度 CUDA 异步硬件事件计时器构建
    # ========================================================
    total_pure_inference_time_ms = 0.0
    total_processed_samples = 0

    # 创建 CUDA 硬件时间戳事件（启用高精度计时）
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    predict_label_list = []
    true_label_list = []

    # 开始遍历 Dataloader
    for data, targets in test_loader:
        actual_batch_size = data.size(0)
        data_gpu = data.to(device)  # 数据载入显存不计入严格时延

        # --------------------------------------------------------
        # 🌟 纯硬件计时区域：仅包裹双模型纯算力链路
        # --------------------------------------------------------
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.synchronize()  # 确保显存搬运及先前指令完全出栈

            start_event.record()  # 录入硬件时间戳：开始计算

            # 完整的串联网络推理图
            _, feature = Sup_model(data_gpu)
            softmax, logits = Classifier_model(feature)

            end_event.record()  # 录入硬件时间戳：结束计算

            if device.type == 'cuda':
                torch.cuda.synchronize()  # 强行阻塞异步机制，阻塞 CPU 直到 GPU 计算收尾

        # 提取当前批次的硬件净计算时间（单位：毫秒）
        batch_time_ms = start_event.elapsed_time(end_event) if device.type == 'cuda' else 0.0
        total_pure_inference_time_ms += batch_time_ms
        total_processed_samples += actual_batch_size
        # --------------------------------------------------------

        # 后续的矩阵运算、多卡后处理解包均在计时区域外进行，防止干扰
        _, predicted = torch.max(softmax.data, 1)

        # 优化点：暂存整型 Tensor 索引，不在批次循环内部频繁调用 CPU 密集的逆转换
        predict_label_list.append(predicted.cpu().numpy())
        true_label_list.append(targets.numpy())

    # ========================================================
    # 🌟 核心技术点 3：多维严格推理延迟报告打印
    # ========================================================
    avg_latency_per_batch_ms = total_pure_inference_time_ms / len(test_loader)
    avg_latency_per_sample_ms = total_pure_inference_time_ms / total_processed_samples
    throughput_fps = (total_processed_samples / total_pure_inference_time_ms) * 1000.0

    print("\n================== 双模型串联严格推理性能报告 ==================")
    print(f"严格评测样本总数 (Total Samples)         : {total_processed_samples}")
    print(f"总推理批次数 (Total Batches)             : {len(test_loader)} (Batch Size: {batch_size})")
    print(f"网络总计纯硬件算力耗时 (Total GPU Time)  : {total_pure_inference_time_ms / 1000.0:.4f} 秒")
    print(f"批次平均推理时延 (Avg Batch Latency)     : {avg_latency_per_batch_ms:.2f} 毫秒 (ms)")
    print(f"单个样本平均严格时延 (Avg Latency/Sample): {avg_latency_per_sample_ms:.4f} 毫秒 (ms)")
    print(f"系统吞吐量速度 (Throughput FPS)           : {throughput_fps:.2f} 帧/秒")
    print("===============================================================\n")

    # ========================================================
    # 4. 后处理与报告生成（混淆矩阵与最终指标映射）
    # ========================================================
    # 一次性在循环外部合并并逆转换标签字符串，最大化 CPU 利用率
    predict_idx = np.concatenate(predict_label_list, axis=0)
    true_idx = np.concatenate(true_label_list, axis=0)

    predict_label_str_list = encoder.inverse_transform(predict_idx)
    true_label_str_list = encoder.inverse_transform(true_idx)

    # 混淆矩阵映射
    conf += confusion_matrix(true_label_str_list, predict_label_str_list, labels=encoder.classes_)
    conf_normalized = conf / conf.sum(axis=1)[:, np.newaxis]

    save_folder = config.get('general', 'folder_to_result')
    os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_normalized, annot=True, cmap='Blues', fmt='.2f',
                xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')

    filename = os.path.join(save_folder, config.get('general', 'path_test_result_gan'))
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # 打印标准分类科学报告
    report = classification_report(true_label_str_list, predict_label_str_list, target_names=encoder.classes_, digits=4)
    print("Classification Report:")
    print(report)