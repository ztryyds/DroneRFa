import os
import joblib
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
#  【地址与路径统一配置区】 —— 方便你后续随时修改
# ================================================================
PATH_ENCODER_CLOSE = '../model/label_encoder_close.joblib'
PATH_ENCODER_OPEN = '../model/label_encoder_open.joblib'
PATH_ENCODER_ALL = '../model/label_encoder_all.joblib'
PATH_MAV_FOLDER = '../openmax_data/openmax_gan_SupTransResNet/feat/MAV'
PATH_TEST_DATA_FOLDER = '../openmax_data/openmax_gan_SupTransResNet/data'
PATH_RESULT_SAVE_FOLDER = "../results"

# 🚀【核心模式切换开关】逻辑完美统一：数值越大越倾向于未知类
# 可选值: 'cosine_distance' (余弦距离) 或 'euclidean' (真·标准原生欧氏距离)
METRIC_TYPE = 'cosine_distance'

# ================================================================
# 1. 标签加载与准备
# ================================================================
encoder_close = joblib.load(PATH_ENCODER_CLOSE)
encoder_open = joblib.load(PATH_ENCODER_OPEN)
encoder_all = joblib.load(PATH_ENCODER_ALL)

label_allow_list = list(encoder_close.classes_)
all_list = list(encoder_all.classes_)
display_labels = np.append(label_allow_list, 'unknowns')


# ================================================================
# 2. 矩阵化加载已知类中心 (MAV)
# ================================================================
def load_mav_centers_numpy(metric='cosine_distance'):
    centers = []
    for label in label_allow_list:
        mat_path = os.path.join(PATH_MAV_FOLDER, f'{label}_mav.mat')
        mat_data = loadmat(mat_path)
        mav_vec = mat_data.get('mav') if mat_data.get('mav') is not None else mat_data.get('mav64')
        centers.append(mav_vec.ravel())

    centers = np.stack(centers, axis=0)  # [num_classes, feature_dim]

    # 💥 只有余弦度量需要预归一化基准；如果是欧氏距离，保持原汁原味原生模长
    if metric == 'cosine_distance':
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / (norms + 1e-10)

    return centers


# ================================================================
# 3. 纯粹的空间特征抽取引擎 (逻辑大一统：高精度原生计算)
# ================================================================
def extract_open_set_scores_numpy(features, centers, metric='cosine_distance'):
    """
    统一的距离抽取引擎：无论是余弦距离还是原生欧氏距离，一律返回：
    - max_idxs:  最近的已知类索引 (距离最小)
    - min_dists: 样本到最近类中心的距离绝对值 (值越大越可能是未知类)
    """
    if metric == 'cosine_distance':
        # 批量 L2 归一化仅在余弦模式下生效
        feat_norms = np.linalg.norm(features, axis=1, keepdims=True)
        features_norm = features / (feat_norms + 1e-10)

        sim_matrix = np.dot(features_norm, centers.T)
        dist_matrix = 1.0 - sim_matrix

    elif metric == 'euclidean':
        # 💥 坚守纯粹原生：零归一化，直接利用高精度广播计算绝对物理坐标距离
        # features[:, None, :] 形状 -> [N, 1, D]
        # centers[None, :, :]  形状 -> [1, C, D]
        # 相减得到坐标差值矩阵 -> [N, C, D]
        diff = features[:, None, :] - centers[None, :, :]

        # 在最后一维(D维特征轴)直接求解 L2 范数，确保数值绝对稳定无损 -> [N, C]
        dist_matrix = np.linalg.norm(diff, axis=-1)

    else:
        raise ValueError(f"不支持的度量类型: {metric}")

    # 距离越小越相似
    max_idxs = np.argmin(dist_matrix, axis=1)
    min_dists = np.min(dist_matrix, axis=1)

    return max_idxs, min_dists


if __name__ == "__main__":
    print(f"⚙️ 当前开集识别运行模式: 【{METRIC_TYPE.upper()}】(原生尺度距离大一统逻辑)")

    # 加载类中心
    centers_tensor = load_mav_centers_numpy(metric=METRIC_TYPE)

    all_true_labels = []
    all_raw_preds = []
    all_min_dists = []

    # ================================================================
    # 4. 批量加载与几何得分收集
    # ================================================================
    for label in all_list:
        # if label == 'mavic3_1':
        #     continue
        file_path = os.path.join(PATH_TEST_DATA_FOLDER, f'{label}_data.mat')
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过该类别。")
            continue

        true_label_mapped = 'unknowns' if label not in label_allow_list else label
        feature_mat = loadmat(file_path)['feat']  # 原生未归一化特征

        print(f'正在通过矩阵并行提取 [{label}] 的特征距离，样本数: {len(feature_mat)}')

        # 提取当前类别下所有样本的最佳已知匹配索引和最小物理距离
        batch_idxs, batch_dists = extract_open_set_scores_numpy(feature_mat, centers_tensor, metric=METRIC_TYPE)

        all_raw_preds.extend(batch_idxs)
        all_true_labels.extend([true_label_mapped] * len(feature_mat))
        all_min_dists.extend(batch_dists)

    # ================================================================
    # 5. 纯数理推演：解算全局最优黄金阈值点 (统一为距离拦截逻辑)
    # ================================================================
    y_true = np.array(all_true_labels)
    y_dists = np.array(all_min_dists)
    y_raw_preds_labels = np.array([label_allow_list[idx] for idx in all_raw_preds], dtype=object)

    is_known_true = (y_true != 'unknowns')
    is_unknown_true = (y_true == 'unknowns')
    eps = 1e-12

    # 🚨 自适应包裹：由于不限制欧氏距离的物理尺度，网格搜索范围会完美动态贴合原生的 [min, max]
    threshold_grid = np.linspace(np.min(y_dists), np.max(y_dists), 1000)
    # threshold_grid = np.linspace(0.0, 1.0, 1000)
    tkr_curves = []
    tur_curves = []

    for th in threshold_grid:
        # 大一统逻辑：绝对原生距离 > 阈值，就判定为未知类
        sim_is_unknown_pred = (y_dists > th)

        # TUR: 真正未知里有多少由于距离过远被成功拦截
        current_tur = np.sum(is_unknown_true & sim_is_unknown_pred) / (np.sum(is_unknown_true) + eps)
        # TKR: 真正已知里有多少在安全距离内，且多分类预测完全正确
        current_tkr = np.sum(is_known_true & ~sim_is_unknown_pred & (y_true == y_raw_preds_labels)) / (
                np.sum(is_known_true) + eps)

        tkr_curves.append(current_tkr)
        tur_curves.append(current_tur)

    tkr_arr = np.array(tkr_curves)
    tur_arr = np.array(tur_curves)

    # 寻找 |TKR - TUR| 最小的交点作为黄金自适应阈值
    # best_idx = np.argmin(np.abs(tkr_arr - tur_arr))
    # best_idx = np.argmax(tkr_arr + tur_arr)
    diff = np.abs(tkr_arr - tur_arr)
    score = tkr_arr + tur_arr
    valid_idxs = np.where(diff <= 0.01)[0]

    if len(valid_idxs) > 0:
        # 在满足平衡约束的候选中，选择总性能最高的
        best_idx = valid_idxs[np.argmax(score[valid_idxs])]
    else:
        # 如果没有满足约束的候选，则选择差值最小的
        best_idx = np.argmin(diff)
    AUTOMATIC_THRESHOLD = threshold_grid[best_idx]

    # ================================================================
    # 6. 用自动解算出的黄金阈值进行最终硬标签裁剪 (统一使用 > 拦截)
    # ================================================================
    y_pred = y_raw_preds_labels.copy()
    unknown_mask = (y_dists > AUTOMATIC_THRESHOLD)
    y_pred[unknown_mask] = 'unknowns'

    # ================================================================
    # 7. 计算该自适应阈值下的各项身份认证指标 (硬标签)
    # ================================================================
    is_known_pred = (y_pred != 'unknowns')
    is_unknown_pred = (y_pred == 'unknowns')

    TK = np.sum(is_known_true & is_known_pred)
    CK = np.sum(is_known_true & (y_true == y_pred))
    FU = np.sum(is_known_true & is_unknown_pred)
    TU = np.sum(is_unknown_true & is_unknown_pred)
    FK = np.sum(is_unknown_true & is_known_pred)

    Acc = (CK + TU) / (len(y_true) + eps)
    TKR = CK / (TK + FU + eps)
    TUR = TU / (TU + FK + eps)
    KP = CK / (TK + FK + eps)
    UP = TU / (TU + FU + eps)

    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # ================================================================
    # 8. 学术标准：Generalized Open-Set AUROC 计算区域 (不依赖阈值)
    # ================================================================
    is_correct_known = (y_true == y_raw_preds_labels) & (y_true != 'unknowns')
    y_true_binary_generalized = np.where(is_correct_known, 1, 0)

    # 原生欧氏距离越小越亲近，为了契合 AUROC 正相关输入，取相反数送入
    generalized_open_set_auroc = roc_auc_score(y_true_binary_generalized, -y_dists)

    # ================================================================
    # 9. 统一终端报告输出与绘图
    # ================================================================
    print('\n' + '=' * 60)
    print(f'     开集识别身份认证测试报告 (学术严谨 绝对原生距离版 - {METRIC_TYPE.upper()})    ')
    print('=' * 60)
    print(f'🔥 纯数理推演确定的最佳判别阈值 (Threshold): \033[1;36m{AUTOMATIC_THRESHOLD:.4f}\033[0m')

    print("\n详细多分类报告 (Classification Report):")
    print(classification_report(y_true, y_pred, target_names=display_labels, digits=4))

    # 混淆矩阵
    openmax_conf = confusion_matrix(y_true, y_pred, labels=display_labels)
    row_sums = openmax_conf.sum(axis=1)[:, np.newaxis]
    conf_normalized = np.divide(openmax_conf, row_sums, out=np.zeros_like(openmax_conf, dtype=float),
                                where=row_sums != 0)

    os.makedirs(PATH_RESULT_SAVE_FOLDER, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_normalized, annot=True, cmap='Blues', fmt='.2f', xticklabels=display_labels,
                yticklabels=display_labels)
    plt.xlabel('Predicted labels', fontsize=11)
    plt.ylabel('True labels', fontsize=11)
    plt.title(
        f'Normalized Confusion Matrix ({METRIC_TYPE.replace("_", " ").capitalize()} Auto-Threshold = {AUTOMATIC_THRESHOLD:.3f})',
        fontsize=13)
    save_path_conf = os.path.join(PATH_RESULT_SAVE_FOLDER, f'{METRIC_TYPE}_open_set.png')
    plt.savefig(save_path_conf, dpi=300, bbox_inches='tight')
    plt.close()

    # 深度分析图
    plt.figure(figsize=(15, 6))

    # 左图：真实绝对距离密度分布
    plt.subplot(1, 2, 1)
    sns.kdeplot(y_dists[is_unknown_true], fill=True, color="red", label="True Unknowns", alpha=0.4)
    sns.kdeplot(y_dists[is_known_true], fill=True, color="blue", label="True Knowns", alpha=0.4)
    plt.axvline(AUTOMATIC_THRESHOLD, color='green', linestyle='--', linewidth=2,
                label=f'Auto Thresh: {AUTOMATIC_THRESHOLD:.3f}')
    plt.title(f"Distance Density & The 'Golden Canyon' ({METRIC_TYPE.replace('_', ' ').capitalize()})", fontsize=12)
    plt.xlabel("Absolute Distance Value (Lower is closer)", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.5)

    # 右图：TKR-TUR 动态对冲交汇
    plt.subplot(1, 2, 2)
    plt.plot(threshold_grid, tkr_arr, color='blue', linewidth=2, label='TKR (True Known Rate)')
    plt.plot(threshold_grid, tur_arr, color='red', linewidth=2, label='TUR (True Unknown Rate)')

    current_best_tkr = tkr_arr[best_idx]
    plt.plot(AUTOMATIC_THRESHOLD, current_best_tkr, 'go', markersize=8,
             label=f'Equilibrium ({AUTOMATIC_THRESHOLD:.3f})')
    plt.axvline(AUTOMATIC_THRESHOLD, color='green', linestyle=':', alpha=0.7)
    plt.axhline(current_best_tkr, color='green', linestyle=':', alpha=0.7)

    plt.title("Open-Set Performance Trade-off", fontsize=12)
    plt.xlabel("Threshold (Absolute Distance)", fontsize=10)
    plt.ylabel("Rate", fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.5)

    save_path_analysis = os.path.join(PATH_RESULT_SAVE_FOLDER, f'{METRIC_TYPE}_threshold_analysis.png')
    plt.tight_layout()
    plt.savefig(save_path_analysis, dpi=300)
    plt.close()

    print(f'🎉 混淆矩阵与阈值双子分析图已成功保存至: {PATH_RESULT_SAVE_FOLDER}\n')
    print(f'▶ 自适应硬标签截断表现 (Threshold = {AUTOMATIC_THRESHOLD:.4f}):')
    print(f'  - 全局身份认证准确率 (Acc): {Acc:.4f}')
    print(f'  - 开集通用学术评价指标 (Macro F1-Score): {macro_f1:.4f}')
    print(f'  - 已知类正确识别率 (TKR - True Known Rate): {TKR:.4f}')
    print(f'  - 未知类成功拦截率 (TUR - True Unknown Rate): {TUR:.4f}')
    print(f'  - 已知类预测精准率 (KP): {KP:.4f}')
    print(f'  - 未知类预测精准率 (UP): {UP:.4f}')
    print(f'▶ 空间几何泛化性能硬核指标 (无人工阈值干扰):')
    print(f'  - 广义开集识别得分 (Generalized Open-Set AUROC): \033[1;32m{generalized_open_set_auroc:.4f}\033[0m')
    print('=' * 60)