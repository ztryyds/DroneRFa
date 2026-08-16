import os
import pickle
import joblib
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib

matplotlib.use('Agg')  # 使用Agg后端，这个后端适用于生成图像文件但不显示它们
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as spd

open_list = ['T0001', 'T10001', 'T10011', 'T11000', 'T10110']

openmax_conf = np.zeros((21, 21))
NCHANNELS = 1
NCLASSES = 21
encoder = joblib.load('../model/label_encoder_unknown.joblib')
label_allow_list = list(encoder.classes_)
all_list = label_allow_list + open_list
# all_list = np.append(label_allow_list, 'T10001')
all_list1 = label_allow_list
# eucos euclidean cosine
distance_type = 'euclidean'
alpharank = 3
TK = 0
TU = 0
FK = 0
FU = 0
CK = 0


def query_weibull(label_name, weibull_model):
    category_weibull = []
    category_weibull += [weibull_model[label_name]['mean_vec']]
    category_weibull += [weibull_model[label_name]['distances']]
    category_weibull += [weibull_model[label_name]['weibull_model']]
    return category_weibull


def computeOpenMaxProbability(openmax_fc8, openmax_score_u):
    prob_scores, prob_unknowns = [], []
    for channel in range(NCHANNELS):
        channel_scores, channel_unknowns = [], []
        for category in range(NCLASSES):
            channel_scores += [np.exp(openmax_fc8[channel, category])]

        total_denominator = np.sum(np.exp(openmax_fc8[channel, :])) + np.exp(np.sum(openmax_score_u[channel, :]))
        prob_scores += [channel_scores / total_denominator]
        prob_unknowns += [np.exp(np.sum(openmax_score_u[channel, :])) / total_denominator]

    prob_scores = np.asarray(prob_scores)
    prob_unknowns = np.asarray(prob_unknowns)

    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    assert len(modified_scores) == NCLASSES + 1
    return modified_scores


def recalibrate_scores(weibull_model, label_list, softmax, logits, feature):
    """
    使用OpenMax重新校准分数
    """
    ranked_list = softmax.argsort().ravel()[::-1]
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    # alpha_weights = [1, 0.5, 0.5]
    ranked_alpha = np.zeros(21)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]
    openmax_fc8, openmax_score_u = [], []
    for channel in range(NCHANNELS):
        channel_feature = feature
        channel_logits = logits
        openmax_fc8_channel = []
        openmax_fc8_unknown = []
        for label_id in range(NCLASSES):
            if label_id == 20:
                openmax_fc8_channel += [channel_logits[label_id] * (1 - ranked_alpha[label_id])]
                openmax_fc8_unknown += [channel_logits[label_id] * ranked_alpha[label_id]]
            else:
                category_weibull = query_weibull(label_list[label_id], weibull_model)
                if distance_type == 'euclidean':
                    channel_distance = spd.euclidean(channel_feature, category_weibull[0][0])
                else:
                    print('distance_type error', distance_type)
                    return
                wscore = category_weibull[2][channel].w_score(channel_distance)
                modified_fc8_score = channel_logits[label_id] * (1 - wscore * ranked_alpha[label_id])
                openmax_fc8_channel += [modified_fc8_score]
                openmax_fc8_unknown += [channel_logits[label_id] - modified_fc8_score]
        openmax_fc8 += [openmax_fc8_channel]
        openmax_score_u += [openmax_fc8_unknown]
    openmax_fc8 = np.array(openmax_fc8)
    openmax_score_u = np.array(openmax_score_u)

    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u)

    return np.array(openmax_probab)


def apply_openmax_to_sample(softmax, logits, feature):
    with open('../model/weibull_model_gan.pkl', 'rb') as f:
        weibull_model = pickle.load(f)

    openmax = recalibrate_scores(weibull_model, label_allow_list, softmax, logits, feature)
    return openmax


if __name__ == "__main__":
    predict_label_list = []
    true_label_list = []
    for label in all_list:
        if label == 'unknowns':
            continue
        file_path = f'../openmax_data/openmax_gan_SupTransResNet/data/{label}_data.mat'
        if label not in label_allow_list:
            label = 'unknowns'
            softmax_list = loadmat(file_path)['logits']
            logits_list = loadmat(file_path)['logits']
            feature_list = loadmat(file_path)['logits']
        else:
            softmax_list = loadmat(file_path)['logits']
            logits_list = loadmat(file_path)['logits']
            feature_list = loadmat(file_path)['logits']
        for i, softmax in enumerate(softmax_list):
            logits = logits_list[i]

            logits_min = np.min(logits)
            logits_max = np.max(logits)
            logits = (logits - logits_min) / (logits_max - logits_min)
            feature = feature_list[i]
            openmax = apply_openmax_to_sample(softmax, logits, feature)
            predict_label = np.argmax(openmax)
            if predict_label == 21:
                predict_label = 'unknowns'
            else:
                predict_label = encoder.inverse_transform([predict_label])[0]
            true_label = label
            predict_label_list.append(predict_label)
            true_label_list.append(true_label)

            if true_label != 'unknowns':
                # 已知类别
                if predict_label != 'unknowns':
                    TK += 1  # 正确识别为已知
                    if predict_label == true_label:
                        CK += 1  # 正确分类到确切已知类别
                else:
                    FU += 1  # 错误识别为未知
            else:
                # 未知类别
                if predict_label == 'unknowns':
                    TU += 1  # 正确识别为未知
                else:
                    FK += 1  # 错误识别为已知

            openmax_conf += confusion_matrix([true_label], [predict_label], labels=all_list1)

    # openmax处理
    conf_normalized = openmax_conf / openmax_conf.sum(axis=1)[:, np.newaxis]
    save_folder = "../results"
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_normalized, annot=True, cmap='Blues', fmt='.2f',
                xticklabels=all_list1, yticklabels=all_list1)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    filename = os.path.join(save_folder, 'openmax_gan.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # 计算分类报告
    report = classification_report(true_label_list, predict_label_list, target_names=all_list1, digits=4)
    print("openmax Report:")
    print(report)
    TAK = CK / (TK + FU)
    UAK = TU / (TU + FK)
    KP = CK / (TK + FK)
    UP = TU / (TU + FU)
    print('TAK:', TAK)
    print('UAK:', UAK)
    print('KP:', KP)
    print('UP:', UP)
