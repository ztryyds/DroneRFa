import glob
import os
import pickle
import h5py
import joblib
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as spd


files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']
files2_list = ['T0000', 'T0110', 'T1000', 'T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010', 'T10100', 'T10101', 'T10110', 'T10111', 'T11000']
openmax6_list = ['T0100']
openmax2_list = ['T1111', 'T1110', 'T10011', 'T10001']


openmax_conf = np.zeros((21, 21))
NCHANNELS = 1
NCLASSES = 20
threshold = 0.6  ##增加阈值参数
alpharank = 3
encoder = joblib.load('../model/label_encoder.joblib')
label_allow_list = encoder.classes_
all_list = np.append(label_allow_list, 'unknowns')
all_list1 = np.append(label_allow_list, 'unknowns')
# eucos euclidean cosine
distance_type = 'euclidean'


def query_weibull(category_name, weibull_model):
    category_weibull = []
    category_weibull += [weibull_model[category_name]['mean_vec'][category_name]]
    category_weibull += [weibull_model[category_name]['distances_%s' % distance_type]]
    category_weibull += [weibull_model[category_name]['weibull_model']]
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


def recalibrate_scores(weibull_model, label_list, softmax, score, av):
    """
    使用OpenMax重新校准分数
    """
    ranked_list = tf.argsort(softmax, direction='DESCENDING')
    ranked_list = tf.reshape(ranked_list, [-1])
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    ranked_alpha = np.zeros(20)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    openmax_fc8, openmax_score_u = [], []
    for channel in range(NCHANNELS):
        channel_scores = score[channel, :]
        channel_av = av[channel, :]
        openmax_fc8_channel = []
        openmax_fc8_unknown = []
        for categoryid in range(NCLASSES):
            category_weibull = query_weibull(label_list[categoryid], weibull_model)

            if distance_type == 'euclidean':
                channel_distance = spd.euclidean(channel_av, category_weibull[0])
            elif distance_type == 'cosine':
                channel_distance = spd.cosine(channel_scores, category_weibull[0])
            elif distance_type == 'eucos':
                channel_distance = spd.euclidean(channel_scores, category_weibull[0])/200 \
                                   + spd.cosine(channel_scores, category_weibull[0])
            else:
                print('distance_type error', distance_type)
                return
            wscore = category_weibull[2][channel].w_score(channel_distance)
            modified_fc8_score = channel_scores[categoryid] * (1 - wscore * ranked_alpha[categoryid])
            openmax_fc8_channel += [modified_fc8_score]
            openmax_fc8_unknown += [channel_scores[categoryid] - modified_fc8_score]
        openmax_fc8 += [openmax_fc8_channel]
        openmax_score_u += [openmax_fc8_unknown]
    openmax_fc8 = np.array(openmax_fc8)
    openmax_score_u = np.array(openmax_score_u)

    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u)
    return np.array(openmax_probab)


def apply_openmax_to_sample(softmax, score, av):
    if distance_type == 'euclidean':
        with open('../model/euclidean_model.pkl', 'rb') as f:
            weibull_model = pickle.load(f)
    elif distance_type == 'cosine':
        with open('../model/cosine_model.pkl', 'rb') as f:
            weibull_model = pickle.load(f)
    elif distance_type == 'eucos':
        with open('../model/eucos_model.pkl', 'rb') as f:
            weibull_model = pickle.load(f)
    else:
        print('distance_type error', distance_type)
        return
    openmax = recalibrate_scores(weibull_model, label_allow_list, softmax, score, av)
    return openmax


if __name__ == "__main__":
    predict_label_list = []
    true_label_list = []
    for label in all_list1:
        # if label != 'T0101':
        #     continue
        file_path = f'../softmax/{label}_softmax.mat'
        if label not in label_allow_list:
            label = 'unknowns'
        softmax_list = loadmat(file_path)['softmax']
        logits_list = loadmat(file_path)['logits']
        av_list = loadmat(file_path)['av']
        for i, softmax in enumerate(softmax_list):
            score = tf.abs(logits_list[i])
            av = av_list[i]
            openmax = apply_openmax_to_sample(softmax, score, av)
            predict_label = tf.argmax(openmax)

            # if (predict_label == 20) or (openmax.max()<threshold) :
            if predict_label == 20:
                predict_label = 'unknowns'
            else:
                predict_label = encoder.inverse_transform([predict_label])[0]
            true_label = label
            predict_label_list.append(predict_label)
            true_label_list.append(true_label)
            openmax_conf += confusion_matrix([true_label], [predict_label], labels=all_list)

    # openmax处理
    conf_normalized = openmax_conf / openmax_conf.sum(axis=1)[:, np.newaxis]
    save_folder = "../Results"
    os.makedirs(save_folder, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_normalized, annot=True, cmap='Blues', fmt='.2f',
               xticklabels=all_list, yticklabels=all_list)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Normalized Confusion Matrix')
    filename = os.path.join(save_folder, 'openmax.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # 计算分类报告
    report = classification_report(true_label_list, predict_label_list, target_names=all_list)
    print("openmax Report:")
    print(report)
