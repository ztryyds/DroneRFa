import glob
import os
import pickle
import h5py
import joblib
import numpy as np
import tensorflow as tf
from SupConResNet_model import LinearClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as spd

# files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
# files5_list = ['T0011']
# files2_list = ['T0000', 'T0110', 'T1000', 'T1010', 'T1011', 'T1100', 'T1101',
#                'T10000', 'T10010', 'T10100', 'T10101', 'T10110', 'T10111', 'T11000']
#
# openmax6_list = ['T0100']
# openmax2_list = ['T1111', 'T1110', 'T10011', 'T10001']
files6_list = ['T0001', 'T0010', 'T0100', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']

files2_list = ['T0000', 'T0110', 'T1000','T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010','T1111', 'T1110', 'T10011', 'T10001']

openmax2_list = ['T10100','T10101','T10110','T10111','T11000']
openmax_conf = np.zeros((21, 21))
NCHANNELS = 1
NCLASSES = 20
threshold = 0.6           ##增加阈值参数

encoder = joblib.load('../model/label_encoder1.joblib')
label_allow_list = encoder.classes_
all_list = np.append(label_allow_list, 'unknowns')
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


def recalibrate_scores(weibull_model, label_list, score, feature, alpharank=8):
    """
    使用OpenMax重新校准分数
    """
    ranked_list = tf.argsort(score, direction='DESCENDING')
    ranked_list = tf.reshape(ranked_list, [-1])
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    ranked_alpha = np.zeros(20)
    for i in range(len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    openmax_fc8, openmax_score_u = [], []
    for channel in range(NCHANNELS):
        channel_scores = feature[channel, :]
        openmax_fc8_channel = []
        openmax_fc8_unknown = []
        for categoryid in range(NCLASSES):
            category_weibull = query_weibull(label_list[categoryid], weibull_model)

            if distance_type == 'euclidean':
                channel_distance = spd.euclidean(channel_scores, category_weibull[0])
            elif distance_type == 'cosine':
                channel_distance = spd.cosine(channel_scores, category_weibull[0])
            elif distance_type == 'eucos':
                channel_distance = spd.euclidean(channel_scores, category_weibull[0])\
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
    openmax_fc8 = tf.abs(openmax_fc8)
    openmax_score_u = tf.abs(openmax_score_u)
    openmax_fc8 = np.array(openmax_fc8)
    openmax_score_u = np.array(openmax_score_u)

    openmax_probab = computeOpenMaxProbability(openmax_fc8, openmax_score_u)
    softmax_probab = tf.reshape(score, [-1])
    return np.array(openmax_probab), np.array(softmax_probab)


def apply_openmax_to_sample(test_path, model):

    if distance_type == 'euclidean':
        with open('../model/euclidean_model1.pkl', 'rb') as f:
            weibull_model = pickle.load(f)
    elif distance_type == 'cosine':
        with open('../model/cosine_model1.pkl', 'rb') as f:
            weibull_model = pickle.load(f)
    elif distance_type == 'eucos':
        with open('../model/eucos_model1.pkl', 'rb') as f:
            weibull_model = pickle.load(f)
    else:
        print('distance_type error',distance_type)
        return
    with h5py.File(test_path, 'r') as file:
        input = file['STFT Magnitude'][:].view(np.complex64)
        input = tf.expand_dims(input, axis=0)
        outputs = model(input)
        feature = outputs[0]
        score = outputs[1]
    openmax, softmax = recalibrate_scores(weibull_model, label_allow_list, score, feature)
    return openmax, softmax


def set_test_files(path_to_test_files):
    test_file_paths = []
    test_labels = []
    for parent_path in glob.glob(path_to_test_files):
        parent_dir_name = os.path.basename(parent_path)
        label = parent_dir_name.split('_')[0]
        if label in files6_list:
            file_nums = 150
        elif label in files5_list:
            file_nums = 150
        elif label in files2_list:
            file_nums = 300
        elif label in openmax2_list:
            label = 'unknowns'
            file_nums = 100
        # elif label in openmax6_list:
        #     label = 'unknowns'
        #     file_nums = 50
        else:
            print('label error', parent_path)
            continue
        h5_file_path = os.path.join(parent_path, 'stft/*.h5')
        for file_path in glob.glob(h5_file_path):
            file_basename = os.path.splitext(os.path.basename(file_path))[0]
            file_id = file_basename.split('_')[1]  # Extract label from filename
            if int(file_id) >= file_nums:
                continue
            test_file_paths.append(file_path)
            test_labels.append(label)
    return test_file_paths, test_labels


if __name__ == "__main__":
    model = LinearClassifier(input_shape=(256, 236, 2), num_classes=20)
    model.load_weights('../model/best_Classifier_model1.h5')  # 加载训练好的模型权重
    new_model = tf.keras.Model(inputs=model.input,
                               outputs=[model.get_layer('logits').output, model.output])
    new_model.summary()

    path_to_test_files = '../test/*'
    test_file_paths, test_labels = set_test_files(path_to_test_files)
    predict_label_list = []
    true_label_list = []
    for i, test_path in enumerate(test_file_paths):
        openmax, softmax = apply_openmax_to_sample(test_path, new_model)
        predict_label = tf.argmax(openmax)

        #if (predict_label == 20) or (openmax.max()<threshold) :
        if predict_label == 20:
            predict_label = 'unknowns'
        else:
            predict_label = encoder.inverse_transform([predict_label])[0]
        true_label = test_labels[i]
        predict_label_list.append(predict_label)
        true_label_list.append(true_label)
        openmax_conf += confusion_matrix([true_label], [predict_label], labels=all_list)

    # openmax处理
    #conf_normalized = openmax_conf / openmax_conf.sum(axis=1)[:, np.newaxis]
    #save_folder = "../Results"
    #os.makedirs(save_folder, exist_ok=True)
    #plt.figure(figsize=(10, 8))
    #sns.heatmap(conf_normalized, annot=True, cmap='Blues', fmt='.2f',
    #            xticklabels=all_list, yticklabels=all_list)
    #plt.xlabel('Predicted labels')
    #plt.ylabel('True labels')
    #plt.title('Normalized Confusion Matrix')
    #filename = os.path.join(save_folder, 'openmax_cos.png')
    #plt.savefig(filename, dpi=300, bbox_inches='tight')

    # 计算分类报告
    report = classification_report(true_label_list, predict_label_list, target_names=all_list)
    print("openmax Report:")
    print(report)
