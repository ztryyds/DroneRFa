import joblib
import pickle
from scipy.io import loadmat
import libmr

NCHANNELS = 1
# 加载已保存的MAV和类别编码器
mean_vector_path = '../openmax_data/openmax_gan_SupTransResNet/logits/MAV'
distance_save_path = '../openmax_data/openmax_gan_SupTransResNet/logits/distances'
distance_type = 'euclidean'
tail_percent_dict = {
    'T0000': 0.01,
    'T0010': 0.01,
    'T0011': 0.01,
    'T0100': 0.01,
    'T0101': 0.01,
    'T0110': 0.01,
    'T0111': 0.01,
    'T1000': 0.01,
    'T10000': 0.01,
    'T1001': 0.01,
    'T1010': 0.01,
    'T10100': 0.01,
    'T1011': 0.01,
    'T1100': 0.01,
    'T1101': 0.01,
    'T1111': 0.01,

    'T10010': 0.03,
    'T10101': 0.15,
    'T10111': 0.11,
    'T1110': 0.16,
}


def weibull_tailfitting(mean_vector_path, distance_save_path, label_allow_list, distance_type):
    weibull_model = {}
    # for each category, read meanfile, distance file, and perform weibull fitting
    for label_name in label_allow_list:
        if label_name == 'unknowns':
            continue
        print(f"Processing label: {label_name}")
        weibull_model[label_name] = {}
        distance_scores = loadmat(f'{distance_save_path}/{label_name}_distances.mat')[distance_type]
        meantrain_vec = loadmat(f'{mean_vector_path}/{label_name}_mav.mat')['logits']

        weibull_model[label_name]['distances'] = distance_scores
        weibull_model[label_name]['mean_vec'] = meantrain_vec
        weibull_model[label_name]['weibull_model'] = []
        for channel in range(NCHANNELS):
            print(label_name, 'total size', len(distance_scores[channel, :]))
            mr = libmr.MR()
            tail_percent = tail_percent_dict[label_name]
            tailsize = int(len(distance_scores[channel, :]) * tail_percent)
            print(label_name, 'tailsize', tailsize)
            tailtofit = sorted(distance_scores[channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[label_name]['weibull_model'] += [mr]
    return weibull_model


def main():
    encoder = joblib.load('../model/label_encoder_close.joblib')
    label_allow_list = encoder.classes_
    # 进行Weibull拟合
    weibull_model = weibull_tailfitting(mean_vector_path, distance_save_path, label_allow_list, distance_type)

    with open('../model/weibull_model_gan.pkl', 'wb') as f:
        pickle.dump(weibull_model, f)
        print('weibull_model has saved')


if __name__ == "__main__":
    main()
