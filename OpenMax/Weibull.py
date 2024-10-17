import joblib
import pickle
from scipy.io import loadmat
import libmr

NCHANNELS = 1
# 加载已保存的MAV和类别编码器
mean_vector_path = '../data/MAV'
distance_save_path = '../data/distances'
tail_percent = 0.013

def weibull_tailfitting(meanfiles_path, distancefiles_path, labellist, distance_type):
    weibull_model = {}
    # for each category, read meanfile, distance file, and perform weibull fitting
    for category in labellist:
        print(f"Processing label: {category}")
        weibull_model[category] = {}
        distance_scores = loadmat('%s/%s_distances.mat' % (distancefiles_path, category))[distance_type]
        meantrain_vec = loadmat('%s/%s_mav.mat' % (meanfiles_path, category))

        weibull_model[category]['distances_%s' % distance_type] = distance_scores
        weibull_model[category]['mean_vec'] = meantrain_vec
        weibull_model[category]['weibull_model'] = []
        for channel in range(NCHANNELS):
            mr = libmr.MR()
            tailsize = int(len(distance_scores[channel, :]) * tail_percent)
            print(category, 'tailsize', tailsize)
            tailtofit = sorted(distance_scores[channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category]['weibull_model'] += [mr]
    return weibull_model


def main():
    encoder = joblib.load('../model/label_encoder.joblib')
    label_allow_list = encoder.classes_
    # 进行Weibull拟合
    euclidean_model = weibull_tailfitting(mean_vector_path, distance_save_path, label_allow_list,
                                          distance_type='euclidean')
    cosine_model = weibull_tailfitting(mean_vector_path, distance_save_path, label_allow_list, distance_type='cosine')
    eucos_model = weibull_tailfitting(mean_vector_path, distance_save_path, label_allow_list, distance_type='eucos')
    with open('../model/euclidean_model.pkl', 'wb') as f:
        pickle.dump(euclidean_model, f)
        print('euclidean_model has saved')

    with open('../model/cosine_model.pkl', 'wb') as f:
        pickle.dump(cosine_model, f)
        print('cosine_model has saved')

    with open('../model/eucos_model.pkl', 'wb') as f:
        pickle.dump(eucos_model, f)
        print('eucos_model has saved')


if __name__ == "__main__":
    main()
