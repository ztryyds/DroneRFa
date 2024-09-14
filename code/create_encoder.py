from sklearn.preprocessing import LabelEncoder
import joblib
# 定义需要识别的无人机类别列表
# files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
# files5_list = ['T0011']
# files2_list = ['T0000', 'T0110', 'T1000','T1010', 'T1011', 'T1100', 'T1101',
#                'T10000', 'T10010','T10100','T10101','T10110','T10111','T11000']
#
# openmax6_list = ['T0100']
# openmax2_list = ['T1111','T1110','T10011','T10001']
files6_list = ['T0001', 'T0010', 'T0100', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']

files2_list = ['T0000', 'T0110', 'T1000','T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010','T1111', 'T1110', 'T10011', 'T10001']

openmax2_list = ['T10100','T10101','T10110','T10111','T11000']
# 标签编码
encoder = LabelEncoder()
label_new = files2_list+files5_list+files6_list

# for label in label_allow_list:
#     if label in files6_list:
#         continue
#     if label in files5_list:
#         continue
#     label_new.append(label)
print(len(label_new))
label_train_encoded = encoder.fit_transform(label_new)
# 保存LabelEncoder到文件
joblib.dump(encoder, '../model/label_encoder1.joblib')
# encoder_new = joblib.load('../model/label_encoder.joblib')
# print(encoder_new.classes_)
