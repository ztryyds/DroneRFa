from sklearn.preprocessing import LabelEncoder
import joblib
# 定义需要识别的无人机类别列表
files6_list = ['T0001', 'T0010', 'T0101', 'T0111', 'T1001']
files5_list = ['T0011']
files2_list = ['T0000', 'T0110', 'T1000','T1010', 'T1011', 'T1100', 'T1101',
               'T10000', 'T10010','T10100','T10101','T10110','T10111','T11000']

openmax6_list = ['T0100']
openmax2_list = ['T1111','T1110','T10011','T10001']

# # 标签编码
encoder = LabelEncoder()
label_list = files2_list+files5_list+files6_list

print(len(label_list))
label_train_encoded = encoder.fit_transform(label_list)
# 保存LabelEncoder到文件
joblib.dump(encoder, '../model/label_encoder.joblib')
encoder_new = joblib.load('../model/label_encoder.joblib')
# print(encoder.inverse_transform([13]))
print(encoder_new.classes_)
