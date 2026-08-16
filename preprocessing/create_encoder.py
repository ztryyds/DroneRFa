from sklearn.preprocessing import LabelEncoder
import joblib

# 定义需要识别的无人机类别
close_list = ['T0010', 'T0101', 'T0111', 'T0011', 'T0000', 'T0110', 'T1010', 'T1011', 'T1100', 'T1101',
              'T10000', 'T10010', 'T10100', 'T10101', 'T10111', 'T1111', 'T0100', 'T1000', 'T1001', 'T1110']
open_list = ['T0001', 'T10001', 'T10011', 'T11000', 'T10110']

all_list = close_list + open_list
unknown_list = close_list + ['unknowns']


def fit_and_print(name, encoder_obj, data_list, save_path):
    """
    封装的函数：执行拟合、打印类别映射关系、并保存模型
    """
    print(f"=== {name} (Length: {len(data_list)}) ===")
    encoded_labels = encoder_obj.fit_transform(data_list)

    # 打印每个数字编码对应的原始类别
    print(encoder_obj.classes_)
    # 保存模型
    joblib.dump(encoder_obj, save_path)


# 1. label_encoder_close
encoder = LabelEncoder()
fit_and_print('label_encoder_close', encoder, close_list, '../model/label_encoder_close.joblib')

# 2. label_encoder_open
encoder1 = LabelEncoder()
fit_and_print('label_encoder_open', encoder1, open_list, '../model/label_encoder_open.joblib')

# 3. label_encoder_all
encoder2 = LabelEncoder()
fit_and_print('label_encoder_all', encoder2, all_list, '../model/label_encoder_all.joblib')

# 4. label_encoder_unknown
encoder3 = LabelEncoder()
fit_and_print('label_encoder_unknown', encoder3, unknown_list, '../model/label_encoder_unknown.joblib')

# 5. label_encoder_1 (close_list 前10个)
encoder4 = LabelEncoder()
fit_and_print('label_encoder_1', encoder4, close_list[:10], '../model/label_encoder_1.joblib')

# 6. label_encoder_2 (close_list 后10个)
encoder5 = LabelEncoder()
fit_and_print('label_encoder_2', encoder5, close_list[10:], '../model/label_encoder_2.joblib')