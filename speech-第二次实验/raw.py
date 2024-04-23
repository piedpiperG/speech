## 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
import soundfile as sf
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

# 可以根据需要导入其他库，比如librosa用于音频处理

TrainDir = "Dataset/TRAIN"
TestDir = "Dataset/TEST"


## 请在这里写代码加载我们划分好的TIMIT训练集和测试集

def load_timit_dataset(base_path):
    speakers_data = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".wav"):
                speaker_id = root.split(os.sep)[-1]
                file_path = os.path.join(root, file)
                data, samplerate = sf.read(file_path)
                if speaker_id not in speakers_data:
                    speakers_data[speaker_id] = []
                speakers_data[speaker_id].append((data, samplerate))
    return speakers_data


train_data = load_timit_dataset(TrainDir)
test_data = load_timit_dataset(TestDir)


# 请编写或使用库函数提取MFCC等音频特征
def extract_features(audio_data, samplerate):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=samplerate, n_mfcc=13)
    return mfccs


def prepare_dataset(speakers_data):
    features = []
    labels = []
    for speaker_id, audios in speakers_data.items():
        for audio, rate in audios:
            mfcc = extract_features(audio, rate)
            features.append(mfcc)
            labels.append(speaker_id)
    return features, np.array(labels)  # 这里将 labels 转换为 numpy 数组


train_data_features, train_data_labels = prepare_dataset(train_data)
test_data_features, test_data_labels = prepare_dataset(test_data)


# 在这部分，你可以选择不同的分类器和模型如GMM模型来进行实验
def flatten_features(features):
    # 将特征从三维数组转换为二维数组，以便可以用于GMM
    flattened_features = []
    for feature in features:
        # 使用均值跨时间序列来简化数据，可以尝试其他方法如最大值、中位数等
        flattened_features.append(np.mean(feature, axis=1))
    return np.vstack(flattened_features)


# 数据扁平化
X_train = flatten_features(train_data_features)
X_test = flatten_features(test_data_features)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 确定GMM组件的数量
n_components = len(np.unique(train_data_labels))

# 训练GMM
gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=200, random_state=0)
gmm.fit(X_train)

## 请编写代码或使用库函数accuracy_score计算测试集上的准确率Accuracy
# 对训练数据进行聚类
train_labels_gmm = gmm.predict(X_train)

# 创建从GMM组件到实际标签的映射
from scipy.stats import mode

labels_mapping = {}
for i in range(n_components):
    mask = (train_labels_gmm == i)
    # 为每个组件找出最常见的标签
    if np.any(mask):
        labels_mapping[i] = mode(train_data_labels[mask])[0][0]

# 使用映射进行测试数据的预测
test_labels_gmm = gmm.predict(X_test)
predicted_labels = [labels_mapping[label] for label in test_labels_gmm]

# 计算准确率
accuracy = accuracy_score(test_data_labels, predicted_labels)
print(f"Test Accuracy: {accuracy:.2f}")
