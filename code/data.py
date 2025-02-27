import os
import torch
import torchaudio
import numpy as np
import librosa
import torch.nn.functional as F

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def load_label(label_path):
    with open(label_path, 'r', encoding='utf-8') as file:
        label = file.read().strip()
    return label

def augment_audio(waveform, sample_rate, augment_noise=False, augment_pitch=False, augment_speed=False):
    augmented_audios = []
    
    # 时间移位
    shift = np.random.randint(sample_rate)
    y_shifted = torch.roll(waveform, shifts=shift, dims=1)
    augmented_audios.append(y_shifted)

    # 加噪声
    if augment_noise:
        noise = torch.randn_like(waveform)
        y_noisy = waveform + 0.005 * noise
        augmented_audios.append(y_noisy)

    # 改变音调：确保 new_sample_rate 是整数
    if augment_pitch:
        n_steps = np.random.randint(-5, 5)
        new_sample_rate = int(sample_rate * (2.0 ** (n_steps / 12.0)))  # 强制转换为整数
        y_pitch = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)(waveform)
        augmented_audios.append(y_pitch)

    # 改变速度：确保 new_sample_rate 是整数
    if augment_speed:
        speed_factor = np.random.uniform(0.7, 1.3)
        new_sample_rate = int(sample_rate * speed_factor)  # 强制转换为整数
        y_speed = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)(waveform)
        augmented_audios.append(y_speed)
    
    return augmented_audios







def preprocess_audio(waveform, sample_rate, max_len=300):
    # 归一化音频
    waveform = waveform / waveform.abs().max()

    # 提取梅尔频谱图（Mel Spectrogram）
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128)(waveform).squeeze(dim=0)
    print(f'Mel Spectrogram Shape: {mel_spectrogram.shape}')  # 输出梅尔频谱图的形状

    # 生成 MFCC 特征
    mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=13)(waveform).squeeze(dim=0)
    print(f'MFCC Shape: {mfcc.shape}')  # 输出MFCC的形状

    # 提取 Chroma 特征使用 librosa
    waveform_np = waveform.numpy()[0]  # 将 tensor 转为 numpy 数组
    chroma = librosa.feature.chroma_stft(y=waveform_np, sr=sample_rate)
    chroma = torch.tensor(chroma)  # 转回 torch tensor
    print(f'chroma Shape: {chroma.shape}')  # 输出Chroma的形状

    # 计算零交叉率（ZCR）
    zcr = compute_zero_crossing_rate(waveform).unsqueeze(dim=0)  # 增加维度使其成为 (1, 1, n_frames)
    print(f'ZCR Shape: {zcr.shape}')  # 输出 ZCR 的形状

    # 确保所有特征的时间步数不超过 max_len
    def adjust_length(feature, max_len):
        # 如果特征的时间步数大于 max_len，进行裁剪
        if feature.size(1) > max_len:
            feature = feature[:, :max_len]
        # 如果特征的时间步数小于 max_len，进行零填充
        elif feature.size(1) < max_len:
            padding = max_len - feature.size(1)
            feature = F.pad(feature, (0, padding))  # 在时间维度上填充
        return feature

    # 确保所有特征的时间步数一致，使用最大时间步数
    max_time_steps = max(mel_spectrogram.size(1), mfcc.size(1), chroma.size(1), zcr.size(1))

    mel_spectrogram = adjust_length(mel_spectrogram, max_time_steps)
    mfcc = adjust_length(mfcc, max_time_steps)
    chroma = adjust_length(chroma, max_time_steps)
    zcr = adjust_length(zcr, max_time_steps)

    # 合并所有特征
    features = torch.cat([mel_spectrogram, mfcc, chroma, zcr], dim=0)
    print(f'Final features shape: {features.shape}')  # 输出合并后的特征的形状

    return features

# 计算零交叉率的示例实现（您可以根据需求修改此函数）
def compute_zero_crossing_rate(waveform):
    zcr = torch.abs(torch.sign(waveform[:, 1:]) - torch.sign(waveform[:, :-1]))
    return zcr.sum(dim=-1)





def process_file(audio_file, label_path, max_len, device):
    # 载入音频文件
    waveform, sample_rate = load_audio(audio_file)
    
    # 处理音频数据并将其转移到指定的设备
    features = preprocess_audio(waveform, sample_rate, max_len).to(device)

    # 直接加载对应的标签文件，不再需要通过文件名解析ID
    label = load_label(label_path)

    # 数据增强处理音频，生成增强后的音频数据
    augmented_audios = augment_audio(waveform, sample_rate, augment_noise=True, augment_pitch=True, augment_speed=True)
    
    # 处理增强后的音频，并将每个增强后的特征转移到指定设备
    augmented_features = [preprocess_audio(aug_y, sample_rate, max_len).to(device) for aug_y in augmented_audios]

    # 返回原始特征和增强特征
    return [(features, label)] + [(aug_features, label) for aug_features in augmented_features]



import os
import torch
import torch.nn.functional as F
import numpy as np

def load_dataset(audio_train_dir, audio_test_dir, train_label_dir, test_label_dir, max_len=300, device=None, limit_files=False):
    # 获取所有训练和测试集音频文件列表
    train_files = [f for f in os.listdir(audio_train_dir) if f.endswith('.wav')]
    test_files = [f for f in os.listdir(audio_test_dir) if f.endswith('.wav')]

    # 获取训练和测试标签文件列表
    train_labels = [f for f in os.listdir(train_label_dir) if f.endswith('.txt')]  # 假设标签是txt文件
    test_labels = [f for f in os.listdir(test_label_dir) if f.endswith('.txt')]  # 假设标签是txt文件

    # 如果只处理前10个文件
    if limit_files:
        train_files = train_files[:10]  # 只取前10个训练文件
        test_files = test_files[:10]  # 只取前10个测试文件
        train_labels = train_labels[:10]  # 只取前10个训练标签文件
        test_labels = test_labels[:10]  # 只取前10个测试标签文件

    traindata = []
    testdata = []

    # 处理训练集
    for audio_file, label_file in zip(train_files, train_labels):
        traindata.extend(process_file(os.path.join(audio_train_dir, audio_file), os.path.join(train_label_dir, label_file), max_len, device))

    # 处理测试集
    for audio_file, label_file in zip(test_files, test_labels):
        testdata.extend(process_file(os.path.join(audio_test_dir, audio_file), os.path.join(test_label_dir, label_file), max_len, device))

    # 获取所有唯一标签
    all_labels = sorted(set(label for _, label in traindata).union(set(label for _, label in testdata)))
    
    # 创建标签映射
    label_map = {label: idx for idx, label in enumerate(all_labels)}

    # 将字符串标签转换为整数标签
    traindata = [(features, label_map[label]) for features, label in traindata]
    testdata = [(features, label_map[label]) for features, label in testdata]

    # 提取特征和标签
    train_features, train_labels = zip(*traindata)
    test_features, test_labels = zip(*testdata)

    # 确保所有特征的时间步数一致
    def adjust_length(feature, max_len):
        if feature.size(1) > max_len:
            feature = feature[:, :max_len]
        elif feature.size(1) < max_len:
            padding = max_len - feature.size(1)
            feature = F.pad(feature, (0, padding))
        return feature

    # 对每个特征进行填充或裁剪，使它们的时间步数一致
    train_features = [adjust_length(f, max_len) for f in train_features]
    test_features = [adjust_length(f, max_len) for f in test_features]

    # 转换为 NumPy 数组
    train_features = np.array([f.cpu().numpy() for f in train_features])  # 转为 NumPy 数组
    test_features = np.array([f.cpu().numpy() for f in test_features])  # 转为 NumPy 数组
    train_labels = np.array(train_labels)  # 确保 labels 是整数类型
    test_labels = np.array(test_labels)  # 确保 labels 是整数类型

    # 返回数据：训练集和测试集的特征与标签，及标签映射
    return (train_features, train_labels), (test_features, test_labels), label_map




