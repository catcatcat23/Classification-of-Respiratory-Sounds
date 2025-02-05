import os
import torch
import torchaudio
import numpy as np
from sklearn.model_selection import train_test_split

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

    # 提取梅尔频谱图特征
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128)(waveform)
    mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

    # 填充或截断特征，使其具有相同的时间步数
    if mel_spectrogram_db.shape[2] < max_len:
        pad_width = max_len - mel_spectrogram_db.shape[2]
        mel_spectrogram_db = torch.nn.functional.pad(mel_spectrogram_db, (0, pad_width))
    else:
        mel_spectrogram_db = mel_spectrogram_db[:, :, :max_len]

    return mel_spectrogram_db

def process_file(audio_file, label_dir, max_len, device):
    waveform, sample_rate = load_audio(audio_file)
    features = preprocess_audio(waveform, sample_rate, max_len).to(device)

    file_id = os.path.basename(audio_file).split('=')[1].split('.')[0]
    label_file = f"ID={file_id}.txt"
    label_path = os.path.join(label_dir, label_file)
    label = load_label(label_path)

    augmented_audios = augment_audio(waveform, sample_rate, augment_noise=True, augment_pitch=True, augment_speed=True)
    augmented_features = [preprocess_audio(aug_y, sample_rate, max_len).to(device) for aug_y in augmented_audios]

    return [(features, label)] + [(aug_features, label) for aug_features in augmented_features]

def load_dataset(audio_dir, label_dir, test_size=0.2, max_len=300, device=None):
    audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

    data = []
    for audio_file in audio_files:
        data.extend(process_file(audio_file, label_dir, max_len, device))

    labels = sorted(set(label for _, label in data))
    label_map = {label: idx for idx, label in enumerate(labels)}

    data = [(features.cpu().numpy(), label_map[label]) for features, label in data]

    features, labels = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

    return (np.array(X_train), np.array(y_train)), (np.array(X_test), np.array(y_test)), label_map
