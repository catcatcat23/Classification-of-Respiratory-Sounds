import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import CRNN
from data import load_dataset  
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import shap
def save_data(features, labels, features_file='features.npy', labels_file='labels.npy'):
    # 使用 numpy 保存数据
    np.save(features_file, features)  # 保存特征
    np.save(labels_file, labels)  # 保存标签

    # 或者使用 torch 保存数据
    # torch.save(features, features_file)  # 使用 torch 保存特征
    # torch.save(labels, labels_file)  # 使用 torch 保存标签



def load_or_process_data(audio_train_dir, audio_test_dir, train_label_dir, test_label_dir, max_len=300, device=None, limit_files=False):
    # 检查是否已保存预处理数据
    if os.path.exists('train_features.npy') and os.path.exists('train_labels.npy') and os.path.exists('test_features.npy') and os.path.exists('test_labels.npy'):
        print("加载预处理好的数据...")
        X_train = np.load('train_features.npy')
        y_train = np.load('train_labels.npy')
        X_test = np.load('test_features.npy')
        y_test = np.load('test_labels.npy')
    else:
        print("重新处理音频数据...")
        # 如果没有找到预处理数据，重新处理音频
        (X_train, y_train), (X_test, y_test), label_map = load_dataset(audio_train_dir, audio_test_dir, train_label_dir, test_label_dir, max_len, device, limit_files)

        # 保存数据以便以后使用
        save_data(X_train, y_train, 'train_features.npy', 'train_labels.npy')
        save_data(X_test, y_test, 'test_features.npy', 'test_labels.npy')

    return X_train, y_train, X_test, y_test, label_map


def train_model(audio_train_dir, audio_test_dir, train_label_dir, test_label_dir, num_epochs=40, batch_size=4, learning_rate=0.0001, weight_decay=1e-4, model_path='crnn_model.pth'):
    # 训练模型的逻辑
    print(f"Using {audio_train_dir} for training and {audio_test_dir} for testing")
    print(f"Training labels from {train_label_dir} and testing labels from {test_label_dir}")
    print(f"Training for {num_epochs} epochs, with batch size {batch_size}, learning rate {learning_rate}, weight decay {weight_decay}")

    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    X_train, y_train, X_test, y_test, label_map = load_or_process_data(audio_train_dir, audio_test_dir, train_label_dir, test_label_dir, max_len=300, device=device, limit_files=False)


    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # 确保输入形状为 [batch_size, 1, height, width]
    X_train = X_train.unsqueeze(1)  # 增加通道维度
    X_test = X_test.unsqueeze(1)    # 增加通道维度

    # 打印输入特征的形状
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    # 打印训练集和测试集的大小
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 定义模型
    input_dim = X_train.shape[1]  # Channels
    hidden_dim = 128
    output_dim = len(label_map)
    model = CRNN(input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)  # 移动到指定设备
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()
                pbar.set_postfix(loss=running_loss/(i+1))
                pbar.update(1)
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}%")

        
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # 移动到指定设备
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        val_loss /= len(test_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"测试集准确率: {val_accuracy}%, 验证集损失: {val_loss}")

        scheduler.step(val_loss)

        if epoch > 10 and val_losses[-1] > min(val_losses[:-1]):
            print("验证损失不再改善，提前停止训练")
            break

    torch.save(model.state_dict(), model_path)

    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='训练损失')
    plt.plot(epochs, val_losses, label='验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='训练准确率')
    plt.plot(epochs, val_accuracies, label='验证准确率')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    

    plot_shap_values(model, X_test)



# 强制将每个子模块都设置为训练模式
def ensure_train_mode(model):
    for layer in model.children():
        if isinstance(layer, torch.nn.Module):
            layer.train()  # 确保每一层都处于训练模式

def calculate_shap_values(model, X_test):
    # 确保模型参数启用了梯度计算
    for param in model.parameters():
        param.requires_grad = True

    # 确保输入数据启用了梯度计算
    X_test.requires_grad_()  # 启用梯度计算

    # 强制设置每一层为训练模式
    ensure_train_mode(model)



    # 使用 DeepExplainer 来计算 SHAP 值
    explainer = shap.DeepExplainer(model, X_test[:100])  # 使用前100个样本来初始化解释器
    shap_values = explainer.shap_values(X_test)  # 计算 SHAP 值

    return shap_values



def plot_shap_values(model, X_test):
    # 计算 SHAP 值
    shap_values = calculate_shap_values(model, X_test)

    # 特征名称列表
    feature_names = (
        [f"Mel Spectrogram {i+1}" for i in range(128)] +  # Mel Spectrogram
        [f"MFCC {i+1}" for i in range(13)] +             # MFCC
        [f"Chroma {i+1}" for i in range(12)] +           # Chroma
        ["ZCR"]                                          # Zero Crossing Rate
    )
    
    # 特征范围 (每个特征对应 300 个时间步)
    feature_ranges = (
        [(i, i) for i in range(128)] +  # Mel Spectrogram
        [(i + 128, i + 128) for i in range(13)] +  # MFCC
        [(i + 141, i + 141) for i in range(12)] +  # Chroma
        [(153, 153)]  # ZCR
    )

    # 展示 SHAP 特征重要性
    shap.summary_plot(shap_values[0], X_test.reshape(X_test.shape[0], -1), feature_names=feature_names, feature_ranges=feature_ranges)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='呼吸音分类任务')

    # 训练集和测试集音频文件目录
    parser.add_argument('--audio_train_dir', type=str, required=True, help='训练集音频文件目录')
    parser.add_argument('--audio_test_dir', type=str, required=True, help='测试集音频文件目录')

    # 训练集和测试集标签文件目录
    parser.add_argument('--train_label_dir', type=str, required=True, help='训练集标签文件目录')
    parser.add_argument('--test_label_dir', type=str, required=True, help='测试集标签文件目录')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=20, help='训练的轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 正则化权重衰减')

    # 模型保存路径
    parser.add_argument('--model_path', type=str, default='crnn_model.pth', help='保存模型的路径')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用训练模型函数
    train_model(
        args.audio_train_dir, 
        args.audio_test_dir, 
        args.train_label_dir, 
        args.test_label_dir, 
        args.num_epochs, 
        args.batch_size, 
        args.learning_rate, 
        args.weight_decay, 
        args.model_path
    )
