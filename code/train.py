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


def train_model(audio_dir, label_dir, num_epochs=20, batch_size=4, learning_rate=0.0001, weight_decay=1e-4, model_path='crnn_model.pth'):
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    (X_train, y_train), (X_test, y_test), label_map = load_dataset(audio_dir, label_dir)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    # 打印输入特征的形状
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 定义模型
    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = len(label_map)
    model = CRNN(input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"加载保存的模型权重: {model_path}")

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='呼吸音分类任务')
    parser.add_argument('--audio_dir', type=str, required=True, help='音频文件目录')
    parser.add_argument('--label_dir', type=str, required=True, help='标签文件目录')
    parser.add_argument('--num_epochs', type=int, default=20, help='训练的轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 正则化权重衰减')
    parser.add_argument('--model_path', type=str, default='crnn_model.pth', help='保存模型的路径')
    args = parser.parse_args()

    train_model(args.audio_dir, args.label_dir, args.num_epochs, args.batch_size, args.learning_rate, args.weight_decay, args.model_path)