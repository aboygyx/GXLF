import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from models.model import FWin, FWinwithGCN, FWinwithGCNXLSTM
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import r2_score # R square
from math import sqrt

# 固定随机种子
SEED = 7  # 主要用7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, target_col=-1):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_len]
        y = self.data[index + self.seq_len:index + self.seq_len + self.pred_len, self.target_col]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def prepare_data(csv_path, seq_len=96, pred_len=24):
    df = pd.read_csv(csv_path)
    data = df.values

    train_size = 3044
    train_data_raw = data[:train_size]
    test_data_raw = data[train_size:]

    # 各自归一化
    scaler_train = MinMaxScaler()
    train_data = scaler_train.fit_transform(train_data_raw)

    scaler_test = MinMaxScaler()
    test_data = scaler_test.fit_transform(test_data_raw)

    # 各自的目标值最大最小值（最后一列）
    target_max_train = np.max(train_data_raw[:, -1])
    target_min_train = np.min(train_data_raw[:, -1])

    target_max_test = np.max(test_data_raw[:, -1])
    target_min_test = np.min(test_data_raw[:, -1])

    train_dataset = TimeSeriesDataset(train_data, seq_len, pred_len)
    test_dataset = TimeSeriesDataset(test_data, seq_len, pred_len)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, target_max_test, target_min_test



def train_model(train_loader, test_loader, input_size, output_size, seq_len, pred_len, device, target_max, target_min):
    # 初始化模型
    model = FWinwithGCNXLSTM(
        enc_in=input_size,
        dec_in=input_size,
        c_out=output_size,
        seq_len=seq_len,
        label_len=seq_len // 2,
        out_len=pred_len,
        factor=5,
        d_model=32, #openstack32
        n_heads=4,
        e_layers=1, #android 2  openstack 1
        d_layers=1,
        d_ff=32,
        dropout=0.1,
        attn='prob',
        embed='fixed',
        freq='h',
        activation='gelu',
        output_attention=False,
        distil=False,
        mix=False,
        device=device,
        window_size=4,
        num_windows=1
    ).to(device)
    # ✅ 打印模型结构
    print("=== 模型结构如下 ===")
    print(model)

    # ✅ 打印模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    epochs = 50  #openstack 50 android 50

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            dec_inp = torch.zeros_like(x[:, -pred_len:, :]).to(device)
            dec_inp = torch.cat([x[:, -pred_len:, :], dec_inp], dim=1)
            optimizer.zero_grad()
            outputs = model(x, None, dec_inp, None)
            loss = criterion(outputs.squeeze(-1), y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                dec_inp = torch.zeros_like(x[:, -pred_len:, :]).to(device)
                dec_inp = torch.cat([x[:, -pred_len:, :], dec_inp], dim=1)
                outputs = model(x, None, dec_inp, None)
                loss = criterion(outputs.squeeze(-1), y)
                test_loss += loss.item()

        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, test Loss: {test_loss:.4f}')

    # 绘制测试集结果
    model.eval()
    true_values = []
    predicted_values = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            dec_inp = torch.zeros_like(x[:, -pred_len:, :]).to(device)
            dec_inp = torch.cat([x[:, -pred_len:, :], dec_inp], dim=1)
            outputs = model(x, None, dec_inp, None)
            outputs = outputs.squeeze()
            true_values.extend(y.cpu().numpy())
            predicted_values.extend(outputs.cpu().numpy())

    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)

    true_denorm = true_values * (target_max - target_min) + target_min
    pred_denorm = predicted_values * (target_max - target_min) + target_min

    print("MAE:", mean_absolute_error(true_denorm, pred_denorm))
    print("MSE:", mean_squared_error(true_denorm, pred_denorm))
    print("RMSE:", sqrt(mean_squared_error(true_denorm, pred_denorm)))
    print("R2: ", r2_score(true_denorm, pred_denorm))


    plt.figure(figsize=(10, 6))
    plt.plot(true_denorm, label='Actual TTAF')  # 蓝色线：真实值
    plt.plot(pred_denorm, label='Predicted TTAF')  # 橙色线：预测值
    # 设置坐标轴标签和图例的字体大小
    plt.xlabel('Number of Samples', fontsize=25)
    plt.ylabel('TTAF (×30s)', fontsize=25)
    plt.legend(fontsize=15)  # 设置图例的字体大小
    # 设置坐标轴上的数字的字体大小
    plt.tick_params(axis='x', labelsize=15)  # 设置x轴刻度标签的字体大小
    plt.tick_params(axis='y', labelsize=15)  # 设置y轴刻度标签的字体大小
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备数据
    csv_path = ''
    seq_len = 10
    pred_len = 1
    train_loader, test_loader, target_max, target_min = prepare_data(csv_path, seq_len, pred_len)


    # 获取输入输出维度
    sample_x, sample_y = next(iter(train_loader))
    input_size = sample_x.shape[-1]
    output_size = 1

    # 训练模型
    model = train_model(train_loader, test_loader, input_size, output_size, seq_len, pred_len, device, target_max, target_min)
