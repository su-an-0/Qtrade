import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
import talib
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler
import os
from pathlib import Path
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import logging
import argparse
import json

# # 检查并设置分布式训练环境
# rank = os.environ.get('RANK', -1)
# print("rank:", rank)
# print("world_size:", os.environ.get("WORLD_SIZE", -1))
# print("local_rank:", os.environ.get("LOCAL_RANK", -1))

# # 仅在远程环境安装必要依赖
# if rank != -1:
#     os.system("pip3 install https://automl-nni.oss-cn-beijing.aliyuncs.com/nni/hpo_tools/hpo_tools-0.2.1-py3-none-any.whl")
#     try:
#         import torch
#     except ImportError:
#         os.system("pip3 install torch torchvision --trusted-host yum.tbsite.net -i http://yum.tbsite.net/pypi/simple/")

# 初始化分布式训练模式
def init_distributed_mode(args):
    """初始化分布式训练环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args['rank'] = int(os.environ["RANK"])
        args['world_size'] = int(os.environ["WORLD_SIZE"])

    elif "SLURM_PROCID" in os.environ:
        args['rank'] = int(os.environ["SLURM_PROCID"])

    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args['distributed'] = False
        return

    args['distributed'] = True
    if torch.cuda.device_count() > 0:
        args['local_rank'] = args['rank'] % torch.cuda.device_count()
    if args['use_cuda']:
        torch.cuda.set_device(args['local_rank'])
    print(
        f"| distributed init (rank {args['rank']}): {args['dist_url']}, "
        f"local rank:{args['local_rank']}, world size:{args['world_size']}",
        flush=True)
    dist.init_process_group(backend=args['backend'],
                            init_method=args['dist_url'],
                            world_size=args['world_size'],
                            rank=args['rank'])
    print("init success")

# --------------------------
# 1. Time2Vec时间编码
# --------------------------
class Time2Vec(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(Time2Vec, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # 线性分量
        self.periodic = nn.Linear(input_dim, embed_dim - 1)  # 周期分量

    def forward(self, x):
        # x: [batch_size, seq_len, 1]
        linear = self.linear(x)  # [batch_size, seq_len, 1]
        periodic = torch.sin(self.periodic(x))  # [batch_size, seq_len, embed_dim-1]
        return torch.cat([linear, periodic], dim=-1)  # 拼接后作为时间编码


# --------------------------
# 2. 经验小波变换(EWT)实现
# --------------------------
def ewt_decompose(signal, n_components=3):
    """简化版EWT分解"""
    # 1. 计算信号傅里叶变换
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal))
    amp = np.abs(fft)
    
    # 2. 寻找频谱峰值作为频段分割点
    peaks, _ = find_peaks(amp[:len(amp)//2], height=0.1*np.max(amp))
    # 确保有足够的峰值，如果不够则使用均匀分割
    if len(peaks) < n_components - 1:
        peaks = np.linspace(0, len(amp)//2, n_components, endpoint=False)[1:].astype(int)
    else:
        peaks = sorted(peaks[:n_components-1])
    boundaries = [0] + peaks + [len(amp)//2]
    
    # 3. 分解为n_components个分量
    components = []
    for i in range(n_components):
        mask = (freq >= boundaries[i]/len(signal)) & (freq <= boundaries[i+1]/len(signal))
        mask |= (freq <= -boundaries[i]/len(signal)) & (freq >= -boundaries[i+1]/len(signal))
        component_fft = fft * mask
        component = np.fft.ifft(component_fft).real
        components.append(component)
    return np.array(components)


# --------------------------
# 3. Transformer模型
# --------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers, dropout=0.3):
        super(TransformerEncoder, self).__init__()
        
        # 检查并确保input_dim能被num_heads整除
        if input_dim % num_heads != 0:
            # 调整input_dim使其能被num_heads整除
            input_dim = ((input_dim // num_heads) + 1) * num_heads
            print(f"调整输入维度为 {input_dim} 以满足多头注意力要求")
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.input_dim = input_dim  # 保存调整后的维度

    def forward(self, x):
        # 如果输入维度与预期不符，进行调整
        if x.size(-1) != self.input_dim:
            x = nn.functional.pad(x, (0, self.input_dim - x.size(-1)))
        
        for _ in range(self.num_layers):
            # 多头注意力
            attn_output, _ = self.attention(x, x, x)
            x = self.norm1(x + self.dropout(attn_output))
            # 前馈网络
            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_output))
        return x


class Level1Transformer(nn.Module):
    """第一级Transformer：处理EWT分解后的分量"""
    def __init__(self, ewt_dim=16, time_embed_dim=16, num_heads=4, ff_dim=32, num_layers=3):
        super(Level1Transformer, self).__init__()
        self.time2vec = Time2Vec(input_dim=1, embed_dim=time_embed_dim)
        
        # 确保输入维度能被注意力头数整除
        self.input_dim = ewt_dim + time_embed_dim
        if self.input_dim % num_heads != 0:
            # 调整时间嵌入维度以满足条件
            time_embed_dim += num_heads - (self.input_dim % num_heads)
            self.time2vec = Time2Vec(input_dim=1, embed_dim=time_embed_dim)
            self.input_dim = ewt_dim + time_embed_dim
            print(f"调整Level1时间嵌入维度为 {time_embed_dim} 以满足多头注意力要求")
        
        self.transformer = TransformerEncoder(
            input_dim=self.input_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(self.input_dim, ewt_dim)

    def forward(self, ewt_component, time_features):
        time_embed = self.time2vec(time_features)
        x = torch.cat([ewt_component, time_embed], dim=-1)
        x = self.transformer(x)
        return self.output_layer(x)


class Level2Transformer(nn.Module):
    """第二级Transformer：融合所有特征并预测高低价"""
    def __init__(self, input_dim, time_embed_dim=16, num_heads=4, ff_dim=64, num_layers=4):
        super(Level2Transformer, self).__init__()
        
        # 确保输入维度能被注意力头数整除
        self.input_dim = input_dim + time_embed_dim
        if self.input_dim % num_heads != 0:
            # 调整时间嵌入维度以满足条件
            time_embed_dim += num_heads - (self.input_dim % num_heads)
            self.input_dim = input_dim + time_embed_dim
            print(f"调整Level2时间嵌入维度为 {time_embed_dim} 以满足多头注意力要求")
        
        self.time2vec = Time2Vec(input_dim=1, embed_dim=time_embed_dim)
        self.transformer = TransformerEncoder(
            input_dim=self.input_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(self.input_dim, 2)

    def forward(self, features, time_features):
        time_embed = self.time2vec(time_features)
        x = torch.cat([features, time_embed], dim=-1)
        x = self.transformer(x)
        x = x[:, -1, :]  # 取序列最后一个时间步的特征
        return self.output_layer(x)


# --------------------------
# 4. 完整模型整合
# --------------------------
class TradingModel(nn.Module):
    def __init__(self, ewt_components=3, ewt_dim=16, tech_dim=2):
        super(TradingModel, self).__init__()
        # 第一级Transformer：处理每个EWT分量
        self.level1 = nn.ModuleList([
            Level1Transformer(ewt_dim=ewt_dim) for _ in range(ewt_components)
        ])
        # 第二级Transformer：融合EWT处理结果+技术指标
        total_input_dim = ewt_components * ewt_dim + tech_dim
        self.level2 = Level2Transformer(input_dim=total_input_dim)
        self.ewt_components = ewt_components
        self.ewt_dim = ewt_dim
        self.tech_dim = tech_dim

    def forward(self, ewt_components, tech_indicators, time_features):
        # 1. 第一级处理每个EWT分量
        level1_outputs = []
        for i in range(self.ewt_components):
            component = ewt_components[:, :, i, :]
            level1_out = self.level1[i](component, time_features)
            level1_outputs.append(level1_out)
        ewt_features = torch.cat(level1_outputs, dim=-1)
        
        # 2. 融合EWT特征和技术指标
        all_features = torch.cat([ewt_features, tech_indicators], dim=-1)
        
        # 3. 第二级预测高低价
        high_pred, low_pred = self.level2(all_features, time_features).chunk(2, dim=-1)
        return high_pred.squeeze(-1), low_pred.squeeze(-1)


# --------------------------
# 5. 数据处理与加载函数
# --------------------------
def load_binance_data(currency, timeframe):
    """加载币安数据"""
    # 阿里云环境下可能需要调整数据路径
    data_dir = os.environ.get('DATA_DIR', 'binance_data')
    filename = f"{data_dir}/{currency.lower()}{'usdt'}_{timeframe}.parquet"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"数据文件不存在: {filename}")
    
    df = pd.read_parquet(filename).copy()
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not set(required_columns).issubset(df.columns):
        raise ValueError(f"数据文件缺少必要的列。需要: {required_columns}")
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def preprocess_data(df, seq_len=16, ewt_components=3):
    """处理数据：计算指标、EWT分解、归一化"""
    df = df.copy()
    
    # 计算技术指标
    df['bb_upper'], df['bb_mid'], df['bb_lower'] = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    df['atr'] = talib.ATR(
        df['high'], df['low'], df['close'], timeperiod=14
    )
    df = df.dropna().reset_index(drop=True)
    
    # 计算对数收益率
    df.loc[:, 'log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    # EWT分解
    try:
        ewt_components_vals = ewt_decompose(df['log_return'].values, n_components=ewt_components)
    except:
        print("警告：EWT分解失败，使用随机数据替代")
        ewt_components_vals = np.random.randn(ewt_components, len(df))
    
    # 归一化
    tech_scaler = MinMaxScaler()
    price_scaler = MinMaxScaler()
    
    tech_indicators = tech_scaler.fit_transform(df[['bb_upper', 'atr']].values)
    prices = df[['open', 'high', 'low']].values
    normalized_prices = price_scaler.fit_transform(prices)
    
    # 构建滑动窗口序列
    X_ewt, X_tech, X_time = [], [], []
    y_high, y_low = [], []
    
    # 添加进度条用于数据预处理
    for i in tqdm(range(seq_len, len(df)), desc="预处理数据", unit="样本"):
        ewt_seq = ewt_components_vals[:, i-seq_len:i].T
        X_ewt.append(ewt_seq)
        tech_seq = tech_indicators[i-seq_len:i]
        X_tech.append(tech_seq)
        time_seq = np.arange(i-seq_len, i).reshape(-1, 1) / len(df)
        X_time.append(time_seq)
        y_high.append(normalized_prices[i, 1])
        y_low.append(normalized_prices[i, 2])
    
    if len(X_ewt) == 0:
        return None
    
    # 优化张量创建并移至设备
    X_ewt_np = np.array(X_ewt)
    X_tech_np = np.array(X_tech)
    X_time_np = np.array(X_time)
    y_high_np = np.array(y_high)
    y_low_np = np.array(y_low)
    
    result = {
        'ewt': torch.FloatTensor(X_ewt_np).unsqueeze(-1),
        'tech': torch.FloatTensor(X_tech_np),
        'time': torch.FloatTensor(X_time_np),
        'high': torch.FloatTensor(y_high_np),
        'low': torch.FloatTensor(y_low_np),
        'original_high': prices[seq_len:, 1],
        'original_low': prices[seq_len:, 2],
        'open_prices': prices[seq_len:, 0],
        'price_scaler': price_scaler,
        'tech_scaler': tech_scaler,
        'df': df.iloc[seq_len:].reset_index(drop=True)
    }
    
    return result


# --------------------------
# 6. 模型训练与评估函数
# --------------------------
def train_model(model, train_data, val_data, args, device):
    # 将模型移至设备
    model = model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-5)
    
    # 分布式训练包装
    if args['distributed']:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args['local_rank']], output_device=args['local_rank'])
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae_high': [],
        'train_mae_low': [],
        'val_mae_high': [],
        'val_mae_low': []
    }
    
    best_val_loss = float('inf')
    counter = 0
    num_train_batches = len(train_data['ewt']) // args['batch_size']
    total_train_batches = num_train_batches + (1 if len(train_data['ewt']) % args['batch_size'] != 0 else 0)
    
    # 使用tqdm创建训练进度条（仅主进程显示）
    if args['rank'] == 0 or not args['distributed']:
        epoch_pbar = tqdm(range(args['epochs']), desc=f"训练模型", unit="轮")
    else:
        epoch_pbar = range(args['epochs'])
    
    for epoch in epoch_pbar:
        model.train()
        total_train_loss = 0
        
        # 为每个epoch创建batch进度条（仅主进程显示）
        if args['rank'] == 0 or not args['distributed']:
            batch_pbar = tqdm(range(total_train_batches), 
                             desc=f"Epoch {epoch+1}/{args['epochs']} 批次进度", 
                             unit="批", leave=False)
        else:
            batch_pbar = range(total_train_batches)
        
        for batch_idx in batch_pbar:
            start_idx = batch_idx * args['batch_size']
            end_idx = start_idx + args['batch_size']
            
            # 准备批次数据
            batch_ewt = train_data['ewt'][start_idx:end_idx].to(device)
            batch_tech = train_data['tech'][start_idx:end_idx].to(device)
            batch_time = train_data['time'][start_idx:end_idx].to(device)
            batch_high = train_data['high'][start_idx:end_idx].to(device)
            batch_low = train_data['low'][start_idx:end_idx].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            high_pred, low_pred = model(batch_ewt, batch_tech, batch_time)
            
            # 计算损失
            loss_high = criterion(high_pred, batch_high)
            loss_low = criterion(low_pred, batch_low)
            train_loss = loss_high + loss_low
            
            # 反向传播和优化
            train_loss.backward()
            optimizer.step()
            
            total_train_loss += train_loss.item()
            
            # 更新batch进度条的描述（仅主进程）
            if args['rank'] == 0 or not args['distributed']:
                batch_pbar.set_postfix({"batch_loss": f"{train_loss.item():.6f}"})
        
        # 计算平均训练损失
        avg_train_loss = total_train_loss / total_train_batches
        
        # 在训练集上计算MAE（使用最后一个批次）
        with torch.no_grad():
            mae_high = mean_absolute_error(high_pred.cpu().numpy(), batch_high.cpu().numpy())
            mae_low = mean_absolute_error(low_pred.cpu().numpy(), batch_low.cpu().numpy())
        
        # 在验证集上评估
        model.eval()
        total_val_loss = 0
        val_high_preds = []
        val_low_preds = []
        val_high_trues = []
        val_low_trues = []
        
        with torch.no_grad():
            num_val_batches = len(val_data['ewt']) // args['batch_size']
            total_val_batches = num_val_batches + (1 if len(val_data['ewt']) % args['batch_size'] != 0 else 0)
            
            # 为验证过程创建进度条（仅主进程）
            if args['rank'] == 0 or not args['distributed']:
                val_pbar = tqdm(range(total_val_batches), 
                               desc=f"Epoch {epoch+1}/{args['epochs']} 验证进度", 
                               unit="批", leave=False)
            else:
                val_pbar = range(total_val_batches)
            
            for batch_idx in val_pbar:
                start_idx = batch_idx * args['batch_size']
                end_idx = start_idx + args['batch_size']
                
                batch_ewt = val_data['ewt'][start_idx:end_idx].to(device)
                batch_tech = val_data['tech'][start_idx:end_idx].to(device)
                batch_time = val_data['time'][start_idx:end_idx].to(device)
                batch_high = val_data['high'][start_idx:end_idx].to(device)
                batch_low = val_data['low'][start_idx:end_idx].to(device)
                
                high_pred, low_pred = model(batch_ewt, batch_tech, batch_time)
                
                loss_high = criterion(high_pred, batch_high)
                loss_low = criterion(low_pred, batch_low)
                total_val_loss += (loss_high + loss_low).item()
                
                val_high_preds.append(high_pred.cpu().numpy())
                val_low_preds.append(low_pred.cpu().numpy())
                val_high_trues.append(batch_high.cpu().numpy())
                val_low_trues.append(batch_low.cpu().numpy())
        
        # 计算平均验证损失
        avg_val_loss = total_val_loss / total_val_batches
        
        # 计算验证集MAE（仅主进程）
        if args['rank'] == 0 or not args['distributed']:
            val_high_preds = np.concatenate(val_high_preds)
            val_low_preds = np.concatenate(val_low_preds)
            val_high_trues = np.concatenate(val_high_trues)
            val_low_trues = np.concatenate(val_low_trues)
            
            val_mae_high = mean_absolute_error(val_high_preds, val_high_trues)
            val_mae_low = mean_absolute_error(val_low_preds, val_low_trues)
            
            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_mae_high'].append(mae_high)
            history['train_mae_low'].append(mae_low)
            history['val_mae_high'].append(val_mae_high)
            history['val_mae_low'].append(val_mae_low)
            
            # 更新epoch进度条的描述
            epoch_pbar.set_postfix({
                "train_loss": f"{avg_train_loss:.6f}",
                "val_loss": f"{avg_val_loss:.6f}",
                " batches": f"{total_train_batches}"
            })
        
        # 早停机制和保存最佳模型（仅主进程）
        if (args['rank'] == 0 or not args['distributed']) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            Path(args['save_model']).mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model_without_ddp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_val_loss,
                'ewt_components': model_without_ddp.ewt_components,
                'ewt_dim': model_without_ddp.ewt_dim,
                'tech_dim': model_without_ddp.tech_dim
            }, f"{args['save_model']}/best.pth")
        elif args['rank'] == 0 or not args['distributed']:
            counter += 1
            if counter >= args['patience']:
                if args['rank'] == 0 or not args['distributed']:
                    tqdm.write(f"早停在第 {epoch+1} 轮")
                break
    
    # 保存最终模型（仅主进程）
    if args['rank'] == 0 or not args['distributed']:
        torch.save({
            'model_state_dict': model_without_ddp.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_train_loss,
            'ewt_components': model_without_ddp.ewt_components,
            'ewt_dim': model_without_ddp.ewt_dim,
            'tech_dim': model_without_ddp.tech_dim
        }, f"{args['save_model']}/final.pth")
    
    return model_without_ddp, history

def evaluate_model(model, test_data, price_scaler, args, device):
    # 将模型移至设备
    model = model.to(device)
    model.eval()
    
    high_preds = []
    low_preds = []
    high_trues = []
    low_trues = []
    
    with torch.no_grad():
        num_batches = len(test_data['ewt']) // args['batch_size']
        total_batches = num_batches + (1 if len(test_data['ewt']) % args['batch_size'] != 0 else 0)
        
        # 评估进度条（仅主进程）
        if args['rank'] == 0 or not args['distributed']:
            eval_pbar = tqdm(range(total_batches), desc="评估模型", unit="批")
        else:
            eval_pbar = range(total_batches)
        
        for batch_idx in eval_pbar:
            start_idx = batch_idx * args['batch_size']
            end_idx = start_idx + args['batch_size']
            
            batch_ewt = test_data['ewt'][start_idx:end_idx].to(device)
            batch_tech = test_data['tech'][start_idx:end_idx].to(device)
            batch_time = test_data['time'][start_idx:end_idx].to(device)
            batch_high = test_data['high'][start_idx:end_idx].to(device)
            batch_low = test_data['low'][start_idx:end_idx].to(device)
            
            high_pred, low_pred = model(batch_ewt, batch_tech, batch_time)
            
            # 将结果移回CPU并转换为numpy
            high_preds.append(high_pred.cpu().numpy())
            low_preds.append(low_pred.cpu().numpy())
            high_trues.append(batch_high.cpu().numpy())
            low_trues.append(batch_low.cpu().numpy())
    
    # 只在主进程处理评估结果
    if args['rank'] == 0 or not args['distributed']:
        # 合并所有批次结果
        high_pred = np.concatenate(high_preds)
        low_pred = np.concatenate(low_preds)
        high_true = np.concatenate(high_trues)
        low_true = np.concatenate(low_trues)
    
        # 反归一化
        pred_array = np.column_stack((np.zeros_like(high_pred), high_pred, low_pred))
        true_array = np.column_stack((np.zeros_like(high_true), high_true, low_true))
        
        pred_scaled = price_scaler.inverse_transform(pred_array)
        true_scaled = price_scaler.inverse_transform(true_array)
        
        high_pred_scaled = pred_scaled[:, 1]
        low_pred_scaled = pred_scaled[:, 2]
        high_true_scaled = true_scaled[:, 1]
        low_true_scaled = true_scaled[:, 2]
        
        # 计算评估指标
        metrics = {
            'high': {
                'mae': float(mean_absolute_error(high_true_scaled, high_pred_scaled)),
                'rmse': float(np.sqrt(mean_squared_error(high_true_scaled, high_pred_scaled))),
                'mape': float(np.mean(np.abs((high_true_scaled - high_pred_scaled) / high_true_scaled)) * 100)
            },
            'low': {
                'mae': float(mean_absolute_error(low_true_scaled, low_pred_scaled)),
                'rmse': float(np.sqrt(mean_squared_error(low_true_scaled, low_pred_scaled))),
                'mape': float(np.mean(np.abs((low_true_scaled - low_pred_scaled) / low_true_scaled)) * 100)
            },
            'overall': {
                'mae': float((mean_absolute_error(high_true_scaled, high_pred_scaled) + 
                       mean_absolute_error(low_true_scaled, low_pred_scaled)) / 2),
                'rmse': float((np.sqrt(mean_squared_error(high_true_scaled, high_pred_scaled)) +
                        np.sqrt(mean_squared_error(low_true_scaled, low_pred_scaled))) / 2),
                'mape': float((np.mean(np.abs((high_true_scaled - high_pred_scaled) / high_true_scaled)) +
                        np.mean(np.abs((low_true_scaled - low_pred_scaled) / low_true_scaled))) * 50)
            }
        }
        
        # 创建输出目录（如果不存在）
        Path(args['metric_filepath']).mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON格式（阿里云AutoML推荐格式）
        metric_path = f"{args['metric_filepath']}/metrics.json"
        with open(metric_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        
        # 同时输出一个简洁版指标，只包含主要评估指标（供平台快速识别）
        simple_metrics = {
            "MAE": metrics['overall']['mae'],
            "RMSE": metrics['overall']['rmse'],
            "MAPE": metrics['overall']['mape'],
            # 阿里云通常需要一个主要指标作为优化目标
            "default_metric": metrics['overall']['mae']
        }
        
        simple_metric_path = f"{args['metric_filepath']}/default_metric.json"
        with open(simple_metric_path, "w", encoding="utf-8") as f:
            json.dump(simple_metrics, f, ensure_ascii=False)
        
        return {
            'high_pred': high_pred_scaled,
            'low_pred': low_pred_scaled,
            'high_true': high_true_scaled,
            'low_true': low_true_scaled,
            'metrics': metrics
        }
    else:
        return None



# --------------------------
# 7. 可视化函数（仅主进程执行）
# --------------------------
def plot_training_history(history, title, filename, args):
    if args['rank'] != 0 and args['distributed']:
        return
        
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title(f'{title} - 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mae_high'], label='训练MAE (最高价)')
    plt.plot(history['train_mae_low'], label='训练MAE (最低价)')
    plt.plot(history['val_mae_high'], label='验证MAE (最高价)')
    plt.plot(history['val_mae_low'], label='验证MAE (最低价)')
    plt.title(f'{title} - MAE曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    Path(args['stdout_filepath']).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{args['stdout_filepath']}/{filename}_training_history.png")
    plt.close()

def plot_predictions(results, title, filename, args, n_samples=100):
    if args['rank'] != 0 and args['distributed']:
        return
        
    if len(results['high_true']) > n_samples:
        indices = np.linspace(0, len(results['high_true']) - 1, n_samples, dtype=int)
        high_true = results['high_true'][indices]
        high_pred = results['high_pred'][indices]
        low_true = results['low_true'][indices]
        low_pred = results['low_pred'][indices]
    else:
        high_true = results['high_true']
        high_pred = results['high_pred']
        low_true = results['low_true']
        low_pred = results['low_pred']
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(high_true, label='真实最高价')
    plt.plot(high_pred, label='预测最高价', linestyle='--')
    plt.title(f'{title} - 最高价预测')
    plt.xlabel('时间步')
    plt.ylabel('价格')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(low_true, label='真实最低价')
    plt.plot(low_pred, label='预测最低价价', linestyle='--')
    plt.title(f'{title} - 最低价预测')
    plt.xlabel('时间步')
    plt.ylabel('价格')
    plt.legend()
    
    plt.tight_layout()
    Path(args['stdout_filepath']).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{args['stdout_filepath']}/{filename}_predictions.png")
    plt.close()


# --------------------------
# 8. 参数配置与主函数
# --------------------------
def get_params():
    # 训练参数配置
    parser = argparse.ArgumentParser(description='Trading Price Prediction Model')
    # 数据参数
    parser.add_argument("--currency", type=str, default='btc', help="加密货币类型")
    parser.add_argument("--timeframe", type=str, default='5m', help="时间框架")
    parser.add_argument("--seq_len", type=int, default=64, help="序列长度")
    parser.add_argument("--ewt_components", type=int, default=3, help="EWT分解分量数")
    parser.add_argument("--ewt_dim", type=int, default=16, help="EWT特征维度")
    parser.add_argument("--tech_dim", type=int, default=2, help="技术指标维度")
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.00005, help='学习率')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--train_ratio', type=float, default=0.65, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.25, help='测试集比例')
    
    # 分布式训练参数
    parser.add_argument('--no-cuda', action='store_true', default=False, help='禁用CUDA训练')
    parser.add_argument('--local_rank', type=int, help='本地进程编号，由分布式框架传入')
    parser.add_argument("--world-size", default=1, type=int, help="分布式进程数量")
    parser.add_argument("--dist-url", default="env://", type=str, help="分布式训练URL")
    parser.add_argument("--backend", type=str, help='分布式后端',
                        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                        default=dist.Backend.GLOO)
    
    # 阿里云AutoML所需参数
    parser.add_argument("--save_model", type=str, default='../model', help='模型保存路径')
    parser.add_argument("--metric_filepath", default="../metric", type=str, help="指标保存路径")
    parser.add_argument("--stdout_filepath", default="../stdout", type=str, help="输出保存路径")
    
    args, _ = parser.parse_known_args()
    return args

def main():
    # 获取参数配置
    args = get_params()
    args_dict = vars(args)
    
    # 初始化设备
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args_dict['use_cuda'] = use_cuda
    
    # 初始化分布式训练
    init_distributed_mode(args_dict)
    
    # 设置设备
    device = torch.device("cuda" if use_cuda else "cpu")
    if args_dict['rank'] == 0 or not args_dict['distributed']:
        print(f"使用设备: {device}")
        if torch.cuda.is_available():
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    
    # 只在主进程执行数据加载和预处理
    if args_dict['rank'] == 0 or not args_dict['distributed']:
        try:
            print("加载数据...")
            df = load_binance_data(args.currency, args.timeframe)
            print(f"加载完成，共 {len(df)} 条数据")
            
            print("预处理数据...")
            processed_data = preprocess_data(
                df, 
                seq_len=args.seq_len,
                ewt_components=args.ewt_components
            )
            
            if processed_data is None:
                print("警告：数据预处理失败，无法继续训练")
                return
            
            total_samples = len(processed_data['high'])
            train_size = int(total_samples * args.train_ratio)
            val_size = int(total_samples * args.val_ratio)
            
            # 划分数据集
            train_data = {k: v[:train_size] for k, v in processed_data.items() 
                         if k in ['ewt', 'tech', 'time', 'high', 'low']}
            val_data = {k: v[train_size:train_size+val_size] for k, v in processed_data.items()
                       if k in ['ewt', 'tech', 'time', 'high', 'low']}
            test_data = {k: v[train_size+val_size:] for k, v in processed_data.items()
                        if k in ['ewt', 'tech', 'time', 'high', 'low']}
            
            print(f"数据集划分: 训练集 {train_size}, 验证集 {val_size}, 测试集 {total_samples - train_size - val_size}")
            print(f"使用批次大小: {args.batch_size}")
            
        except Exception as e:
            print(f"数据处理出错: {str(e)}")
            return
    else:
        # 非主进程初始化空数据结构
        train_data = val_data = test_data = processed_data = None
    
    # 等待主进程完成数据处理
    if args_dict['distributed']:
        dist.barrier()
    
    # 如果是分布式训练，广播数据到所有进程
    if args_dict['distributed'] and args_dict['rank'] != 0:
        # 这里简化处理，实际应用中可能需要更复杂的分布式数据加载
        pass
    
    # 仅主进程继续训练流程
    if args_dict['rank'] == 0 or not args_dict['distributed']:
        print("初始化模型...")
        model = TradingModel(
            ewt_components=args.ewt_components,
            ewt_dim=args.ewt_dim,
            tech_dim=args.tech_dim
        )
        
        print("开始训练模型...")
        start_time = time.time()
        model, history = train_model(
            model, 
            train_data, 
            val_data, 
            args_dict,
            device
        )
        training_time = time.time() - start_time
        print(f"训练完成，耗时 {training_time:.2f} 秒")
        
        # 绘制训练历史
        model_name = f"{args.currency}_{args.timeframe}"
        plot_training_history(history, f"{args.currency.upper()} {args.timeframe} 模型训练历史", model_name, args_dict)
        
        print("加载最佳模型...")
        checkpoint = torch.load(f"{args.save_model}/best.pth", map_location=device)
        best_model = TradingModel(
            ewt_components=checkpoint['ewt_components'],
            ewt_dim=checkpoint['ewt_dim'],
            tech_dim=checkpoint['tech_dim']
        )
        best_model.load_state_dict(checkpoint['model_state_dict'])
        
        print("评估模型...")
        results = evaluate_model(
            best_model, 
            test_data, 
            processed_data['price_scaler'],
            args_dict,
            device
        )
        
        if results:
            print("评估指标:")
            print(f"最高价 MAE: {results['metrics']['high']['mae']:.4f}")
            print(f"最高价 RMSE: {results['metrics']['high']['rmse']:.4f}")
            print(f"最高价 MAPE: {results['metrics']['high']['mape']:.2f}%")
            print(f"最低价 MAE: {results['metrics']['low']['mae']:.4f}")
            print(f"最低价 RMSE: {results['metrics']['low']['rmse']:.4f}")
            print(f"最低价 MAPE: {results['metrics']['low']['mape']:.2f}%")
            print(f"总体 MAE: {results['metrics']['overall']['mae']:.4f}")
            
            plot_predictions(
                results, 
                f"{args.currency.upper()} {args.timeframe} 模型预测结果", 
                model_name,
                args_dict
            )
    
    # 清理分布式进程组
    if args_dict['distributed']:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()


'''
    python3 /root/code/cloud_search.py \
    --seq_len=${seq_len} \
    --ewt_components=${ewt_components} \
    --ewt_dim=${ewt_dim} \
    --batch_size=${batch_size} \
    --epochs=${epochs} \
    --lr=${lr} \
    --train_ratio=${train_ratio} \
    --save_model=/root/code/model/model_${exp_id}_${trial_id} \
    --metric_filepath=/root/code/metric/metric_${exp_id}_${trial_id} \ '''
