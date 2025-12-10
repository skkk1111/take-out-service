import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import random
from scipy import stats
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 加载本地中文字体
zh_font = FontProperties(fname="C:/Windows/Fonts/simsun.ttc", size=12)

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示异常


# 设置随机种子以确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


# 定义地理空间感知的深度学习模型
class GeoAwareNN(nn.Module):
    def __init__(self, input_dim, spatial_dim, hidden_dim=64, num_layers=3, dropout=0.2):
        """
        地理空间感知的神经网络模型

        参数:
            input_dim: 特征维度
            spatial_dim: 空间坐标维度 (通常为2)
            hidden_dim: 隐藏层维度
            num_layers: 隐藏层数量
            dropout: Dropout比率，用于防止过拟合
        """
        super(GeoAwareNN, self).__init__()

        # 特征处理网络
        feature_layers = []
        feature_layers.append(nn.Linear(input_dim, hidden_dim))
        feature_layers.append(nn.ReLU())
        feature_layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            feature_layers.append(nn.Linear(hidden_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            feature_layers.append(nn.Dropout(dropout))

        self.feature_net = nn.Sequential(*feature_layers)

        # 空间处理网络
        spatial_layers = []
        spatial_layers.append(nn.Linear(spatial_dim, hidden_dim))
        spatial_layers.append(nn.ReLU())
        spatial_layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            spatial_layers.append(nn.Linear(hidden_dim, hidden_dim))
            spatial_layers.append(nn.ReLU())
            spatial_layers.append(nn.Dropout(dropout))

        self.spatial_net = nn.Sequential(*spatial_layers)

        # 合并网络
        merge_layers = []
        merge_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        merge_layers.append(nn.ReLU())
        merge_layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            merge_layers.append(nn.Linear(hidden_dim, hidden_dim))
            merge_layers.append(nn.ReLU())
            merge_layers.append(nn.Dropout(dropout))

        merge_layers.append(nn.Linear(hidden_dim, 1))

        self.merge_net = nn.Sequential(*merge_layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, coords):
        """
        前向传播

        参数:
            x: 特征张量 [batch_size, input_dim]
            coords: 空间坐标张量 [batch_size, spatial_dim]
        """
        # 处理特征
        x_features = self.feature_net(x)

        # 处理空间坐标
        x_spatial = self.spatial_net(coords)

        # 合并特征和空间信息
        x_combined = torch.cat([x_features, x_spatial], dim=1)

        # 最终预测
        output = self.merge_net(x_combined)

        return output


# 自定义数据集类
class GeoDataset(Dataset):
    def __init__(self, features, coords, targets, weights=None):
        self.features = torch.FloatTensor(features)
        self.coords = torch.FloatTensor(coords)
        self.targets = torch.FloatTensor(targets)

        if weights is None:
            self.weights = torch.ones(len(targets))
        else:
            self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'coords': self.coords[idx],
            'targets': self.targets[idx],
            'weights': self.weights[idx]
        }


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=10):
    """
    训练模型并进行验证

    参数:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备 (CPU或GPU)
        epochs: 训练轮数
        patience: 早停等待轮数
    """
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch['features'].to(device)
            coords = batch['coords'].to(device)
            targets = batch['targets'].to(device)
            weights = batch['weights'].to(device)

            # 前向传播
            outputs = model(features, coords)
            loss = criterion(outputs.squeeze(), targets)

            # 应用样本权重
            weighted_loss = (loss * weights).mean()

            # 反向传播和优化
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            train_loss += weighted_loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                coords = batch['coords'].to(device)
                targets = batch['targets'].to(device)
                weights = batch['weights'].to(device)

                outputs = model(features, coords)
                loss = criterion(outputs.squeeze(), targets)
                weighted_loss = (loss * weights).mean()

                val_loss += weighted_loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 加载最佳模型
    model.load_state_dict(best_model)
    return model, train_losses, val_losses


# 预测函数
def predict_model(model, data_loader, device):
    """使用模型进行预测"""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            coords = batch['coords'].to(device)
            targets = batch['targets'].to(device)

            outputs = model(features, coords)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy())

    return np.array(predictions), np.array(actuals)


# 计算局部系数函数
def compute_local_coefficients(model, features, coords, scaler, device):
    """
    计算局部系数（特征重要性）

    参数:
        model: 训练好的模型
        features: 特征数据
        coords: 空间坐标
        scaler: 特征标准化器
        device: 计算设备
    """
    model.eval()
    n_samples = len(features)
    n_features = features.shape[1]
    local_coefs = np.zeros((n_samples, n_features))
    std_errors = np.zeros((n_samples, n_features))  # 新增：存储标准误差

    # 转换为张量
    features_tensor = torch.FloatTensor(features).to(device)
    coords_tensor = torch.FloatTensor(coords).to(device)

    # 设置requires_grad为True以计算梯度
    features_tensor.requires_grad = True

    with torch.no_grad():
        # 获取原始预测
        original_outputs = model(features_tensor, coords_tensor).detach()

    # 计算每个特征的梯度
    for i in range(n_features):
        model.zero_grad()
        features_tensor.requires_grad = True

        # 计算输出对特征的梯度
        outputs = model(features_tensor, coords_tensor)
        torch.sum(outputs).backward()

        # 获取梯度
        gradients = features_tensor.grad.cpu().numpy()[:, i]

        # 乘以特征的标准差以获取标准化的重要性
        feature_std = scaler.scale_[i] if hasattr(scaler, 'scale_') else 1.0
        local_coefs[:, i] = gradients * feature_std

        # 计算标准误差（假设梯度近似正态分布）
        std_errors[:, i] = np.std(gradients) / np.sqrt(n_samples)  # 新增：计算标准误差

    return local_coefs, std_errors


# 主分析代码
def main():
    # 1. 加载数据
    data_path = r"D:\文件\研究生文件\小论文\外卖\地理加权回归\MGWR\地理加权回归11.xlsx"
    data = pd.read_excel(data_path)

    # 定义目标文件夹路径
    output_dir = os.path.join(os.path.dirname(data_path), "results11")  # 新增：指定结果保存目录
    os.makedirs(output_dir, exist_ok=True)  # 确保目标文件夹存在

    # 定义特征列
    X_cols = ['Population size', 'Road density', 'University',
              'Hospital', 'Office buildings',
              'Shopping center', 'Metro station',
              'Bus stop', 'Residential areas']

    # 检查必要列
    required_cols = ['weight_C', 'log', 'lat', 'Y'] + X_cols
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"数据中缺少以下列: {missing_cols}")

    # 提取权重列
    weights = data['weight_C'].values

    # 提取其他数据
    coords = data[['log', 'lat']].values
    Y = data['Y'].values.reshape(-1, 1)
    X = data[X_cols].values

    # 2. 数据探索与预处理
    print("\n数据探索:")
    print(f"数据行数: {len(data)}, 列数: {len(data.columns)}")
    print(f"Y变量统计: 均值={Y.mean():.4f}, 标准差={Y.std():.4f}, 最小值={Y.min():.4f}, 最大值={Y.max():.4f}")

    # 检查特征相关性
    corr_matrix = np.corrcoef(X.T, Y.flatten())
    corr_with_y = corr_matrix[:-1, -1]
    print("\n特征与Y的相关性:")
    for i, col in enumerate(X_cols):
        print(f"{col}: {corr_with_y[i]:.4f}")

    # 3. 划分训练测试集
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.2, random_state=42
    )

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    coords_train, coords_test = coords[train_idx], coords[test_idx]
    weights_train, weights_test = weights[train_idx], weights[test_idx]

    print(f"\n训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # 4. 特征标准化
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    # 坐标标准化
    coord_scaler = StandardScaler()
    coords_train_scaled = coord_scaler.fit_transform(coords_train)
    coords_test_scaled = coord_scaler.transform(coords_test)

    # 目标变量标准化
    target_scaler = StandardScaler()
    Y_train_scaled = target_scaler.fit_transform(Y_train).flatten()
    Y_test_scaled = target_scaler.transform(Y_test).flatten()

    # 5. 创建数据集和数据加载器
    train_dataset = GeoDataset(X_train_scaled, coords_train_scaled, Y_train_scaled, weights_train)
    test_dataset = GeoDataset(X_test_scaled, coords_test_scaled, Y_test_scaled, weights_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 7. 初始化模型
    model = GeoAwareNN(
        input_dim=X_train.shape[1],
        spatial_dim=coords_train.shape[1],
        hidden_dim=128,
        num_layers=3,
        dropout=0.2
    ).to(device)

    # 8. 定义损失函数和优化器
    criterion = nn.MSELoss(reduction='none')  # 不减少，以便应用样本权重
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 9. 训练模型
    print("\n开始训练模型...")
    model, train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, device, epochs=200, patience=50
    )

    # 10. 可视化训练过程
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次', fontname='SimSun')
    plt.ylabel('损失', fontname='SimSun')
    plt.title('模型训练过程', fontname='SimSun')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=300)  # 修改：保存到目标文件夹
    plt.close()

    # 11. 评估模型
    print("\n评估模型...")
    train_predictions_scaled, train_actuals_scaled = predict_model(model, train_loader, device)
    test_predictions_scaled, test_actuals_scaled = predict_model(model, test_loader, device)

    # 反标准化预测值和实际值
    train_predictions = target_scaler.inverse_transform(train_predictions_scaled.reshape(-1, 1)).flatten()
    train_actuals = target_scaler.inverse_transform(train_actuals_scaled.reshape(-1, 1)).flatten()
    test_predictions = target_scaler.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()
    test_actuals = target_scaler.inverse_transform(test_actuals_scaled.reshape(-1, 1)).flatten()

    # 计算评估指标
    train_mse = mean_squared_error(train_actuals, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(train_actuals, train_predictions)

    test_mse = mean_squared_error(test_actuals, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(test_actuals, test_predictions)

    print(f"训练集 - MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"测试集 - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

    # 12. 计算局部系数（特征重要性）
    print("\n计算局部系数...")
    train_local_coefs, train_std_errors = compute_local_coefficients(
        model, X_train_scaled, coords_train_scaled, feature_scaler, device
    )
    test_local_coefs, test_std_errors = compute_local_coefficients(
        model, X_test_scaled, coords_test_scaled, feature_scaler, device
    )

    # 13. 准备结果数据
    def prepare_results_data(coords_data, X_data, Y_data, weights_data, predictions, local_coefs, std_errors, indices):
        """准备结果数据的辅助函数"""
        results_data = {}
        n = len(coords_data)

        # 获取FID
        try:
            fid = data.iloc[indices]['FID'].values if 'FID' in data.columns else indices
        except:
            fid = np.arange(n)

        results_data['FID'] = fid
        results_data['log'] = coords_data[:, 0]
        results_data['lat'] = coords_data[:, 1]
        results_data['weight_C'] = weights_data
        results_data['actual'] = Y_data.flatten()
        results_data['predicted'] = predictions
        results_data['residual'] = Y_data.flatten() - predictions

        # 添加每个自变量的系数和p值
        for i, col in enumerate(X_cols):
            results_data[f'{col}_coef'] = local_coefs[:, i]
            t_statistic = local_coefs[:, i] / std_errors[:, i]  # 计算t统计量
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=n-1))  # 计算p值
            results_data[f'{col}_pvalue'] = p_value  # 新增：添加p值

        return results_data

    # 14. 准备训练集和测试集结果
    train_results_data = prepare_results_data(
        coords_train, X_train, Y_train, weights_train, train_predictions, train_local_coefs, train_std_errors, train_idx
    )

    test_results_data = prepare_results_data(
        coords_test, X_test, Y_test, weights_test, test_predictions, test_local_coefs, test_std_errors, test_idx
    )

    # 15. 创建DataFrame
    train_results_df = pd.DataFrame(train_results_data)
    test_results_df = pd.DataFrame(test_results_data)

    # 16. 保存到Excel文件
    excel_path = os.path.join(output_dir, "geo_nn_results.xlsx")  # 修改：保存到目标文件夹
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 保存训练集结果
        train_results_df.to_excel(writer, sheet_name='Training_Results', index=False)

        # 保存测试集结果
        test_results_df.to_excel(writer, sheet_name='Testing_Results', index=False)

        # 添加模型摘要信息
        summary_data = {
            'Metric': ['训练集观测数', '测试集观测数', '变量数',
                       '训练集MSE', '测试集MSE', '训练集R²', '测试集R²','训练集RMSE', '测试集RMSE'],
            'Value': [len(X_train), len(X_test), len(X_cols),
                      f"{train_mse:.4f}", f"{test_mse:.4f}", f"{train_r2:.4f}", f"{test_r2:.4f}", f"{train_rmse:.4f}", f"{test_rmse:.4f}"]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Model_Summary', index=False)

        # 添加特征重要性
        feature_importance = []
        for i, col in enumerate(X_cols):
            train_coefs = train_local_coefs[:, i]
            test_coefs = test_local_coefs[:, i]
            feature_importance.append({
                'Variable': col,
                'Train_Mean_Importance': np.mean(np.abs(train_coefs)),
                'Train_Std_Importance': np.std(train_coefs),
                'Test_Mean_Importance': np.mean(np.abs(test_coefs)),
                'Test_Std_Importance': np.std(test_coefs)
            })

        feature_importance_df = pd.DataFrame(feature_importance)
        feature_importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

    print(f"\nExcel文件已保存到: {excel_path}")

    # 17. 可视化预测结果与实际值对比
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(train_actuals, train_predictions, alpha=0.5)
    plt.plot([train_actuals.min(), train_actuals.max()], [train_actuals.min(), train_actuals.max()], 'r--')
    plt.title('训练集 - 实际值 vs 预测值', fontname='SimSun')
    plt.xlabel('实际值', fontname='SimSun')
    plt.ylabel('预测值', fontname='SimSun')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(test_actuals, test_predictions, alpha=0.5)
    plt.plot([test_actuals.min(), test_actuals.max()], [test_actuals.min(), test_actuals.max()], 'r--')
    plt.title('测试集 - 实际值 vs 预测值', fontname='SimSun')
    plt.xlabel('实际值', fontname='SimSun')
    plt.ylabel('预测值', fontname='SimSun')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_vs_actual.png"), dpi=300, bbox_inches='tight')  # 修改：保存到目标文件夹
    plt.close()

    print("预测值与实际值对比图已保存")

    # 18. 可视化局部系数空间分布
    coef_output_dir = os.path.join(output_dir, "geo_nn_coef_maps")  # 修改：子文件夹路径
    os.makedirs(coef_output_dir, exist_ok=True)  # 确保子文件夹存在

    for i, col in enumerate(X_cols):
        # 训练集系数分布
        plt.figure(figsize=(8, 6))
        gdf = gpd.GeoDataFrame(
            train_results_df, geometry=gpd.points_from_xy(train_results_df['log'], train_results_df['lat'])
        )
        gdf.plot(column=f'{col}_coef', cmap='coolwarm', legend=True)
        plt.title(f'训练集 - 地理神经网络: {col}系数空间分布', fontname='SimSun')
        plt.xlabel('Longitude', fontname='SimSun')
        plt.ylabel('Latitude', fontname='SimSun')
        plt.tight_layout()
        plt.savefig(os.path.join(coef_output_dir, f"train_{col}_coef.png"), dpi=300, bbox_inches='tight')  # 修改：保存到目标文件夹
        plt.close()

        # 测试集系数分布
        plt.figure(figsize=(8, 6))
        gdf = gpd.GeoDataFrame(
            test_results_df, geometry=gpd.points_from_xy(test_results_df['log'], test_results_df['lat'])
        )
        gdf.plot(column=f'{col}_coef', cmap='coolwarm', legend=True)
        plt.title(f'测试集 - 地理神经网络: {col}系数空间分布', fontname='SimSun')
        plt.xlabel('Longitude', fontname='SimSun')
        plt.ylabel('Latitude', fontname='SimSun')
        plt.tight_layout()
        plt.savefig(os.path.join(coef_output_dir, f"test_{col}_coef.png"), dpi=300, bbox_inches='tight')  # 修改：保存到目标文件夹
        plt.close()

    print(f"局部系数空间分布图已保存到: {coef_output_dir}")


if __name__ == "__main__":
    main()