import pickle
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
import onnx  # 用于验证ONNX模型合法性

# ---------------------- 1. 日志配置 ----------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------- 2. Criteo 数据预处理器（修复Jython兼容性） ----------------------
class DataPreprocessor(BaseEstimator, TransformerMixin):
    """自定义预处理器：处理缺失值、数值log转换、类别编码、标准化
    核心改进：仅使用原生Python类型，避免NumPy/Sklearn对象序列化，兼容Jython
    """

    def __init__(self, num_cols: List[str], cat_cols: List[str], min_category_freq: int = 100):
        self.num_cols = num_cols  # 数值特征列（I1~I13）
        self.cat_cols = cat_cols  # 类别特征列（C1~C26）
        self.min_category_freq = min_category_freq  # 类别低频阈值
        self.train_median_values: Dict[str, float] = {}  # 数值特征中位数（原生float）
        self.trained_categories: Dict[str, List[str]] = {}  # 高频类别列表（原生list）
        self.label_encoder_maps: Dict[str, Dict[str, int]] = {}  # 类别→编码映射（原生dict）
        self.scaler_mean: Dict[str, float] = {}  # 标准化均值（原生float）
        self.scaler_scale: Dict[str, float] = {}  # 标准化标准差（原生float）
        self._is_fitted = False  # 拟合状态标记

    def fit(self, X: pd.DataFrame, y=None):
        """用训练数据拟合预处理参数（仅保存原生Python类型）"""
        # 1. 处理数值特征：计算中位数（填充缺失值用）
        for col in self.num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')  # 无效值转NaN
            # 关键：NumPy中位数转原生float
            self.train_median_values[col] = float(X[col].median())

            # 2. 处理类别特征：筛选高频类别+构建编码映射（避免序列化OrdinalEncoder）
        for col in self.cat_cols:
            X[col] = X[col].fillna('UNK').astype(str)  # 缺失值填'UNK'
            freq = X[col].value_counts()  # 统计类别频率
            # 高频类别筛选（转原生list）
            high_freq_cats = freq[freq > self.min_category_freq].index.tolist()
            self.trained_categories[col] = high_freq_cats

            # 构建类别→编码映射（0开始，UNK编码为-1）
            cat_to_idx = {cat: idx for idx, cat in enumerate(high_freq_cats)}
            cat_to_idx['UNK'] = -1  # 明确UNK的编码
            self.label_encoder_maps[col] = cat_to_idx

        # 3. 拟合数值特征标准化（手动计算mean和scale，转原生float）
        numeric_data = self._prepare_numeric(X[self.num_cols])
        # 按列计算mean和scale，转原生float并存入字典
        for idx, col in enumerate(self.num_cols):
            self.scaler_mean[col] = float(numeric_data.iloc[:, idx].mean())
            self.scaler_scale[col] = float(numeric_data.iloc[:, idx].std())
            # 处理标准差为0的情况（避免Java端除以0）
            if self.scaler_scale[col] < 1e-9:
                self.scaler_scale[col] = 1e-9

        self._is_fitted = True
        logger.info("预处理器拟合完成（仅保存原生Python类型）")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """对数据应用预处理（需先调用fit）"""
        if not self._is_fitted:
            raise NotFittedError("预处理器未拟合，请先调用fit()")

        X_processed = X.copy()
        # 1. 处理数值特征：填充缺失值→log转换→标准化
        X_processed[self.num_cols] = self._process_numeric(X[self.num_cols])
        # 2. 处理类别特征：填充缺失值→过滤低频→编码
        for col in self.cat_cols:
            X_processed[col] = self._process_categorical(X[col], col)

        return X_processed

    def _prepare_numeric(self, X_num: pd.DataFrame) -> pd.DataFrame:
        """数值特征预处理（填充缺失值+log转换）"""
        processed = X_num.copy()
        for col in self.num_cols:
            processed[col] = pd.to_numeric(processed[col], errors='coerce')
            # 中位数填充（用原生float）
            processed[col] = processed[col].fillna(self.train_median_values[col])
            # log1p转换（避免log(0)，clip下限与Java端一致）
            processed[col] = np.log1p(processed[col].clip(lower=-0.999))
        return processed

    def _process_numeric(self, X_num: pd.DataFrame) -> pd.DataFrame:
        """数值特征最终处理（标准化：(x-mean)/scale）"""
        prepared = self._prepare_numeric(X_num)
        processed = prepared.copy()
        for col in self.num_cols:
            # 标准化计算（用原生float参数）
            processed[col] = (processed[col] - self.scaler_mean[col]) / self.scaler_scale[col]
        return processed

    def _process_categorical(self, X_cat: pd.Series, col: str) -> pd.Series:
        """单类别特征处理（填充→过滤低频→编码，用原生映射）"""
        # 缺失值填充为UNK
        processed = X_cat.fillna('UNK').astype(str)
        # 过滤低频类别（不在高频列表中的转为UNK）
        processed = processed.apply(lambda x: x if x in self.trained_categories[col] else 'UNK')
        # 用原生映射编码（避免依赖OrdinalEncoder）
        processed = processed.map(self.label_encoder_maps[col])
        return processed

    def save(self, filepath: str):
        """保存预处理器（仅序列化原生Python类型，协议版本2）"""
        if not self._is_fitted:
            raise NotFittedError("未拟合的预处理器无法保存")
        # 仅保存Java端需要的核心参数（原生类型）
        saved_data = {
            'train_median_values': self.train_median_values,  # {col: float}
            'trained_categories': self.trained_categories,  # {col: [str]}
            'label_encoders': self.label_encoder_maps,  # {col: {str: int}}
            'scaler_mean': self.scaler_mean,  # {col: float}
            'scaler_scale': self.scaler_scale,  # {col: float}
            'num_cols': self.num_cols,  # [str]
            'cat_cols': self.cat_cols  # [str]
        }
        # 关键：指定协议版本2，兼容Jython
        with open(filepath, 'wb') as f:
            pickle.dump(saved_data, f, protocol=2)
        logger.info(f"预处理器保存到: {filepath}（协议版本2，原生Python类型）")

    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """加载预处理器（仅加载原生Python类型）"""
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
        preprocessor = cls(
            num_cols=saved_data['num_cols'],
            cat_cols=saved_data['cat_cols'],
            min_category_freq=100  # 兼容旧逻辑，实际加载后不影响
        )
        # 恢复核心参数（均为原生Python类型）
        preprocessor.train_median_values = saved_data['train_median_values']
        preprocessor.trained_categories = saved_data['trained_categories']
        preprocessor.label_encoder_maps = saved_data['label_encoder_maps']
        preprocessor.scaler_mean = saved_data['scaler_mean']
        preprocessor.scaler_scale = saved_data['scaler_scale']
        preprocessor._is_fitted = True
        logger.info(f"预处理器从: {filepath} 加载完成（原生Python类型）")
        return preprocessor


# ---------------------- 3. Criteo 数据集类 ----------------------
class CriteoDataset(Dataset):
    """将预处理后的Criteo数据转为PyTorch张量"""

    def __init__(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        # 转换为numpy数组后再转张量（避免DataFrame直接转换的类型问题）
        self.X = torch.tensor(X.values.astype(np.float32), dtype=torch.float32)
        self.y = torch.tensor(y.values.astype(np.float32).reshape(-1, 1),
                              dtype=torch.float32) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx], None


# ---------------------- 4. PyTorch CTR 模型 ----------------------
class CTRNet(nn.Module):
    """CTR预测神经网络：捕捉特征非线性交互"""

    def __init__(self, input_dim: int = 39):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------- 5. PyTorch CTR 预测器 ----------------------
class PyTorchCTRPredictor:
    def __init__(self, num_cols: List[str], cat_cols: List[str], preprocessor: DataPreprocessor):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.preprocessor = preprocessor
        self.input_dim = len(num_cols) + len(cat_cols)  # 39维特征（13数值+26类别）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def build_model(self) -> nn.Module:
        model = CTRNet(input_dim=self.input_dim)
        model.to(self.device)
        logger.info(f"模型构建完成：输入维度={self.input_dim}，设备={self.device}")
        return model

    def train(
            self,
            X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series,
            batch_size: int = 256, epochs: int = 10, lr: float = 1e-3
    ):
        logger.info("开始训练PyTorch CTR模型...")

        # 构建DataLoader（兼容CPU/GPU）
        train_dataset = CriteoDataset(X_train, y_train)
        val_dataset = CriteoDataset(X_val, y_val)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        # 初始化模型、损失函数、优化器
        self.model = self.build_model()
        criterion = nn.BCELoss()  # 二分类交叉熵（CTR预测场景）
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)  # L2正则化

        # 训练循环
        for epoch in range(1, epochs + 1):
            # 训练阶段
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                y_pred = self.model(batch_X)
                loss = criterion(y_pred, batch_y)

                # 反向传播与优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)

            # 计算训练集平均损失
            train_avg_loss = train_loss / len(train_loader.dataset)

            # 验证阶段（禁用梯度计算，节省内存）
            self.model.eval()
            val_loss = 0.0
            y_val_true = []
            y_val_pred = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    y_pred = self.model(batch_X)
                    val_loss += criterion(y_pred, batch_y).item() * batch_X.size(0)

                    # 收集真实标签和预测概率（转原生list，避免NumPy依赖）
                    y_val_true.extend(batch_y.cpu().numpy().flatten().tolist())
                    y_val_pred.extend(y_pred.cpu().numpy().flatten().tolist())

            # 计算验证集指标
            val_avg_loss = val_loss / len(val_loader.dataset)
            val_auc = roc_auc_score(y_val_true, y_val_pred)  # AUC指标（CTR核心指标）
            val_acc = accuracy_score(y_val_true, np.round(y_val_pred).tolist())  # 准确率（参考）

            logger.info(
                f"Epoch {epoch:2d}/{epochs} | "
                f"Train Loss: {train_avg_loss:.4f} | "
                f"Val Loss: {val_avg_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        logger.info("模型训练完成")
        return self.model

    def save_onnx(self, onnx_path: str = "criteo_ctr_model.onnx"):
        """导出ONNX模型（确保Java端ONNX Runtime可加载）"""
        if self.model is None:
            raise NotFittedError("模型未训练，无法导出ONNX")

        self.model.eval()
        # 构造 dummy input（与Java端输入维度一致：[batch_size, 39]）
        dummy_input = torch.randn(1, self.input_dim, device=self.device, dtype=torch.float32)

        # 导出ONNX模型（关键参数与Java端对齐）
        torch.onnx.export(
            model=self.model,
            args=dummy_input,
            f=onnx_path,
            input_names=["criteo_features"],  # 输入节点名（Java端需对应）
            output_names=["ctr_prob"],  # 输出节点名（Java端需对应）
            dynamic_axes={  # 动态batch_size（支持任意批量）
                "criteo_features": {0: "batch_size"},
                "ctr_prob": {0: "batch_size"}
            },
            opset_version=12,  # 兼容主流ONNX Runtime版本（1.10+）
            do_constant_folding=True,  # 优化常量折叠，减小模型体积
            export_params=True  # 导出模型参数（含权重）
        )

        # 验证ONNX模型合法性（避免导出损坏模型）
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX模型导出完成，路径: {onnx_path}")

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """批量预测（用于测试）"""
        if self.model is None:
            raise NotFittedError("模型未训练，无法预测")

        self.model.eval()
        X_tensor = torch.tensor(X.values.astype(np.float32), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor)
        return y_pred.cpu().numpy().flatten()


# ---------------------- 6. 完整的Criteo数据加载函数 ----------------------
def load_criteo_data(filepath: str, cols: List[str], max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    加载Criteo数据集（TSV格式，无表头）
    参数：
        filepath: 数据文件路径
        cols: 列名列表（需与Criteo格式一致：label + I1~I13 + C1~C26）
        max_rows: 最大加载行数（控制内存使用，None表示全量加载）
    返回：
        加载后的DataFrame
    """
    logger.info(f"从 {filepath} 加载Criteo数据（最大行数: {max_rows or '全部'}）")
    try:
        # Criteo数据特点：TSV分隔、无表头、可能包含大量缺失值
        if max_rows:
            # 分块加载（避免大文件内存溢出）
            chunks = pd.read_csv(
                filepath,
                sep='\t',  # TSV格式，用制表符分隔
                header=None,  # 无表头
                names=cols,  # 指定列名
                chunksize=10000,  # 每块10000行
                low_memory=False  # 禁用低内存模式（避免类别特征类型推断错误）
            )

            data_chunks = []
            current_rows = 0

            for chunk in chunks:
                # 控制总加载行数不超过max_rows
                if current_rows + len(chunk) > max_rows:
                    remaining = max_rows - current_rows
                    if remaining > 0:
                        data_chunks.append(chunk.iloc[:remaining])
                    break
                data_chunks.append(chunk)
                current_rows += len(chunk)

            # 合并分块数据
            data = pd.concat(data_chunks, ignore_index=True)
        else:
            # 全量加载
            data = pd.read_csv(
                filepath,
                sep='\t',
                header=None,
                names=cols,
                low_memory=False
            )

        logger.info(f"数据加载完成：共 {len(data)} 行，{len(data.columns)} 列")
        return data

    except FileNotFoundError:
        logger.error(f"数据文件未找到：{filepath}")
        raise
    except Exception as e:
        logger.error(f"数据加载失败：{str(e)}")
        raise


# ---------------------- 7. 主函数（完整流程） ----------------------
def main():
    # 配置参数（可根据实际环境调整）
    config = {
        'data_path': '../../data/raw/train.txt',  # Criteo训练数据路径
        'test_path': '../../data/raw/test.txt',  # Criteo测试数据路径
        'preprocessor_path': 'criteo_preprocessor.pkl',  # 预处理器保存路径
        'onnx_model_path': 'criteo_ctr_model.onnx',  # ONNX模型保存路径
        'max_train_rows': 10000000,  # 训练数据最大行数（按需调整）
        'test_size': 0.2,  # 验证集比例
        'batch_size': 256,  # 训练批量大小
        'epochs': 10,  # 训练轮数
        'lr': 1e-3  # 学习率
    }

    # Criteo列名定义（1个label + 13个数值特征 + 26个类别特征）
    cols = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
    num_cols = [f'I{i}' for i in range(1, 14)]  # 数值特征列名
    cat_cols = [f'C{i}' for i in range(1, 27)]  # 类别特征列名

    try:
        # 1. 加载训练数据
        train_data = load_criteo_data(
            filepath=config['data_path'],
            cols=cols,
            max_rows=config['max_train_rows']
        )

        # 2. 初始化并拟合预处理器
        preprocessor = DataPreprocessor(
            num_cols=num_cols,
            cat_cols=cat_cols,
            min_category_freq=100  # 类别频率低于100的视为低频
        )
        preprocessor.fit(train_data[num_cols + cat_cols])  # 仅用特征列拟合
        preprocessor.save(config['preprocessor_path'])

        # 3. 预处理训练数据
        X = preprocessor.transform(train_data[num_cols + cat_cols])
        y = train_data['label']  # 标签列（点击=1，未点击=0）

        # 4. 拆分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=42,
            stratify=y  # 保持标签分布一致（CTR数据不平衡）
        )

        # 5. 初始化并训练模型
        predictor = PyTorchCTRPredictor(
            num_cols=num_cols,
            cat_cols=cat_cols,
            preprocessor=preprocessor
        )
        predictor.train(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            lr=config['lr']
        )

        # 6. 导出ONNX模型（供Java调用）
        predictor.save_onnx(config['onnx_model_path'])

        # 7. 用测试数据验证预测功能
        test_data = load_criteo_data(
            filepath=config['test_path'],
            cols=num_cols + cat_cols,  # 测试数据无label列
            max_rows=100  # 仅加载100条测试
        )
        test_data_processed = preprocessor.transform(test_data)
        test_preds = predictor.predict_batch(test_data_processed)

        logger.info("测试数据预测结果：")
        for i in range(0, len(test_preds)):
            logger.info(f"样本{i + 1} 点击概率: {test_preds[i]:.4f}")

    except Exception as e:
        logger.error(f"程序运行失败: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
