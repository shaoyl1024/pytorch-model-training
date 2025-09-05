import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.exceptions import NotFittedError
import onnx


# ---------------------- 1. 全局配置类 ----------------------
class Config:
    def __init__(self):
        # 路径配置
        self.data_path = "/Users/charles/Workspace/pytorch-prediction/pytorch-model-training/data/raw/train.txt"
        self.test_path = "/Users/charles/Workspace/pytorch-prediction/pytorch-model-training/data/raw/test.txt"
        self.preprocessor_path = "./criteo_preprocessor.json"  # 改为JSON格式
        self.onnx_model_path = "./criteo_ctr_model.onnx"

        # 数据处理配置
        self.max_train_rows = 100000  # 可根据内存调整
        self.test_sample_size = 100
        self.val_size = 0.2
        self.min_category_freq = 100
        self.chunk_size = 10000

        # 模型训练配置
        self.batch_size = 256
        self.epochs = 10
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.dropout_rate = 0.2
        self.hidden_dims = [64, 32]

        # 特征列名
        self.label_col = "label"
        self.num_cols = [f"I{i}" for i in range(1, 14)]
        self.cat_cols = [f"C{i}" for i in range(1, 27)]
        self.all_feature_cols = self.num_cols + self.cat_cols
        self.train_cols = [self.label_col] + self.all_feature_cols


GLOBAL_CONFIG = Config()

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ---------------------- 2. 数据加载工具 ----------------------
def load_criteo_data(
        filepath: str,
        cols: List[str],
        max_rows: Optional[int] = None
) -> pd.DataFrame:
    logger.info(f"开始加载数据：{filepath}（最大行数：{max_rows or '全量'}）")

    try:
        reader = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=cols,
            chunksize=GLOBAL_CONFIG.chunk_size,
            low_memory=False,
            na_values=["", "NA"],
            dtype={col: str for col in GLOBAL_CONFIG.cat_cols}
        )

        data_chunks = []
        total_rows = 0
        num_cols = GLOBAL_CONFIG.num_cols

        for chunk in reader:
            for col in num_cols:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

            if max_rows and total_rows + len(chunk) > max_rows:
                remaining = max_rows - total_rows
                data_chunks.append(chunk.iloc[:remaining])
                total_rows += remaining
                break

            data_chunks.append(chunk)
            total_rows += len(chunk)

        data = pd.concat(data_chunks, ignore_index=True)
        logger.info(f"数据加载完成：{len(data)} 行 × {len(data.columns)} 列")
        return data

    except FileNotFoundError:
        logger.error(f"数据文件不存在：{filepath}")
        raise
    except Exception as e:
        logger.error(f"数据加载失败：{str(e)}", exc_info=True)
        raise


# ---------------------- 3. 数据预处理器（JSON序列化） ----------------------
class CriteoPreprocessor:
    """使用JSON序列化预处理参数，完全移除pickle依赖，增强跨语言兼容性"""

    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.is_fitted = False

        # 预处理参数（仅使用JSON可序列化类型：dict/list/float/int/str）
        self.num_params: Dict[str, Dict[str, float]] = {
            col: {"median": 0.0, "mean": 0.0, "scale": 1.0}
            for col in self.config.num_cols
        }
        self.cat_params: Dict[str, Dict[str, Union[List[str], Dict[str, int]]]] = {
            col: {"high_freq": [], "code_map": {"UNK": -1}}  # 确保值都是JSON兼容类型
            for col in self.config.cat_cols
        }

    def fit(self, X: pd.DataFrame) -> "CriteoPreprocessor":
        if not self.is_fitted:
            logger.info("开始拟合预处理器...")
            num_cols = self.config.num_cols
            cat_cols = self.config.cat_cols
            min_freq = self.config.min_category_freq

            # 拟合数值特征参数
            for col in num_cols:
                # 计算中位数（缺失值填充用）
                median = float(X[col].median())  # 转为Python原生float
                # 填充缺失值后做log1p转换
                filled_vals = X[col].fillna(median)
                log1p_vals = np.log1p(filled_vals.clip(lower=-0.999))
                # 计算标准化参数（均值+标准差）
                mean = float(log1p_vals.mean())
                scale = float(log1p_vals.std())
                scale = scale if scale > 1e-9 else 1e-9  # 避免除零错误

                self.num_params[col] = {
                    "median": median,
                    "mean": mean,
                    "scale": scale
                }

            # 拟合类别特征参数
            for col in cat_cols:
                # 填充缺失值为"UNK"
                cat_series = X[col].fillna("UNK")
                # 统计类别频率，筛选高频类别
                freq = cat_series.value_counts()
                high_freq_cats = freq[freq >= min_freq].index.tolist()  # 原生list
                # 构建编码映射（确保键值都是JSON兼容类型）
                code_map = {cat: idx for idx, cat in enumerate(high_freq_cats)}  # str→int
                code_map["UNK"] = -1  # 固定UNK编码

                self.cat_params[col] = {
                    "high_freq": high_freq_cats,
                    "code_map": code_map
                }

            self.is_fitted = True
            logger.info("预处理器拟合完成")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise NotFittedError("预处理器未拟合，请先调用fit()方法")

        X_processed = X.copy()
        num_cols = self.config.num_cols
        cat_cols = self.config.cat_cols

        # 处理数值特征
        for col in num_cols:
            params = self.num_params[col]
            X_processed[col] = X_processed[col].fillna(params["median"])
            X_processed[col] = np.log1p(X_processed[col].clip(lower=-0.999))
            X_processed[col] = (X_processed[col] - params["mean"]) / params["scale"]

            test = '1'

        # 处理类别特征
        for col in cat_cols:
            params = self.cat_params[col]
            high_freq_set = set(params["high_freq"])
            code_map = params["code_map"]

            X_processed[col] = X_processed[col].fillna("UNK")
            X_processed[col] = X_processed[col].apply(
                lambda x: x if x in high_freq_set else "UNK"
            )
            X_processed[col] = X_processed[col].map(lambda x: code_map.get(x, -1))

        return X_processed[self.config.all_feature_cols]

    def save(self, filepath: Optional[str] = None) -> None:
        """保存预处理参数到JSON文件（完全兼容Java等其他语言）"""
        if not self.is_fitted:
            raise NotFittedError("未拟合的预处理器无法保存")

        filepath = filepath or self.config.preprocessor_path
        # 准备JSON可序列化的数据（仅包含原生Python类型）
        saved_data = {
            "num_params": self.num_params,
            "cat_params": self.cat_params,
            "config": {
                "num_cols": self.config.num_cols,
                "cat_cols": self.config.cat_cols
            }
        }

        # 保存为JSON（缩进2空格，确保可读性）
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(saved_data, f, ensure_ascii=False, indent=2)

        logger.info(f"预处理器参数已保存至JSON：{filepath}")

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "CriteoPreprocessor":
        """从JSON文件加载预处理参数"""
        filepath = filepath or GLOBAL_CONFIG.preprocessor_path
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                saved_data = json.load(f)  # 加载JSON数据

            preprocessor = cls()
            # 恢复参数（均为JSON兼容类型）
            preprocessor.num_params = saved_data["num_params"]
            preprocessor.cat_params = saved_data["cat_params"]
            preprocessor.is_fitted = True

            # 验证特征列配置一致性
            config = saved_data["config"]
            assert config["num_cols"] == preprocessor.config.num_cols, "数值列配置不匹配"
            assert config["cat_cols"] == preprocessor.config.cat_cols, "类别列配置不匹配"

            logger.info(f"预处理器已从JSON加载：{filepath}")
            return preprocessor

        except FileNotFoundError:
            logger.error(f"预处理器JSON文件不存在：{filepath}")
            raise
        except Exception as e:
            logger.error(f"预处理器加载失败：{str(e)}", exc_info=True)
            raise


# ---------------------- 4. 数据集与DataLoader ----------------------
class CTRDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx], None


def create_dataloaders(
        X_train: pd.DataFrame, y_train: pd.Series,
        X_val: pd.DataFrame, y_val: pd.Series
) -> Tuple[DataLoader, DataLoader]:
    config = GLOBAL_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CTRDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=False
    )

    val_dataset = CTRDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=1 if device.type == "cuda" else 0,
        pin_memory=True if device.type == "cuda" else False
    )

    logger.info(f"DataLoader创建完成：训练批次={len(train_loader)}，验证批次={len(val_loader)}")
    return train_loader, val_loader


# ---------------------- 5. CTR模型与训练器 ----------------------
class CTRNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        config = GLOBAL_CONFIG
        hidden_dims = config.hidden_dims
        dropout_rate = config.dropout_rate

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CTRTrainer:
    def __init__(self, preprocessor: CriteoPreprocessor):
        self.config = GLOBAL_CONFIG
        self.preprocessor = preprocessor
        self.input_dim = len(self.config.all_feature_cols)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[CTRNet] = None
        self.best_val_auc = 0.0

    def build_model(self) -> CTRNet:
        model = CTRNet(input_dim=self.input_dim)
        model.to(self.device)
        logger.info(f"模型构建完成：输入维度={self.input_dim}，设备={self.device}")
        return model

    def train_epoch(self, model: CTRNet, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)

        return total_loss / len(loader.dataset)

    def validate(self, model: CTRNet, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float, float]:
        model.eval()
        total_loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                total_loss += loss.item() * batch_X.size(0)

                # 收集预测结果（转为原生Python类型）
                y_true.extend(batch_y.cpu().numpy().flatten().tolist())
                y_pred.extend(pred.cpu().numpy().flatten().tolist())

        avg_loss = total_loss / len(loader.dataset)
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, np.round(y_pred).tolist())

        return avg_loss, auc, acc

    def train(self) -> CTRNet:
        # 加载并预处理训练数据
        train_data = load_criteo_data(
            filepath=self.config.data_path,
            cols=self.config.train_cols,
            max_rows=self.config.max_train_rows
        )

        # 拟合预处理器
        self.preprocessor.fit(train_data[self.config.all_feature_cols])
        self.preprocessor.save()  # 保存为JSON

        # 预处理特征并拆分数据集
        X = self.preprocessor.transform(train_data[self.config.all_feature_cols])
        y = train_data[self.config.label_col]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.val_size,
            random_state=42,
            stratify=y
        )

        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(X_train, y_train, X_val, y_val)

        # 初始化模型、损失函数和优化器
        self.model = self.build_model()
        criterion = nn.BCELoss()  # 二分类交叉熵损失
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        # 训练循环
        logger.info("开始模型训练...")
        for epoch in range(1, self.config.epochs + 1):
            # 训练
            train_loss = self.train_epoch(self.model, train_loader, criterion, optimizer)

            # 验证
            val_loss, val_auc, val_acc = self.validate(self.model, val_loader, criterion)

            # 记录最佳模型
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                logger.info(f"更新最佳模型（AUC: {self.best_val_auc:.4f}）")

            # 打印 epoch 日志
            logger.info(
                f"Epoch {epoch:2d}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        logger.info(f"训练完成，最佳验证AUC: {self.best_val_auc:.4f}")
        return self.model

    def export_onnx(self, filepath: Optional[str] = None) -> None:
        if not self.model:
            raise NotFittedError("模型未训练，无法导出")

        filepath = filepath or self.config.onnx_model_path
        self.model.eval()

        # 创建虚拟输入（匹配模型输入维度）
        dummy_input = torch.randn(1, self.input_dim, device=self.device, dtype=torch.float32)

        # 导出ONNX模型
        torch.onnx.export(
            model=self.model,
            args=dummy_input,
            f=filepath,
            input_names=["criteo_features"],
            output_names=["ctr_prob"],
            dynamic_axes={
                "criteo_features": {0: "batch_size"},
                "ctr_prob": {0: "batch_size"}
            },
            opset_version=12,
            do_constant_folding=True,
            export_params=True
        )

        # 验证模型合法性
        onnx_model = onnx.load(filepath)
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX模型已导出至：{filepath}")

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        if not self.model:
            raise NotFittedError("模型未训练，无法预测")

        self.model.eval()
        # 预处理原始数据
        X_processed = self.preprocessor.transform(X_raw)
        # 转换为张量并预测
        X_tensor = torch.tensor(X_processed.values, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = self.model(X_tensor)

        return preds.cpu().numpy().flatten()


# ---------------------- 6. 主函数（完整流程入口） ----------------------
def main():
    try:
        # 1. 初始化预处理器（JSON序列化）
        preprocessor = CriteoPreprocessor()

        # 2. 初始化训练器并训练模型
        trainer = CTRTrainer(preprocessor)
        trainer.train()

        # 3. 导出ONNX模型（供Java部署）
        trainer.export_onnx()

        # 4. 用测试数据验证预测功能
        test_data = load_criteo_data(
            filepath=GLOBAL_CONFIG.test_path,
            cols=GLOBAL_CONFIG.all_feature_cols,
            max_rows=GLOBAL_CONFIG.test_sample_size
        )

        # 5. 批量预测并输出结果
        predictions = trainer.predict(test_data)
        logger.info("\n测试数据预测结果（前20条）：")
        for i in range(min(20, len(predictions))):
            logger.info(f"样本 {i + 1:2d} 点击概率: {predictions[i]:.4f}")

        logger.info("所有流程执行完成")

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
