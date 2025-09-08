import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


# ---------------------- 1. 全局配置类 ----------------------
class Config:
    def __init__(self):
        # 路径配置
        self.data_path = "/Users/charles/Workspace/pytorch-prediction/pytorch-model-training/data/raw/train.txt"
        self.test_path = "/Users/charles/Workspace/pytorch-prediction/pytorch-model-training/data/raw/test.txt"
        self.preprocessor_path = "./criteo_preprocessor.json"
        self.onnx_model_path = "./criteo_ctr_model.onnx"

        # 数据处理配置
        self.max_train_rows = 100000  # 可根据内存调整
        self.test_sample_size = 100
        self.val_size = 0.2
        self.min_category_freq = 100
        self.chunk_size = 10000

        # DeepFM模型训练配置
        self.batch_size = 256
        self.epochs = 10
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.dropout_rate = 0.2
        self.hidden_dims = [128, 64, 32]  # DeepFM的DNN部分维度

        # DeepFM特有的嵌入层配置
        self.embed_dim = 16  # FM部分的特征嵌入维度

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

        # 预处理参数（仅使用JSON可序列化类型）
        self.num_params: Dict[str, Dict[str, float]] = {
            col: {"median": 0.0, "mean": 0.0, "scale": 1.0}
            for col in self.config.num_cols
        }
        self.cat_params: Dict[str, Dict[str, Union[List[str], Dict[str, int]]]] = {
            col: {"high_freq": [], "code_map": {"UNK": -1}}
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
                median = float(X[col].median())
                filled_vals = X[col].fillna(median)
                log1p_vals = np.log1p(filled_vals.clip(lower=-0.999))
                mean = float(log1p_vals.mean())
                scale = float(log1p_vals.std())
                scale = scale if scale > 1e-9 else 1e-9

                self.num_params[col] = {
                    "median": median,
                    "mean": mean,
                    "scale": scale
                }

            # 拟合类别特征参数
            for col in cat_cols:
                cat_series = X[col].fillna("UNK")
                freq = cat_series.value_counts()
                high_freq_cats = freq[freq >= min_freq].index.tolist()
                code_map = {cat: idx for idx, cat in enumerate(high_freq_cats)}
                code_map["UNK"] = -1

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

        # 处理类别特征
        for col in cat_cols:
            params = self.cat_params[col]
            high_freq_set = set(params["high_freq"])
            code_map = params["code_map"]

            X_processed[col] = X_processed[col].fillna("UNK")
            X_processed[col] = X_processed[col].apply(
                lambda x: x if x in high_freq_set else "UNK"
            )
            X_processed[col] = X_processed[col].map(
                lambda x: code_map.get(x, -1)
            ).astype(float)

        return X_processed[self.config.all_feature_cols]

    def save(self, filepath: Optional[str] = None) -> None:
        if not self.is_fitted:
            raise NotFittedError("未拟合的预处理器无法保存")

        filepath = filepath or self.config.preprocessor_path
        saved_data = {
            "num_params": self.num_params,
            "cat_params": self.cat_params,
            "config": {
                "num_cols": self.config.num_cols,
                "cat_cols": self.config.cat_cols
            }
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(saved_data, f, ensure_ascii=False, indent=2)

        logger.info(f"预处理器参数已保存至JSON：{filepath}")

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "CriteoPreprocessor":
        filepath = filepath or GLOBAL_CONFIG.preprocessor_path
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                saved_data = json.load(f)

            preprocessor = cls()
            preprocessor.num_params = saved_data["num_params"]
            preprocessor.cat_params = saved_data["cat_params"]
            preprocessor.is_fitted = True

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


# ---------------------- 5. DeepFM模型定义与训练器 ----------------------
class DeepFM(nn.Module):
    """DeepFM模型：结合FM和DNN的点击率预测模型
    包含三个部分：
    1. 线性部分：学习一阶特征影响
    2. FM部分：学习二阶特征交互
    3. DNN部分：学习高阶特征交互
    """

    def __init__(self, input_dim: int):
        super().__init__()
        config = GLOBAL_CONFIG
        self.input_dim = input_dim  # 输入特征维度
        self.embed_dim = config.embed_dim  # 嵌入维度

        # 1. 线性部分 (一阶特征)
        self.linear = nn.Linear(input_dim, 1)

        # 2. FM部分 (二阶特征交互)
        # 特征嵌入层：将每个特征映射到低维空间
        self.fm_embedding = nn.Linear(input_dim, input_dim * self.embed_dim)

        # 3. DNN部分 (高阶特征交互)
        dnn_layers = []
        # DNN输入维度 = 特征数 × 嵌入维度
        prev_dim = input_dim * self.embed_dim

        for dim in config.hidden_dims:
            dnn_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),  # 增加批归一化稳定训练
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = dim

        self.dnn = nn.Sequential(*dnn_layers)

        # 4. 输出层：结合FM和DNN的输出
        self.output = nn.Linear(prev_dim + 1, 1)  # +1是FM的二阶部分输出
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：同时计算三个部分并融合结果"""
        # 1. 线性部分输出 (batch_size, 1)
        linear_out = self.linear(x)

        # 2. FM部分计算
        # 嵌入层输出 (batch_size, input_dim * embed_dim) -> 重塑为 (batch_size, input_dim, embed_dim)
        embeds = self.fm_embedding(x).view(-1, self.input_dim, self.embed_dim)

        # FM二阶部分: 0.5 * sum[(sum_v_i)^2 - sum(v_i^2)]
        sum_square = torch.sum(embeds, dim=1) ** 2  # (batch_size, embed_dim)
        square_sum = torch.sum(embeds ** 2, dim=1)  # (batch_size, embed_dim)
        fm_out = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # (batch_size, 1)

        # 3. DNN部分输出
        dnn_out = self.dnn(embeds.view(-1, self.input_dim * self.embed_dim))  # (batch_size, last_hidden_dim)

        # 4. 融合所有部分输出
        combined = torch.cat([fm_out, dnn_out], dim=1)  # (batch_size, last_hidden_dim + 1)
        output = self.output(combined)  # (batch_size, 1)

        return self.sigmoid(output)  # 输出点击率概率


class CTRTrainer:
    def __init__(self, preprocessor: CriteoPreprocessor):
        self.config = GLOBAL_CONFIG
        self.preprocessor = preprocessor
        self.input_dim = len(self.config.all_feature_cols)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[DeepFM] = None
        self.best_val_auc = 0.0

    def build_model(self) -> DeepFM:
        """构建DeepFM模型并移动到计算设备"""
        model = DeepFM(input_dim=self.input_dim)
        model.to(self.device)
        logger.info(
            f"DeepFM模型构建完成：输入维度={self.input_dim}, "
            f"嵌入维度={self.config.embed_dim}, "
            f"设备={self.device}"
        )
        return model

    def train_epoch(self, model: DeepFM, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """单轮训练"""
        model.train()
        total_loss = 0.0

        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            # 前向传播
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            # 反向传播与参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_X.size(0)

        return total_loss / len(loader.dataset)

    def validate(self, model: DeepFM, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float, float]:
        """模型验证"""
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

                y_true.extend(batch_y.cpu().numpy().flatten().tolist())
                y_pred.extend(pred.cpu().numpy().flatten().tolist())

        avg_loss = total_loss / len(loader.dataset)
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, np.round(y_pred).tolist())

        return avg_loss, auc, acc

    def train(self) -> DeepFM:
        # 加载并预处理训练数据
        train_data = load_criteo_data(
            filepath=self.config.data_path,
            cols=self.config.train_cols,
            max_rows=self.config.max_train_rows
        )

        # 拟合预处理器
        self.preprocessor.fit(train_data[self.config.all_feature_cols])
        self.preprocessor.save()

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
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )

        # 训练循环
        logger.info("开始DeepFM模型训练...")
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
        """导出DeepFM模型为ONNX格式"""
        if not self.model:
            raise NotFittedError("模型未训练，无法导出")

        filepath = filepath or self.config.onnx_model_path
        self.model.eval()

        # 创建虚拟输入
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
        logger.info(f"DeepFM ONNX模型已导出至：{filepath}")

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        """模型推理预测"""
        if not self.model:
            raise NotFittedError("模型未训练，无法预测")

        self.model.eval()
        X_processed = self.preprocessor.transform(X_raw)
        X_tensor = torch.tensor(X_processed.values, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            preds = self.model(X_tensor)

        return preds.cpu().numpy().flatten()


# ---------------------- 6. 主函数（完整流程入口） ----------------------
def main():
    try:
        # 1. 初始化预处理器
        preprocessor = CriteoPreprocessor()

        # 2. 初始化训练器并训练DeepFM模型
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
            logger.info(f"样本 {i + 1:2d} 点击概率: {predictions[i]}")

        logger.info("所有流程执行完成")

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
