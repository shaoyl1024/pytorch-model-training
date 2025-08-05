import pickle
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn2pmml import PMMLPipeline, sklearn2pmml, make_pmml_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 日志级别（INFO及以上会被记录）
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式（时间、级别、内容）
)
logger = logging.getLogger(__name__)  # 创建日志实例（后续用logger输出信息）


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """自定义数据预处理器，处理数值和类别特征"""

    def __init__(self, num_cols: List[str], cat_cols: List[str], min_category_freq: int = 100):
        """
        初始化预处理器

        参数:
            num_cols: 数值特征列名列表
            cat_cols: 类别特征列名列表
            min_category_freq: 类别最小出现频率(低于此频率的类别将被归为'UNK')
        """
        self.num_cols = num_cols  # 数值特征列名
        self.cat_cols = cat_cols  # 类别特征列名
        self.min_category_freq = min_category_freq  # 类别最小出现频率（过滤低频类别）
        self.train_median_values = {}  # 存储训练集数值特征的中位数（用于填充缺失值）
        self.trained_categories = {}  # 存储训练集高频类别（key:列名，value:高频类别集合）
        self.label_encoders = {}  # 存储每个类别特征的编码器（OrdinalEncoder）
        self.scaler = StandardScaler()  # 数值特征标准化器（均值0、方差1）
        self._is_fitted = False  # 标记是否已拟合（防止未拟合时调用transform）

    def fit(self, X: pd.DataFrame, y=None):
        """拟合预处理参数"""

        # 处理数值特征：计算中位数（用于后续填充缺失值）
        for col in self.num_cols:
            X[col] = pd.to_numeric(X[col], errors='coerce')  # 强制转为数值型（无效值转为NaN）
            self.train_median_values[col] = X[col].median()  # 存储中位数

        # 处理类别特征：筛选高频类别+拟合编码器
        for col in self.cat_cols:
            X[col] = X[col].fillna('UNK').astype(str)  # 缺失值填充为'UNK'，转为字符串
            freq = X[col].value_counts()  # 统计每个类别的出现频率
            high_freq = freq[freq > self.min_category_freq].index  # 筛选高频类别（频率>阈值）
            self.trained_categories[col] = set(high_freq)  # 存储高频类别

            # 初始化并拟合标签编码器（将类别转为整数）
            le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)  # 未知类别编码为-1
            le.fit(X[col].values.reshape(-1, 1))  # 拟合（输入需为二维数组）
            self.label_encoders[col] = le  # 存储编码器

        # 拟合标准化器
        numeric_data = self._prepare_numeric(X[self.num_cols])
        self.scaler.fit(numeric_data)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用预处理转换"""
        if not self._is_fitted:
            raise NotFittedError("预处理器尚未拟合，请先调用fit()方法")

        # 复制数据避免修改原始数据
        X_processed = X.copy()

        # 处理数值特征（填充缺失值+log转换+标准化）
        X_processed[self.num_cols] = self._process_numeric(X[self.num_cols])

        # 处理类别特征（填充缺失值+过滤低频类别+编码）
        for col in self.cat_cols:
            X_processed[col] = self._process_categorical(X[col], col)

        return X_processed

    def _prepare_numeric(self, X_num: pd.DataFrame) -> pd.DataFrame:
        """准备数值数据用于标准化（填充缺失值+log转换）"""
        processed = X_num.copy()
        for col in self.num_cols:
            processed[col] = pd.to_numeric(processed[col], errors='coerce')  # 转为数值型
            processed[col] = processed[col].fillna(self.train_median_values[col])  # 用训练集中位数填充缺失值
            processed[col] = np.log1p(processed[col].clip(lower=-0.999))  # log1p转换（处理长尾分布），避免log(0)
        return processed

    def _process_numeric(self, X_num: pd.DataFrame) -> pd.DataFrame:
        """处理数值特征（调用_prepare_numeric后标准化）"""
        prepared = self._prepare_numeric(X_num)
        return pd.DataFrame(
            self.scaler.transform(prepared),  # 用训练好的标准化器转换
            columns=self.num_cols,
            index=prepared.index  # 保留原始索引
        )

    def _process_categorical(self, X_cat: pd.Series, col: str) -> pd.Series:
        """处理单个类别特征（填充缺失值+过滤低频类别+编码）"""
        processed = X_cat.fillna('UNK').astype(str)  # 缺失值填充为'UNK'
        # 低频类别（不在训练集高频类别中）转为'UNK'
        processed = processed.where(processed.isin(self.trained_categories[col]), 'UNK')

        # 用训练好的编码器转换为整数
        return pd.Series(
            self.label_encoders[col].transform(processed.values.reshape(-1, 1)).flatten())

    def save(self, filepath: str):
        """保存预处理对象（序列化）"""
        if not self._is_fitted:
            raise NotFittedError("预处理器尚未拟合，无法保存")

        with open(filepath, 'wb') as f:
            pickle.dump({  # 保存关键参数
                'train_median_values': self.train_median_values,
                'trained_categories': self.trained_categories,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'num_cols': self.num_cols,
                'cat_cols': self.cat_cols,
                'min_category_freq': self.min_category_freq
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'DataPreprocessor':
        """加载预处理对象（反序列化）"""
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)

        # 重建预处理器实例并恢复参数
        preprocessor = cls(
            saved_data['num_cols'],
            saved_data['cat_cols'],
            saved_data['min_category_freq']
        )

        preprocessor.train_median_values = saved_data['train_median_values']
        preprocessor.trained_categories = saved_data['trained_categories']
        preprocessor.label_encoders = saved_data['label_encoders']
        preprocessor.scaler = saved_data['scaler']
        preprocessor._is_fitted = True  # 标记为已拟合

        return preprocessor


class CTRPredictor:
    """CTR预测模型"""

    def __init__(self, num_cols: List[str], cat_cols: List[str], preprocessor: Optional[DataPreprocessor] = None):
        """
        初始化预测器

        参数:
            num_cols: 数值特征列名列表
            cat_cols: 类别特征列名列表
            preprocessor: 预处理器实例
        """
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.preprocessor = preprocessor  # 关联预处理器
        self.pipeline = None  # 存储PMML管道（模型+预处理）

    def build_pipeline(self) -> PMMLPipeline:
        """构建PMML管道（整合预处理和模型）"""
        if self.preprocessor is None:
            raise ValueError("预处理器未提供")

        if not hasattr(self.preprocessor, '_is_fitted') or not self.preprocessor._is_fitted:
            raise NotFittedError("预处理器尚未拟合")

        # 构建列转换器（分别处理数值和类别特征）
        preprocessor = ColumnTransformer(
            transformers=[
                # 数值特征：用预处理器中的标准化器
                ('num', self.preprocessor.scaler, self.num_cols),
                # 类别特征：用序数编码器（与预处理器逻辑一致）
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), self.cat_cols)
            ],
            remainder='drop'  # 丢弃未指定的列
        )

        # 构建PMML管道（预处理 -> 逻辑回归）
        return PMMLPipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=0.1,  # 正则化强度（较小值表示强正则化）
                max_iter=5000,  # 迭代次数（确保收敛）
                solver='lbfgs',  # 优化器
                class_weight='balanced',  # 类别不平衡时自动调整权重
                random_state=42  # 随机种子（保证结果可复现）
            ))
        ])

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """训练模型并评估性能"""
        logger.info("开始训练模型...")

        # 拆分训练集和测试集（按比例）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 构建并训练管道
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train, y_train)  # 拟合整个管道（预处理+模型）

        # 评估模型
        train_score = accuracy_score(y_train, self.pipeline.predict(X_train))  # 训练集准确率
        test_score = accuracy_score(y_test, self.pipeline.predict(X_test))  # 测试集准确率

        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]  # 预测正例（点击）的概率
        auc_score = roc_auc_score(y_test, y_pred_proba)  # AUC（评估排序能力，CTR预测核心指标）

        logger.info(f"训练完成 - 训练集准确率: {train_score:.4f}, 测试集准确率: {test_score:.4f}, AUC: {auc_score:.4f}")

        return self.pipeline

    def save_model(self, pmml_path: str):
        """保存模型为PMML格式（便于部署）"""
        if self.pipeline is None:
            raise NotFittedError("模型未训练，请先调用train()方法")

        logger.info(f"保存PMML模型到 {pmml_path}")
        pmml_pipeline = make_pmml_pipeline(self.pipeline, active_fields=self.num_cols + self.cat_cols)
        sklearn2pmml(pmml_pipeline, pmml_path, with_repr=True)  # 导出为PMML文件

    def predict_single(self, sample: Dict) -> Dict:
        """单条样本预测（返回点击概率）"""
        if self.pipeline is None:
            raise NotFittedError("模型未训练，请先调用train()方法")

        sample_df = pd.DataFrame([sample])  # 转换为DataFrame（适配管道输入）
        proba = self.pipeline.predict_proba(sample_df)[0][1]  # 取正例概率
        return {'probability(1)': proba}

    def predict_batch(self, data: pd.DataFrame) -> np.ndarray:
        """批量预测（返回点击概率数组）"""
        if self.pipeline is None:
            raise NotFittedError("模型未训练，请先调用train()方法")

        return self.pipeline.predict_proba(data)[:, 1]


def load_data(filepath: str, cols: List[str], max_rows: Optional[int] = None) -> pd.DataFrame:
    """加载数据（支持分块加载和限制最大行数）"""
    logger.info(f"从 {filepath} 加载数据...")

    try:
        if max_rows:  # 限制最大行数（避免内存溢出）
            chunks = pd.read_csv(filepath, sep='\t', header=None, names=cols, chunksize=10000)  # 分块读取
            data_chunks = []
            current_rows = 0

            for chunk in chunks:
                if current_rows + len(chunk) > max_rows:  # 超过最大行数时截断
                    remaining_rows = max_rows - current_rows
                    if remaining_rows > 0:
                        data_chunks.append(chunk.iloc[:remaining_rows])
                    break
                else:
                    data_chunks.append(chunk)
                    current_rows += len(chunk)

            data = pd.concat(data_chunks)  # 合并分块
        else:
            data = pd.read_csv(filepath, sep='\t', header=None, names=cols)

        logger.info(f"成功加载 {len(data)} 行数据")
        return data
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        raise


def main():
    # 配置参数
    config = {
        'data_path': '../../data/raw/train.txt',  # 训练数据路径
        'test_path': '../../data/raw/test.txt',  # 测试数据路径
        'preprocessor_path': 'preprocess_objects.pkl',  # 预处理器保存路径
        'pmml_path': 'criteo_ctr_lr.pmml',  # 模型保存路径
        'max_rows': 100000,  # None表示不限制
        'test_size': 0.2  # 测试集比例
    }

    # 定义列名（参考Criteo数据集格式：1个label+13个数值特征+26个类别特征）
    cols = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
    num_cols = [f'I{i}' for i in range(1, 14)]  # 数值特征列名
    cat_cols = [f'C{i}' for i in range(1, 27)]  # 类别特征列名

    try:
        # 1. 加载数据
        data = load_data(config['data_path'], cols, config['max_rows'])

        # 2. 初始化并拟合预处理器
        preprocessor = DataPreprocessor(num_cols, cat_cols)
        preprocessor.fit(data)  # 用训练数据拟合
        preprocessor.save(config['preprocessor_path'])  # 保存预处理器

        # 3. 预处理数据
        X = preprocessor.transform(data[num_cols + cat_cols])  # 特征
        y = data['label']  # 标签（点击=1，未点击=0）

        # 4. 训练模型 (传入预处理器)
        predictor = CTRPredictor(num_cols, cat_cols, preprocessor)
        pipeline = predictor.train(X, y, config['test_size'])

        # 5. 保存模型
        predictor.save_model(config['pmml_path'])

        # 6. 测试预测功能
        test_data = load_data(config['test_path'], cols[1:], 10)  # 加载10条测试数据（不含label）
        test_data_processed = preprocessor.transform(test_data)  # 预处理测试数据

        # 单条样本预测
        sample = test_data_processed.iloc[0].to_dict()
        result = predictor.predict_single(sample)
        logger.info(f"单条样本预测结果: {result}")

        # 批量预测
        batch_results = predictor.predict_batch(test_data_processed)
        logger.info("批量预测结果(前5条):")
        for i, prob in enumerate(batch_results[:5]):
            logger.info(f"样本{i + 1}: {prob:.4f}")

        logger.info(f"预测统计 - 平均: {np.mean(batch_results):.4f}, "
                    f"最大值: {np.max(batch_results):.4f}, "
                    f"最小值: {np.min(batch_results):.4f}")

    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
