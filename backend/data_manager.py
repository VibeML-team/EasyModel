"""
数据管理模块 - 数据上传、预处理、特征工程

支持：
- CSV/Excel 数据上传
- 自动列类型推断
- 缺失值处理
- 特征编码（类别、数值、文本）
- 数据验证
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class ColumnType(str, Enum):
    """列数据类型"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TARGET = "target"
    ID = "id"


@dataclass
class ColumnInfo:
    """列元信息"""
    name: str
    column_type: ColumnType
    dtype: str = ""
    missing_count: int = 0
    unique_count: int = 0
    sample_values: list[Any] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "column_type": self.column_type.value,
            "dtype": self.dtype,
            "missing_count": self.missing_count,
            "unique_count": self.unique_count,
            "sample_values": self.sample_values[:5],
            "statistics": self.statistics,
        }


@dataclass
class DataSpec:
    """数据规范"""
    dataset_id: str
    filename: str
    n_rows: int
    n_cols: int
    columns: list[ColumnInfo] = field(default_factory=list)
    target_column: str | None = None
    id_column: str | None = None
    feature_columns: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "target_column": self.target_column,
            "id_column": self.id_column,
            "feature_columns": self.feature_columns,
            "columns": [
                {
                    "name": c.name,
                    "column_type": c.column_type.value,
                    "dtype": c.dtype,
                    "missing_count": c.missing_count,
                    "unique_count": c.unique_count,
                    "sample_values": c.sample_values[:5],
                    "statistics": c.statistics,
                }
                for c in self.columns
            ],
        }


class DataTypeDetector:
    """自动数据类型检测器"""
    
    @staticmethod
    def detect_column_type(series: pd.Series) -> ColumnType:
        """检测单列数据类型"""
        # 检查是否为ID列（唯一值很多，且不是数值型）
        if series.nunique() == len(series) and series.dtype == object:
            # 检查是否是UUID或长ID
            sample = str(series.dropna().iloc[0]) if len(series.dropna()) > 0 else ""
            if len(sample) > 8:
                return ColumnType.ID
        
        # 检查是否为布尔值
        if series.dtype == bool or set(series.dropna().unique()).issubset({True, False, 0, 1, "yes", "no", "true", "false"}):
            return ColumnType.BOOLEAN
        
        # 检查是否为日期时间
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME
        
        # 尝试解析日期
        if series.dtype == object:
            try:
                pd.to_datetime(series.dropna().iloc[:100], errors='raise')
                return ColumnType.DATETIME
            except:
                pass
        
        # 检查是否为数值型
        if pd.api.types.is_numeric_dtype(series):
            # 检查是否是类别型编码（如0,1,2）
            unique_vals = series.nunique()
            if unique_vals <= 10 and unique_vals / len(series) < 0.05:
                return ColumnType.CATEGORICAL
            return ColumnType.NUMERIC
        
        # 检查类别型
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.05 or series.nunique() < 20:
            return ColumnType.CATEGORICAL
        
        # 默认为文本
        return ColumnType.TEXT
    
    @staticmethod
    def infer_target_column(df: pd.DataFrame, hint: str | None = None) -> str | None:
        """推断目标列"""
        if hint and hint in df.columns:
            return hint
        
        # 常见目标列名
        target_candidates = [
            "target", "label", "y", "class", "category", "outcome", "result",
            "churn", "purchase", "click", "fraud", "default", "conversion",
            "目标", "标签", "结果", "类别"
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            if any(candidate in col_lower for candidate in target_candidates):
                return col
        
        # 默认最后一列
        return df.columns[-1] if len(df.columns) > 0 else None
    
    @staticmethod
    def analyze_column(series: pd.Series) -> ColumnInfo:
        """分析单列统计信息"""
        col_type = DataTypeDetector.detect_column_type(series)
        
        stats = {}
        if col_type == ColumnType.NUMERIC:
            stats = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
            }
        elif col_type in [ColumnType.CATEGORICAL, ColumnType.BOOLEAN]:
            stats = {
                "top_categories": series.value_counts().head(5).to_dict(),
            }
        elif col_type == ColumnType.TEXT:
            text_lengths = series.dropna().astype(str).str.len()
            stats = {
                "avg_length": float(text_lengths.mean()),
                "max_length": int(text_lengths.max()),
            }
        
        return ColumnInfo(
            name=series.name,
            column_type=col_type,
            dtype=str(series.dtype),
            missing_count=int(series.isnull().sum()),
            unique_count=int(series.nunique()),
            sample_values=series.dropna().head(5).tolist(),
            statistics=stats,
        )


class FeatureEngineer:
    """特征工程 - 构建预处理管道"""
    
    def __init__(self):
        self.preprocessor: ColumnTransformer | None = None
        self.target_encoder: LabelEncoder | None = None
        self.fitted = False
    
    def build_preprocessor(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        column_types: dict[str, ColumnType],
    ) -> ColumnTransformer:
        """构建 sklearn 预处理管道"""
        
        # 按类型分组列
        numeric_cols = [c for c in feature_columns if column_types.get(c) == ColumnType.NUMERIC]
        categorical_cols = [c for c in feature_columns if column_types.get(c) == ColumnType.CATEGORICAL]
        
        transformers = []
        
        # 数值特征处理
        if numeric_cols:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ])
            transformers.append(('num', numeric_pipeline, numeric_cols))
        
        # 类别特征处理
        if categorical_cols:
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ])
            transformers.append(('cat', categorical_pipeline, categorical_cols))
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # 丢弃未指定的列
        )
        
        return self.preprocessor
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        column_types: dict[str, ColumnType],
    ) -> tuple[np.ndarray, np.ndarray]:
        """拟合并转换数据"""
        
        # 分离特征和目标
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # 构建并拟合预处理器
        preprocessor = self.build_preprocessor(df, feature_columns, column_types)
        X_processed = preprocessor.fit_transform(X)
        
        # 处理目标变量
        if y.dtype == object or y.dtype.name == 'category':
            self.target_encoder = LabelEncoder()
            y_processed = self.target_encoder.fit_transform(y.astype(str))
        else:
            y_processed = y.values
        
        self.fitted = True
        return X_processed, y_processed
    
    def transform(self, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        """转换新数据"""
        if not self.fitted or self.preprocessor is None:
            raise ValueError("Preprocessor not fitted yet")
        
        X = df[feature_columns].copy()
        return self.preprocessor.transform(X)
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """反编码目标"""
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(y.astype(int))
        return y
    
    def get_feature_names(self) -> list[str]:
        """获取处理后特征名"""
        if self.preprocessor is None:
            return []
        
        feature_names = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                # 获取 one-hot 编码后的特征名
                cat_features = transformer.get_feature_names_out(columns)
                feature_names.extend(cat_features)
        
        return feature_names
    
    def save(self, path: Path) -> None:
        """保存预处理器"""
        with open(path, 'wb') as f:
            pickle.dump({
                'preprocessor': self.preprocessor,
                'target_encoder': self.target_encoder,
                'fitted': self.fitted,
            }, f)
    
    def load(self, path: Path) -> None:
        """加载预处理器"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.preprocessor = data['preprocessor']
            self.target_encoder = data['target_encoder']
            self.fitted = data['fitted']


class DataManager:
    """数据管理主类"""
    
    def __init__(self):
        self.detector = DataTypeDetector()
        self.datasets: dict[str, DataSpec] = {}
    
    def upload_csv(
        self,
        content: bytes,
        filename: str,
        dataset_id: str | None = None,
        target_hint: str | None = None,
    ) -> DataSpec:
        """上传并解析 CSV 文件"""
        
        if dataset_id is None:
            import uuid
            dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
        
        # 保存原始文件
        file_path = DATA_DIR / f"{dataset_id}_{filename}"
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # 读取数据
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"无法解析 CSV: {e}")
        
        # 分析每列
        columns_info = []
        column_types = {}
        
        for col in df.columns:
            info = self.detector.analyze_column(df[col])
            columns_info.append(info)
            column_types[col] = info.column_type
        
        # 推断目标列
        target_column = self.detector.infer_target_column(df, target_hint)
        
        # 推断ID列
        id_column = None
        for col_info in columns_info:
            if col_info.column_type == ColumnType.ID:
                id_column = col_info.name
                break
        
        # 确定特征列
        feature_columns = [
            c.name for c in columns_info
            if c.name not in [target_column, id_column] and c.column_type != ColumnType.ID
        ]
        
        spec = DataSpec(
            dataset_id=dataset_id,
            filename=filename,
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=columns_info,
            target_column=target_column,
            id_column=id_column,
            feature_columns=feature_columns,
        )
        
        # 保存元数据
        meta_path = DATA_DIR / f"{dataset_id}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)
        
        self.datasets[dataset_id] = spec
        return spec
    
    def get_dataset(self, dataset_id: str) -> DataSpec:
        """获取数据集信息"""
        if dataset_id in self.datasets:
            return self.datasets[dataset_id]
        
        # 尝试从文件加载
        meta_path = DATA_DIR / f"{dataset_id}_meta.json"
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            spec = DataSpec(
                dataset_id=data['dataset_id'],
                filename=data['filename'],
                n_rows=data['n_rows'],
                n_cols=data['n_cols'],
                target_column=data.get('target_column'),
                id_column=data.get('id_column'),
                feature_columns=data.get('feature_columns', []),
            )
            self.datasets[dataset_id] = spec
            return spec
        
        raise ValueError(f"数据集不存在: {dataset_id}")
    
    def load_dataframe(self, dataset_id: str) -> pd.DataFrame:
        """加载数据为 DataFrame"""
        spec = self.get_dataset(dataset_id)
        file_path = DATA_DIR / f"{dataset_id}_{spec.filename}"
        return pd.read_csv(file_path)
    
    def list_datasets(self) -> list[DataSpec]:
        """列出所有数据集"""
        # 扫描数据目录
        for meta_file in DATA_DIR.glob("*_meta.json"):
            dataset_id = meta_file.stem.replace("_meta", "")
            if dataset_id not in self.datasets:
                try:
                    self.get_dataset(dataset_id)
                except:
                    pass
        
        return list(self.datasets.values())


# 全局数据管理器实例
data_manager = DataManager()
