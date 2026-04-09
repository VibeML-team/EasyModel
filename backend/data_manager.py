"""
数据管理模块 - 通用数据上传、AI 驱动探索、特征工程

设计理念：不预设用户上传的是什么格式。
Agent 拿到文件后，自己打开看、理解、报告。
"""

from __future__ import annotations

import csv
import json
import os
import pickle
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================
# 数据规范（通用，不限于表格）
# =============================================

class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TARGET = "target"
    ID = "id"


@dataclass
class ColumnInfo:
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
    """通用数据规范 — 适用于表格、图像、文本等任何格式"""
    dataset_id: str
    filename: str
    n_rows: int = 0          # 表格行数 / 图像数量 / 样本数
    n_cols: int = 0          # 表格列数
    data_type: str = "unknown"  # tabular / image_detection / image_classification / text / audio / unknown
    columns: list[ColumnInfo] = field(default_factory=list)
    target_column: str | None = None
    id_column: str | None = None
    feature_columns: list[str] = field(default_factory=list)
    # 通用元数据（LLM 探索结果存这里）
    metadata: dict[str, Any] = field(default_factory=dict)
    # 文件结构快照
    file_scan: dict[str, Any] = field(default_factory=dict)
    # LLM 探索结果
    exploration: dict[str, Any] = field(default_factory=dict)
    # 原始数据存储路径
    storage_path: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "data_type": self.data_type,
            "target_column": self.target_column,
            "id_column": self.id_column,
            "feature_columns": self.feature_columns,
            "metadata": self.metadata,
            "file_scan": self.file_scan,
            "exploration": self.exploration,
            "storage_path": self.storage_path,
            "columns": [c.to_dict() for c in self.columns],
        }


# =============================================
# 通用文件扫描器 — 不解析，只观察
# =============================================

class FileScanner:
    """
    扫描目录结构，生成一份"文件观察报告"给 LLM Agent。
    
    不预设任何格式。只是忠实记录看到了什么：
    - 目录树
    - 文件扩展名统计
    - 采样几个文件的内容（文本文件读前几行，二进制文件报告大小）
    """
    
    # 文本类扩展名
    TEXT_EXTS = {
        '.csv', '.tsv', '.txt', '.json', '.jsonl', '.xml', '.yaml', '.yml',
        '.md', '.py', '.cfg', '.ini', '.conf', '.log', '.names', '.data',
        '.labels', '.classes',
    }
    
    # 图像类扩展名
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    
    # 音频类扩展名
    AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    # 模型/权重文件
    MODEL_EXTS = {'.pt', '.pth', '.onnx', '.h5', '.pb', '.tflite', '.bin', '.safetensors', '.ckpt'}
    
    # 数据文件
    DATA_EXTS = {'.csv', '.tsv', '.xlsx', '.xls', '.parquet', '.feather', '.sqlite', '.db', '.hdf5'}
    
    # 忽略的文件/目录
    IGNORE_PATTERNS = {'__MACOSX', '.DS_Store', '__pycache__', '.git', 'Thumbs.db'}
    
    @classmethod
    def scan_directory(cls, root_dir: str | Path, max_files: int = 500) -> dict[str, Any]:
        """
        扫描目录，返回结构化的观察报告。
        这份报告会直接喂给 LLM，让它判断这是什么数据集。
        """
        root = Path(root_dir)
        
        # 收集文件信息
        all_files = []
        ext_counter = Counter()
        total_size = 0
        dir_set = set()
        
        for f in root.rglob('*'):
            # 跳过忽略的
            if any(p in str(f) for p in cls.IGNORE_PATTERNS):
                continue
            if f.is_dir():
                rel = str(f.relative_to(root))
                if len(dir_set) < 50:
                    dir_set.add(rel)
                continue
            if not f.is_file():
                continue
            
            rel_path = str(f.relative_to(root))
            ext = f.suffix.lower()
            size = f.stat().st_size
            
            ext_counter[ext] += 1
            total_size += size
            
            if len(all_files) < max_files:
                all_files.append({
                    "path": rel_path,
                    "ext": ext,
                    "size": size,
                })
        
        # 按扩展名分类统计
        ext_stats = dict(ext_counter.most_common(20))
        
        # 采样文件内容（给 LLM 看）
        samples = cls._sample_file_contents(root, all_files)
        
        # 生成目录树（简化版，只显示前2层 + 关键文件）
        tree = cls._build_tree_summary(root, max_depth=3, max_items=30)
        
        return {
            "total_files": sum(ext_counter.values()),
            "total_size_bytes": total_size,
            "total_size_human": cls._human_size(total_size),
            "extensions": ext_stats,
            "directories": sorted(list(dir_set))[:30],
            "directory_tree": tree,
            "file_samples": samples,
            "has_images": any(ext in cls.IMAGE_EXTS for ext in ext_counter),
            "has_tabular": any(ext in cls.DATA_EXTS for ext in ext_counter),
            "has_text": any(ext in cls.TEXT_EXTS for ext in ext_counter),
            "has_audio": any(ext in cls.AUDIO_EXTS for ext in ext_counter),
            "has_models": any(ext in cls.MODEL_EXTS for ext in ext_counter),
        }
    
    @classmethod
    def _sample_file_contents(cls, root: Path, files: list[dict], max_samples: int = 10) -> list[dict]:
        """采样几个文件的内容，给 LLM 看"""
        samples = []
        sampled_exts = set()
        
        # 优先采样不同类型的文件
        # 1. 先找配置/元数据文件（yaml, json, txt at root level）
        priority_files = [f for f in files if f["ext"] in {'.yaml', '.yml', '.json', '.cfg', '.names', '.classes', '.data'}]
        # 2. 再找标注/标签文件
        annotation_files = [f for f in files if f["ext"] in {'.txt', '.xml', '.json'} and any(kw in f["path"].lower() for kw in ['label', 'annot', 'train', 'val', 'test'])]
        # 3. CSV/表格
        tabular_files = [f for f in files if f["ext"] in {'.csv', '.tsv'}]
        # 4. 其他文本
        other_text = [f for f in files if f["ext"] in cls.TEXT_EXTS and f not in priority_files + annotation_files + tabular_files]
        
        for file_list in [priority_files, annotation_files, tabular_files, other_text]:
            for finfo in file_list[:3]:
                if len(samples) >= max_samples:
                    break
                fp = root / finfo["path"]
                try:
                    content = cls._read_file_preview(fp, finfo["ext"])
                    if content:
                        samples.append({
                            "path": finfo["path"],
                            "ext": finfo["ext"],
                            "size": finfo["size"],
                            "content_preview": content,
                        })
                        sampled_exts.add(finfo["ext"])
                except Exception:
                    pass
        
        return samples
    
    @classmethod
    def _read_file_preview(cls, path: Path, ext: str, max_lines: int = 20, max_chars: int = 2000) -> str | None:
        """读取文件预览内容"""
        if ext in cls.IMAGE_EXTS or ext in cls.AUDIO_EXTS or ext in cls.MODEL_EXTS:
            return None  # 二进制文件不读内容
        
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = []
                total_chars = 0
                for i, line in enumerate(f):
                    if i >= max_lines or total_chars >= max_chars:
                        lines.append(f"... (共 {i+1}+ 行)")
                        break
                    lines.append(line.rstrip())
                    total_chars += len(line)
                return "\n".join(lines)
        except Exception:
            return None
    
    @classmethod
    def _build_tree_summary(cls, root: Path, max_depth: int = 3, max_items: int = 30) -> str:
        """生成简化的目录树"""
        lines = []
        count = 0
        
        def _walk(dir_path: Path, prefix: str, depth: int):
            nonlocal count
            if depth > max_depth or count > max_items:
                return
            
            items = sorted(dir_path.iterdir(), key=lambda x: (x.is_file(), x.name))
            # 过滤忽略项
            items = [i for i in items if not any(p in i.name for p in cls.IGNORE_PATTERNS)]
            
            for i, item in enumerate(items):
                if count > max_items:
                    lines.append(f"{prefix}... ({len(items) - i} more)")
                    break
                
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "
                
                if item.is_dir():
                    n_children = sum(1 for _ in item.rglob('*') if _.is_file())
                    lines.append(f"{prefix}{connector}{item.name}/ ({n_children} files)")
                    count += 1
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    _walk(item, next_prefix, depth + 1)
                else:
                    size = cls._human_size(item.stat().st_size)
                    lines.append(f"{prefix}{connector}{item.name} ({size})")
                    count += 1
        
        _walk(root, "", 0)
        return "\n".join(lines)
    
    @staticmethod
    def _human_size(size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


class DatasetStructureAnalyzer:
    """确定性数据结构分析器，优先给出稳定结论，再由上层 LLM 补充业务解释。"""

    @classmethod
    def analyze(
        cls,
        scan_root: Path,
        file_scan: dict[str, Any],
        df: pd.DataFrame | None = None,
        target_column: str | None = None,
    ) -> dict[str, Any]:
        if df is not None and not df.empty:
            return cls._analyze_tabular(df, file_scan, target_column)

        image_analysis = cls._analyze_image_directory(scan_root, file_scan)
        if image_analysis:
            return image_analysis

        audio_analysis = cls._analyze_audio_directory(scan_root, file_scan)
        if audio_analysis:
            return audio_analysis

        return {
            "data_type": cls._infer_data_type(file_scan),
            "format_details": cls._summarize_extensions(file_scan),
            "business_summary": "检测到非结构化数据，但尚未识别出稳定的训练样式。",
            "data_understanding": {
                "summary": f"共 {file_scan.get('total_files', 0)} 个文件，大小 {file_scan.get('total_size_human', 'unknown')}",
                "organization": file_scan.get("directory_tree", ""),
                "key_files": [item.get("path", "") for item in file_scan.get("file_samples", [])[:8]],
            },
            "statistics": {
                "total_files": file_scan.get("total_files", 0),
                "extensions": file_scan.get("extensions", {}),
            },
            "quality_issues": ["需要进一步确认标签组织方式或提供训练目标。"],
            "training_implications": [
                "当前可以先完成数据归档与扫描，但训练前还需要明确标签、切分或样本语义。",
            ],
            "suggested_next_steps": [
                "补充目标字段或标签目录说明。",
                "如为网页抓取结果，优先清洗为 CSV/JSONL 或标准目录结构。",
            ],
        }

    @classmethod
    def _analyze_tabular(
        cls,
        df: pd.DataFrame,
        file_scan: dict[str, Any],
        target_column: str | None,
    ) -> dict[str, Any]:
        text_columns = []
        numeric_columns = []
        categorical_columns = []

        for col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                numeric_columns.append(col)
                continue

            non_null = series.dropna().astype(str)
            avg_len = float(non_null.str.len().mean()) if not non_null.empty else 0.0
            uniq_ratio = float(series.nunique(dropna=True)) / max(len(series), 1)
            if avg_len >= 12 and uniq_ratio > 0.1:
                text_columns.append(col)
            else:
                categorical_columns.append(col)

        if text_columns and target_column and target_column not in text_columns:
            data_type = "text_classification"
        else:
            data_type = "tabular"

        missing_cols = [
            col for col in df.columns
            if int(df[col].isna().sum()) > 0
        ]
        duplicate_rows = int(df.duplicated().sum())

        summary = f"{len(df)} 行 × {len(df.columns)} 列"
        if data_type == "text_classification":
            summary += f"，检测到文本字段 {', '.join(text_columns[:3])}"

        stats: dict[str, Any] = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "text_columns": text_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "duplicate_rows": duplicate_rows,
        }

        if target_column and target_column in df.columns:
            value_counts = df[target_column].value_counts(dropna=False).head(20)
            stats["target_distribution"] = {
                str(idx): int(val) for idx, val in value_counts.items()
            }

        quality_issues = []
        if missing_cols:
            quality_issues.append(f"存在缺失值列: {', '.join(missing_cols[:6])}")
        if duplicate_rows:
            quality_issues.append(f"检测到 {duplicate_rows} 行重复样本")

        implications = [
            "可直接走结构化训练流程，优先使用稳定的数据切分与预处理。",
        ]
        if data_type == "text_classification":
            implications = [
                "更适合文本分类/匹配训练，不建议仅按普通表格特征处理。",
                "训练时需要显式指定文本列与标签列。",
            ]

        return {
            "data_type": data_type,
            "format_details": cls._summarize_extensions(file_scan),
            "business_summary": f"检测到可直接训练的{'文本' if data_type == 'text_classification' else '表格'}数据集，{summary}。",
            "data_understanding": {
                "summary": summary,
                "organization": "单文件或少量表格文件组织",
                "key_files": [item.get("path", "") for item in file_scan.get("file_samples", [])[:8]],
                "target_column": target_column,
            },
            "statistics": stats,
            "quality_issues": quality_issues,
            "training_implications": implications,
            "suggested_next_steps": [
                "确认目标列与评估指标。",
                "必要时先做缺失值、异常值和类别不平衡处理。",
            ],
        }

    @classmethod
    def _analyze_image_directory(
        cls,
        scan_root: Path,
        file_scan: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not file_scan.get("has_images"):
            return None

        split_stats = cls._collect_split_class_stats(scan_root, FileScanner.IMAGE_EXTS)
        if not split_stats:
            total_images = sum(
                count for ext, count in file_scan.get("extensions", {}).items()
                if ext in FileScanner.IMAGE_EXTS
            )
            return {
                "data_type": "image",
                "format_details": cls._summarize_extensions(file_scan),
                "business_summary": f"检测到图像数据，共约 {total_images} 张图片，但目录结构尚未稳定识别为标准分类任务。",
                "data_understanding": {
                    "summary": "图像文件已落盘，但标签组织方式未明确。",
                    "organization": file_scan.get("directory_tree", ""),
                    "key_files": [item.get("path", "") for item in file_scan.get("file_samples", [])[:8]],
                },
                "statistics": {
                    "total_images": total_images,
                },
                "quality_issues": ["未识别到清晰的 train/val/test 或 label 目录结构。"],
                "training_implications": [
                    "训练前需要补充标签映射或将图像整理为标准目录结构。",
                ],
                "suggested_next_steps": [
                    "将数据整理为 ImageFolder 风格目录，例如 train/<label>/*.png。",
                ],
            }

        classes = sorted({label for split in split_stats.values() for label in split})
        total_images = sum(sum(labels.values()) for labels in split_stats.values())
        split_counts = {split: int(sum(labels.values())) for split, labels in split_stats.items()}

        return {
            "data_type": "image_classification",
            "format_details": "directory_split/class_labeled_images",
            "business_summary": f"检测到标准图像分类目录，共 {total_images} 张图片，类别数 {len(classes)}。",
            "data_understanding": {
                "summary": "目录结构符合 ImageFolder 训练范式。",
                "organization": ", ".join(f"{split}/<label>/*" for split in split_stats),
                "key_files": [item.get("path", "") for item in file_scan.get("file_samples", [])[:8]],
            },
            "statistics": {
                "total_samples": total_images,
                "classes": classes,
                "class_distribution": {
                    split: {label: int(count) for label, count in labels.items()}
                    for split, labels in split_stats.items()
                },
                "splits": split_counts,
            },
            "quality_issues": [],
            "training_implications": [
                "可直接生成 torchvision ImageFolder 风格训练代码。",
                "如果没有 val split，可从 train 中自动切出一部分验证集。",
            ],
            "suggested_next_steps": [
                "按图像分类训练方案执行。",
            ],
        }

    @classmethod
    def _analyze_audio_directory(
        cls,
        scan_root: Path,
        file_scan: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not file_scan.get("has_audio"):
            return None

        split_stats = cls._collect_split_class_stats(scan_root, FileScanner.AUDIO_EXTS)
        total_audio = sum(
            count for ext, count in file_scan.get("extensions", {}).items()
            if ext in FileScanner.AUDIO_EXTS
        )
        classes = sorted({label for split in split_stats.values() for label in split})

        return {
            "data_type": "audio_classification" if classes else "audio",
            "format_details": cls._summarize_extensions(file_scan),
            "business_summary": f"检测到音频数据，共约 {total_audio} 个文件。",
            "data_understanding": {
                "summary": "已识别音频文件集合。",
                "organization": file_scan.get("directory_tree", ""),
                "key_files": [item.get("path", "") for item in file_scan.get("file_samples", [])[:8]],
            },
            "statistics": {
                "total_samples": total_audio,
                "classes": classes,
                "splits": {split: int(sum(labels.values())) for split, labels in split_stats.items()},
            },
            "quality_issues": [] if classes else ["尚未识别出稳定的类别目录结构。"],
            "training_implications": [
                "音频训练前通常需要频谱图或波形特征提取。",
            ],
            "suggested_next_steps": [
                "确认采样率、标签目录和训练目标。",
            ],
        }

    @staticmethod
    def _collect_split_class_stats(root: Path, valid_exts: set[str]) -> dict[str, dict[str, int]]:
        split_names = {"train", "training", "val", "valid", "validation", "test", "dev"}
        stats: dict[str, dict[str, int]] = {}

        for split_dir in root.iterdir() if root.exists() else []:
            if not split_dir.is_dir():
                continue
            split_key = split_dir.name.lower()
            if split_key not in split_names:
                continue

            labels: dict[str, int] = {}
            for label_dir in split_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                count = sum(
                    1 for item in label_dir.rglob("*")
                    if item.is_file() and item.suffix.lower() in valid_exts
                )
                if count:
                    labels[label_dir.name] = count
            if labels:
                normalized_split = "val" if split_key in {"valid", "validation", "dev"} else split_key
                normalized_split = "train" if split_key == "training" else normalized_split
                stats[normalized_split] = labels

        return stats

    @staticmethod
    def _summarize_extensions(file_scan: dict[str, Any]) -> str:
        exts = file_scan.get("extensions", {})
        if not exts:
            return "unknown"
        return ", ".join(f"{ext or '[no_ext]'}({count})" for ext, count in list(exts.items())[:8])

    @staticmethod
    def _infer_data_type(file_scan: dict[str, Any]) -> str:
        if file_scan.get("has_tabular"):
            return "tabular"
        if file_scan.get("has_images"):
            return "image"
        if file_scan.get("has_audio"):
            return "audio"
        if file_scan.get("has_text"):
            return "text"
        return "unknown"


class DataTypeDetector:
    """表格数据的列类型检测器（用于表格类数据集）"""
    
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
    """
    数据管理主类
    
    核心流程：
    1. 接收文件（任意格式）→ 保存到磁盘
    2. 如果是压缩包 → 解压
    3. 扫描目录结构 → 生成观察报告
    4. （可选）LLM Agent 分析报告 → 理解数据
    5. 如果是表格数据 → 额外做列分析
    """
    
    def __init__(self):
        self.detector = DataTypeDetector()
        self.scanner = FileScanner()
        self.datasets: dict[str, DataSpec] = {}
    
    def upload_file(
        self,
        content: bytes,
        filename: str,
        dataset_id: str | None = None,
        target_hint: str | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> DataSpec:
        """
        上传任意格式的文件。
        
        不预设格式——保存、解压、扫描，然后让 AI Agent 分析。
        
        Args:
            progress_callback: 进度回调 (step_id, message)，用于实时推送给前端
        """
        import uuid
        import shutil
        
        if dataset_id is None:
            dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
        
        def _emit(step: str, msg: str):
            if progress_callback:
                progress_callback(step, msg)
        
        # Step 1: 保存原始文件
        _emit("save", f"📥 接收文件: {filename} ({FileScanner._human_size(len(content))})")
        
        dataset_dir = DATA_DIR / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        raw_file = dataset_dir / filename
        with open(raw_file, 'wb') as f:
            f.write(content)
        
        # Step 2: 解压（如果是压缩包）
        ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
        extract_dir = dataset_dir / "extracted"
        
        if ext == 'zip':
            _emit("extract", f"📦 正在解压 ZIP 文件...")
            import zipfile
            try:
                with zipfile.ZipFile(raw_file, 'r') as zf:
                    zf.extractall(extract_dir)
                _emit("extract", f"✅ 解压完成，共 {sum(1 for _ in extract_dir.rglob('*') if _.is_file())} 个文件")
            except Exception as e:
                _emit("extract", f"⚠️ 解压失败: {e}")
                raise ValueError(f"ZIP 解压失败: {e}")
            scan_root = extract_dir
        elif ext == '7z':
            _emit("extract", f"📦 正在解压 7Z 文件...")
            try:
                import py7zr
                with py7zr.SevenZipFile(raw_file, mode='r') as z:
                    z.extractall(path=str(extract_dir))
                _emit("extract", f"✅ 解压完成")
            except ImportError:
                raise ValueError("处理 7Z 文件需要安装 py7zr")
            except Exception as e:
                raise ValueError(f"7Z 解压失败: {e}")
            scan_root = extract_dir
        elif ext in ('tar', 'gz', 'tgz'):
            _emit("extract", f"📦 正在解压 tar 文件...")
            import tarfile
            try:
                with tarfile.open(raw_file, 'r:*') as tf:
                    tf.extractall(extract_dir)
                _emit("extract", f"✅ 解压完成")
            except Exception as e:
                raise ValueError(f"tar 解压失败: {e}")
            scan_root = extract_dir
        else:
            # 非压缩包，直接扫描文件本身
            scan_root = dataset_dir
        
        # Step 3: 扫描目录结构
        _emit("scan", "🔍 正在扫描文件结构...")
        file_scan = self.scanner.scan_directory(scan_root)
        _emit("scan", f"📁 发现 {file_scan['total_files']} 个文件 ({file_scan['total_size_human']})")
        
        exts = file_scan['extensions']
        if exts:
            ext_summary = ', '.join(f"{e}({n})" for e, n in list(exts.items())[:6])
            _emit("scan", f"📋 文件类型: {ext_summary}")
            
        # Step 4: 尝试读取表格数据（如果有的话）
        df = None
        columns_info = []
        n_rows = file_scan['total_files']  # 默认用文件数作为样本数
        n_cols = 0
        data_type = "unknown"
        target_column = None
        id_column = None
        feature_columns = []
        
        if file_scan['has_tabular']:
            _emit("parse", "📊 发现表格数据，正在解析...")
            try:
                df = self._try_read_tabular(scan_root)
                if df is not None and not df.empty:
                    n_rows = len(df)
                    n_cols = len(df.columns)
                    data_type = "tabular"
                    _emit("parse", f"✅ 表格解析成功: {n_rows} 行 × {n_cols} 列")
                    
                    # 列分析
                    for col in df.columns:
                        info = self.detector.analyze_column(df[col])
                        columns_info.append(info)
                    
                    target_column = self.detector.infer_target_column(df, target_hint)
                    for ci in columns_info:
                        if ci.column_type == ColumnType.ID:
                            id_column = ci.name
                            break
                    feature_columns = [
                        c.name for c in columns_info
                        if c.name not in [target_column, id_column] and c.column_type != ColumnType.ID
                    ]
                    
                    # 保存为标准 CSV
                    csv_path = dataset_dir / f"{dataset_id}_data.csv"
                    df.to_csv(csv_path, index=False)
            except Exception as e:
                _emit("parse", f"⚠️ 表格解析失败: {e}")
        
        if file_scan['has_images']:
            img_count = sum(n for ext, n in exts.items() if ext in FileScanner.IMAGE_EXTS)
            _emit("scan", f"🖼️ 发现 {img_count} 张图片")
            if data_type == "unknown":
                data_type = "image"
                n_rows = img_count
        
        if file_scan['has_audio']:
            audio_count = sum(n for ext, n in exts.items() if ext in FileScanner.AUDIO_EXTS)
            _emit("scan", f"🎵 发现 {audio_count} 个音频文件")
            if data_type == "unknown":
                data_type = "audio"
                n_rows = audio_count
        
        if file_scan['has_models']:
            model_count = sum(n for ext, n in exts.items() if ext in FileScanner.MODEL_EXTS)
            _emit("scan", f"🧠 发现 {model_count} 个模型/权重文件")
        
        _emit("done", "🤖 文件扫描完成，等待 AI Agent 深度分析...")

        exploration = DatasetStructureAnalyzer.analyze(
            scan_root=scan_root,
            file_scan=file_scan,
            df=df,
            target_column=target_column,
        )
        if exploration.get("data_type"):
            data_type = exploration["data_type"]

        # 构建 DataSpec
        spec = DataSpec(
            dataset_id=dataset_id,
            filename=filename,
            n_rows=n_rows,
            n_cols=n_cols,
            data_type=data_type,
            columns=columns_info,
            target_column=target_column,
            id_column=id_column,
            feature_columns=feature_columns,
            file_scan=file_scan,
            exploration=exploration,
            storage_path=str(dataset_dir),
        )
        
        # 保存元数据
        meta_path = DATA_DIR / f"{dataset_id}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)
        
        self.datasets[dataset_id] = spec
        return spec
            
    def upload_from_disk(
        self,
        file_path: Path,
        filename: str,
        dataset_id: str,
        target_hint: str | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> DataSpec:
        """
        处理已在磁盘上的大文件（不读入内存）。
        直接在文件所在目录操作。
        """
        import shutil
        
        def _emit(step: str, msg: str):
            if progress_callback:
                progress_callback(step, msg)
        
        file_path = Path(file_path)
        dataset_dir = file_path.parent
        ext = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
        extract_dir = dataset_dir / "extracted"
        
        # 解压
        if ext == 'zip':
            _emit("extract", f"📦 正在解压 ZIP ({self.scanner._human_size(file_path.stat().st_size)})...")
            import zipfile
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    zf.extractall(extract_dir)
                n_files = sum(1 for _ in extract_dir.rglob('*') if _.is_file())
                _emit("extract", f"✅ 解压完成，共 {n_files} 个文件")
            except Exception as e:
                raise ValueError(f"ZIP 解压失败: {e}")
            scan_root = extract_dir
        elif ext == '7z':
            _emit("extract", "📦 正在解压 7Z...")
            try:
                import py7zr
                with py7zr.SevenZipFile(str(file_path), mode='r') as z:
                    z.extractall(path=str(extract_dir))
            except Exception as e:
                raise ValueError(f"7Z 解压失败: {e}")
            scan_root = extract_dir
        else:
            scan_root = dataset_dir
        
        # 扫描
        _emit("scan", "🔍 正在扫描文件结构...")
        file_scan = self.scanner.scan_directory(scan_root)
        _emit("scan", f"📁 发现 {file_scan['total_files']} 个文件 ({file_scan['total_size_human']})")
        
        exts = file_scan['extensions']
        if exts:
            ext_summary = ', '.join(f"{e}({n})" for e, n in list(exts.items())[:6])
            _emit("scan", f"📋 文件类型: {ext_summary}")
        
        n_rows = file_scan['total_files']
        data_type = "unknown"
        if file_scan.get('has_images'):
            img_count = sum(n for e, n in exts.items() if e in FileScanner.IMAGE_EXTS)
            _emit("scan", f"🖼️ 发现 {img_count} 张图片")
            data_type = "image"
            n_rows = img_count
        if file_scan.get('has_tabular'):
            data_type = "tabular" if not file_scan.get('has_images') else data_type

        _emit("done", "🤖 文件扫描完成，等待 AI Agent 探索...")

        df = None
        target_column = None
        feature_columns: list[str] = []
        columns_info: list[ColumnInfo] = []
        if file_scan.get("has_tabular"):
            try:
                df = self._try_read_tabular(scan_root)
                if df is not None and not df.empty:
                    target_column = self.detector.infer_target_column(df, target_hint)
                    columns_info = [self.detector.analyze_column(df[col]) for col in df.columns]
                    feature_columns = [col for col in df.columns if col != target_column]
                    csv_path = dataset_dir / f"{dataset_id}_data.csv"
                    df.to_csv(csv_path, index=False)
            except Exception:
                pass

        exploration = DatasetStructureAnalyzer.analyze(
            scan_root=scan_root,
            file_scan=file_scan,
            df=df,
            target_column=target_column,
        )
        if exploration.get("data_type"):
            data_type = exploration["data_type"]

        spec = DataSpec(
            dataset_id=dataset_id,
            filename=filename,
            n_rows=int(len(df)) if df is not None and not df.empty else n_rows,
            n_cols=int(len(df.columns)) if df is not None and not df.empty else 0,
            data_type=data_type,
            columns=columns_info,
            target_column=target_column,
            feature_columns=feature_columns,
            file_scan=file_scan,
            exploration=exploration,
            storage_path=str(dataset_dir),
        )
        
        meta_path = DATA_DIR / f"{dataset_id}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(spec.to_dict(), f, ensure_ascii=False, indent=2)
        
        self.datasets[dataset_id] = spec
        return spec
    
    def _try_read_tabular(self, root: Path) -> pd.DataFrame | None:
        """尝试从目录中读取表格数据（任何格式）"""
        def _find(patterns):
            results = []
            for pat in patterns:
                for f in root.rglob(pat):
                    if not f.name.startswith('.') and '__MACOSX' not in str(f):
                        results.append(f)
            return sorted(results, key=lambda f: f.stat().st_size, reverse=True)

        def _try_read_csv(path: Path, sep: str | None = None) -> pd.DataFrame | None:
            encodings = ["utf-8", "utf-8-sig", "gb18030", "latin-1"]
            for encoding in encodings:
                try:
                    if sep is not None:
                        df = pd.read_csv(path, encoding=encoding, sep=sep, low_memory=False)
                    else:
                        df = pd.read_csv(path, encoding=encoding, low_memory=False)
                    if not df.empty and len(df.columns) > 0:
                        return df
                except Exception:
                    if sep is None:
                        try:
                            df = pd.read_csv(
                                path,
                                encoding=encoding,
                                sep=None,
                                engine="python",
                                quoting=csv.QUOTE_MINIMAL,
                                low_memory=False,
                            )
                            if not df.empty and len(df.columns) > 0:
                                return df
                        except Exception:
                            continue
            return None
        
        # CSV
        for f in _find(['*.csv']):
            df = _try_read_csv(f)
            if df is not None:
                return df
        
        # TSV
        for f in _find(['*.tsv']):
            df = _try_read_csv(f, sep="\t")
            if df is not None:
                return df
        
        # Excel
        for f in _find(['*.xlsx', '*.xls']):
            try:
                return pd.read_excel(f)
            except Exception:
                continue
        
        # Parquet
        for f in _find(['*.parquet']):
            try:
                return pd.read_parquet(f)
            except Exception:
                continue
        
        # JSON (tabular)
        for f in _find(['*.json']):
            try:
                data = json.loads(f.read_text(encoding='utf-8'))
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    return pd.DataFrame(data)
                if isinstance(data, dict):
                    if isinstance(data.get("data"), list) and data["data"] and isinstance(data["data"][0], dict):
                        return pd.DataFrame(data["data"])
                    if all(isinstance(v, list) for v in data.values()):
                        return pd.DataFrame(data)
            except Exception:
                continue
        
        # JSONL
        for f in _find(['*.jsonl']):
            try:
                return pd.read_json(f, lines=True)
            except Exception:
                continue
        
        return None
    
    def get_file_scan(self, dataset_id: str) -> dict[str, Any]:
        """获取数据集的文件扫描报告（供 LLM 分析）"""
        spec = self.get_dataset(dataset_id)
        if spec.file_scan:
            return spec.file_scan
        
        # 如果没有缓存的扫描，重新扫描
        storage = Path(spec.storage_path) if spec.storage_path else DATA_DIR / dataset_id
        if storage.exists():
            return self.scanner.scan_directory(storage)
        return {}
    
    def get_dataset(self, dataset_id: str) -> DataSpec:
        """获取数据集信息"""
        if dataset_id in self.datasets:
            return self.datasets[dataset_id]
        
        meta_path = DATA_DIR / f"{dataset_id}_meta.json"
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            spec = DataSpec(
                dataset_id=data['dataset_id'],
                filename=data['filename'],
                n_rows=data.get('n_rows', 0),
                n_cols=data.get('n_cols', 0),
                data_type=data.get('data_type', 'unknown'),
                columns=[
                    ColumnInfo(
                        name=col.get("name", ""),
                        column_type=ColumnType(col.get("column_type", "text")),
                        dtype=col.get("dtype", ""),
                        missing_count=col.get("missing_count", 0),
                        unique_count=col.get("unique_count", 0),
                        sample_values=col.get("sample_values", []),
                        statistics=col.get("statistics", {}),
                    )
                    for col in data.get("columns", [])
                    if col.get("name")
                ],
                target_column=data.get('target_column'),
                id_column=data.get('id_column'),
                feature_columns=data.get('feature_columns', []),
                metadata=data.get('metadata', {}),
                file_scan=data.get('file_scan', {}),
                exploration=data.get('exploration', {}),
                storage_path=data.get('storage_path', ''),
            )
            self.datasets[dataset_id] = spec
            return spec
        
        raise ValueError(f"数据集不存在: {dataset_id}")
    
    def load_dataframe(self, dataset_id: str) -> pd.DataFrame:
        """加载表格数据为 DataFrame"""
        spec = self.get_dataset(dataset_id)
        csv_path = Path(spec.storage_path) / f"{dataset_id}_data.csv" if spec.storage_path else DATA_DIR / spec.filename
        if csv_path.exists():
            return pd.read_csv(csv_path)
        # 回退：尝试数据集目录里的原始文件名
        alt_path = Path(spec.storage_path) / spec.filename if spec.storage_path else DATA_DIR / dataset_id / spec.filename
        if alt_path.exists():
            return pd.read_csv(alt_path)
        raise ValueError(f"表格数据文件不存在: {csv_path}")
    
    def list_datasets(self) -> list[DataSpec]:
        """列出所有数据集"""
        for meta_file in DATA_DIR.glob("*_meta.json"):
            dataset_id = meta_file.stem.replace("_meta", "")
            if dataset_id not in self.datasets:
                try:
                    self.get_dataset(dataset_id)
                except Exception:
                    pass
        return list(self.datasets.values())


# 全局实例
data_manager = DataManager()
