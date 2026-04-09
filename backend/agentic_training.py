from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.compiler import ObjectiveSpec
from backend.data_manager import DataSpec, data_manager
from backend.modal_lab import ModalLabExecutor
from backend.trainer import CHECKPOINT_DIR, TrainingResult


@dataclass
class TrainingBundle:
    files: dict[str, str]
    binary_files: dict[str, bytes]
    entrypoint: str
    summary: str


class TrainingBundleGenerator:
    """按当前项目的训练规划生成可执行 bundle，而不是只生成抽象片段。"""

    def build(self, plan: dict[str, Any], dataset_spec: DataSpec, objective_spec: ObjectiveSpec) -> TrainingBundle:
        modality = plan.get("modality") or dataset_spec.data_type
        training_spec = (plan.get("inputs") or {}).get("training_spec") or {}

        if "image" in modality:
            return self._build_image_bundle(dataset_spec, objective_spec, training_spec)
        if "text" in modality:
            return self._build_text_bundle(dataset_spec, objective_spec)
        if modality in {"audio", "audio_classification"}:
            raise ValueError("当前版本已完成数据识别，但音频训练模板尚未接入。请先整理为频谱图图像或表格特征。")

        return self._build_text_bundle(dataset_spec, objective_spec)

    def _build_image_bundle(
        self,
        dataset_spec: DataSpec,
        objective_spec: ObjectiveSpec,
        training_spec: dict[str, Any],
    ) -> TrainingBundle:
        bundle = {
            "model.py": self._image_model_code(),
            "data_pipeline.py": self._image_data_pipeline_code(training_spec),
            "train_loop.py": self._image_train_loop_code(),
            "run_training.py": self._image_entrypoint_code(dataset_spec, objective_spec, training_spec),
        }
        return TrainingBundle(
            files=bundle,
            binary_files=_stage_directory_bytes(_resolve_scan_root(dataset_spec)),
            entrypoint="run_training.py",
            summary="torchvision ImageFolder CNN training bundle",
        )

    def _build_text_bundle(self, dataset_spec: DataSpec, objective_spec: ObjectiveSpec) -> TrainingBundle:
        bundle = {
            "run_training.py": self._text_entrypoint_code(dataset_spec, objective_spec),
            "dataset/data.csv": data_manager.load_dataframe(dataset_spec.dataset_id).to_csv(index=False).encode("utf-8"),
        }
        return TrainingBundle(
            files={"run_training.py": bundle["run_training.py"]},
            binary_files={"dataset/data.csv": bundle["dataset/data.csv"]},
            entrypoint="run_training.py",
            summary="sklearn text classification training bundle",
        )

    @staticmethod
    def _image_model_code() -> str:
        return textwrap.dedent(
            """
            import torch
            import torch.nn as nn


            class SimpleImageClassifier(nn.Module):
                def __init__(self, num_classes: int, in_channels: int = 3):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                    )
                    self.classifier = nn.Linear(128, num_classes)

                def forward(self, x):
                    feats = self.features(x)
                    feats = feats.view(feats.size(0), -1)
                    return self.classifier(feats)
            """
        ).strip()

    @staticmethod
    def _image_data_pipeline_code(training_spec: dict[str, Any]) -> str:
        color_mode = training_spec.get("color_mode", "RGB")
        image_size = training_spec.get("image_size") or [64, 64]
        input_channels = int(training_spec.get("input_channels") or (1 if color_mode == "L" else 3))
        normalization = training_spec.get("normalization") or (
            {"mean": [0.5], "std": [0.5]} if input_channels == 1 else {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        )
        return textwrap.dedent(
            f"""
            from pathlib import Path

            import torch
            from torch.utils.data import DataLoader, random_split
            from torchvision import datasets, transforms


            class ImageDataModule:
                def __init__(self, dataset_root: str, batch_size: int = 32):
                    self.dataset_root = Path(dataset_root)
                    self.batch_size = batch_size
                    self.mode = {json.dumps(color_mode)}
                    self.in_channels = {input_channels}
                    convert_step = transforms.Grayscale(num_output_channels=1) if self.mode == "L" else transforms.Lambda(lambda img: img.convert("RGB"))
                    self.transform = transforms.Compose([
                        convert_step,
                        transforms.Resize(({image_size[0]}, {image_size[1]})),
                        transforms.ToTensor(),
                        transforms.Normalize(mean={json.dumps(normalization.get("mean", []))}, std={json.dumps(normalization.get("std", []))}),
                    ])

                def setup(self):
                    train_root = self.dataset_root / "train"
                    val_root = self.dataset_root / "val"
                    test_root = self.dataset_root / "test"

                    if train_root.exists():
                        train_full = datasets.ImageFolder(str(train_root), transform=self.transform)
                    else:
                        train_full = datasets.ImageFolder(str(self.dataset_root), transform=self.transform)

                    if val_root.exists():
                        self.train_dataset = train_full
                        self.val_dataset = datasets.ImageFolder(str(val_root), transform=self.transform)
                    else:
                        val_size = max(1, int(len(train_full) * 0.2))
                        train_size = max(1, len(train_full) - val_size)
                        self.train_dataset, self.val_dataset = random_split(
                            train_full,
                            [train_size, val_size],
                            generator=torch.Generator().manual_seed(42),
                        )

                    if test_root.exists():
                        self.test_dataset = datasets.ImageFolder(str(test_root), transform=self.transform)
                    else:
                        self.test_dataset = self.val_dataset

                    self.classes = train_full.classes

                def train_dataloader(self):
                    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

                def val_dataloader(self):
                    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

                def test_dataloader(self):
                    return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            """
        ).strip()

    @staticmethod
    def _image_train_loop_code() -> str:
        return textwrap.dedent(
            """
            import time

            import torch


            class Trainer:
                def __init__(self, device: str = "cpu", epochs: int = 5, lr: float = 1e-3):
                    self.device = device
                    self.epochs = epochs
                    self.lr = lr

                def fit(self, model, datamodule):
                    criterion = torch.nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                    model.to(self.device)
                    start_time = time.time()
                    history = []

                    best_state = None
                    best_acc = -1.0

                    for epoch in range(self.epochs):
                        model.train()
                        total_loss = 0.0
                        total_correct = 0
                        total = 0

                        for inputs, targets in datamodule.train_dataloader():
                            inputs, targets = inputs.to(self.device), targets.to(self.device)
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()

                            total_loss += float(loss.item()) * inputs.size(0)
                            total_correct += int((outputs.argmax(dim=1) == targets).sum().item())
                            total += int(inputs.size(0))

                        val_metrics = self.evaluate(model, datamodule.val_dataloader())
                        epoch_metrics = {
                            "epoch": epoch + 1,
                            "train_loss": total_loss / max(total, 1),
                            "train_acc": total_correct / max(total, 1),
                            **{f"val_{k}": v for k, v in val_metrics.items()},
                        }
                        history.append(epoch_metrics)
                        elapsed_minutes = (time.time() - start_time) / 60
                        print(f"[VIBEML_CALLBACK] epoch={epoch + 1}, time={elapsed_minutes:.2f}, val_accuracy={val_metrics['accuracy']:.4f}")

                        if val_metrics["accuracy"] >= best_acc:
                            best_acc = val_metrics["accuracy"]
                            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

                    if best_state is not None:
                        model.load_state_dict(best_state)

                    return history

                def evaluate(self, model, dataloader):
                    criterion = torch.nn.CrossEntropyLoss()
                    model.eval()
                    total_loss = 0.0
                    total_correct = 0
                    total = 0
                    with torch.no_grad():
                        for inputs, targets in dataloader:
                            inputs, targets = inputs.to(self.device), targets.to(self.device)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            total_loss += float(loss.item()) * inputs.size(0)
                            total_correct += int((outputs.argmax(dim=1) == targets).sum().item())
                            total += int(inputs.size(0))
                    return {
                        "loss": total_loss / max(total, 1),
                        "accuracy": total_correct / max(total, 1),
                    }
            """
        ).strip()

    @staticmethod
    def _image_entrypoint_code(
        dataset_spec: DataSpec,
        objective_spec: ObjectiveSpec,
        training_spec: dict[str, Any],
    ) -> str:
        dataset_root = json.dumps("dataset")
        epochs = max(3, min(12, objective_spec.max_trials or 5))
        num_classes_hint = training_spec.get("num_classes")
        return textwrap.dedent(
            f"""
            import json
            import os
            from pathlib import Path

            import torch

            from data_pipeline import ImageDataModule
            from model import SimpleImageClassifier
            from train_loop import Trainer


            def main():
                dataset_root = {dataset_root}
                artifact_dir = Path(os.environ.get("VIBEML_ARTIFACT_DIR", "artifacts"))
                artifact_dir.mkdir(parents=True, exist_ok=True)

                datamodule = ImageDataModule(dataset_root=dataset_root, batch_size=32)
                datamodule.setup()

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = SimpleImageClassifier(
                    num_classes={int(num_classes_hint) if num_classes_hint else 'len(datamodule.classes)'},
                    in_channels=datamodule.in_channels,
                )
                trainer = Trainer(device=device, epochs={epochs}, lr=1e-3)
                history = trainer.fit(model, datamodule)
                test_metrics = trainer.evaluate(model, datamodule.test_dataloader())

                model_path = artifact_dir / "model.pt"
                meta_path = artifact_dir / "metrics.json"
                torch.save({{
                    "state_dict": model.state_dict(),
                    "classes": datamodule.classes,
                    "in_channels": datamodule.in_channels,
                }}, model_path)

                payload = {{
                    "history": history,
                    "test_metrics": test_metrics,
                    "classes": datamodule.classes,
                    "device": device,
                }}
                meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                print("\\n=== Test Metrics ===")
                print(f"accuracy: {{test_metrics['accuracy']:.4f}}")
                print(f"loss: {{test_metrics['loss']:.4f}}")


            if __name__ == "__main__":
                main()
            """
        ).strip()

    @staticmethod
    def _text_entrypoint_code(dataset_spec: DataSpec, objective_spec: ObjectiveSpec) -> str:
        target_column = json.dumps(dataset_spec.target_column or objective_spec.label.target_column or "label")
        return textwrap.dedent(
            f"""
            import json
            import os
            import pickle
            from pathlib import Path

            import pandas as pd
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline


            def pick_text_column(df, target_column):
                candidates = []
                for col in df.columns:
                    if col == target_column:
                        continue
                    series = df[col].dropna().astype(str)
                    if not series.empty and series.str.len().mean() >= 16:
                        candidates.append(col)
                if not candidates:
                    raise ValueError("无法自动识别文本列")
                return candidates[0]


            def main():
                artifact_dir = Path(os.environ.get("VIBEML_ARTIFACT_DIR", "artifacts"))
                artifact_dir.mkdir(parents=True, exist_ok=True)

                df = pd.read_csv("dataset/data.csv")
                target_column = {target_column}
                if target_column not in df.columns:
                    raise ValueError(f"target column not found: {{target_column}}")
                text_column = pick_text_column(df, target_column)
                X_train, X_test, y_train, y_test = train_test_split(
                    df[text_column].astype(str),
                    df[target_column].astype(str),
                    test_size=0.2,
                    random_state=42,
                    stratify=df[target_column].astype(str) if df[target_column].nunique() < len(df) else None,
                )
                pipeline = Pipeline([
                    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
                    ("clf", LogisticRegression(max_iter=300)),
                ])
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)
                metrics = {{
                    "accuracy": float(accuracy_score(y_test, preds)),
                    "f1": float(f1_score(y_test, preds, average="weighted")),
                    "text_column": text_column,
                    "target_column": target_column,
                }}
                with open(artifact_dir / "model.pkl", "wb") as f:
                    pickle.dump(pipeline, f)
                (artifact_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"accuracy: {{metrics['accuracy']:.4f}}")
                print(f"f1: {{metrics['f1']:.4f}}")


            if __name__ == "__main__":
                main()
            """
        ).strip()


class PlanAwareTrainingExecutor:
    def __init__(self, modal_executor: ModalLabExecutor | None = None):
        self.bundle_generator = TrainingBundleGenerator()
        self.modal_executor = modal_executor or ModalLabExecutor()

    def execute(
        self,
        *,
        job_id: str,
        plan: dict[str, Any],
        objective_spec: ObjectiveSpec,
        progress_callback=None,
    ) -> TrainingResult:
        dataset_id = plan["inputs"]["dataset_id"]
        dataset_spec = data_manager.get_dataset(dataset_id)

        if progress_callback:
            progress_callback({
                "step": "agentic_log",
                "agent_stage": "generating",
                "message": "按训练规划生成可执行训练 bundle...",
                "log_entry": "bundle generation started",
            })

        bundle = self.bundle_generator.build(plan, dataset_spec, objective_spec)

        if progress_callback:
            progress_callback({
                "step": "agentic_log",
                "agent_stage": "validating",
                "message": f"bundle 已生成，准备提交到 {self.modal_executor.config.app_name or 'local runner'} 执行...",
                "log_entry": "bundle validation completed",
            })

        execution = self.modal_executor.run_bundle(
            bundle_files=bundle.files,
            binary_files=bundle.binary_files,
            entrypoint=bundle.entrypoint,
            job_id=job_id,
            gpu_type=plan["inputs"].get("gpu_type"),
            timeout_seconds=int(plan["inputs"].get("timeout_seconds", objective_spec.max_training_time or 3600)),
        )

        output_dir = CHECKPOINT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path: str | None = None
        metadata_path = output_dir / "result.json"
        preprocessor_path: str | None = None

        for rel_path, payload in execution.artifacts.items():
            target_path = output_dir / Path(rel_path).name
            target_path.write_bytes(payload)
            lower_name = target_path.name.lower()
            if lower_name.endswith((".pt", ".pth", ".onnx", ".pkl")) and model_path is None:
                model_path = str(target_path)
            if "preprocessor" in lower_name:
                preprocessor_path = str(target_path)

        metrics_blob = output_dir / "metrics.json"
        metrics_payload: dict[str, Any] = {}
        if metrics_blob.exists():
            try:
                metrics_payload = json.loads(metrics_blob.read_text(encoding="utf-8"))
            except Exception:
                metrics_payload = {}

        metadata_path.write_text(json.dumps({
            "job_id": job_id,
            "backend": execution.backend,
            "status": execution.status,
            "logs": execution.logs[-5000:],
            "metrics": execution.metrics,
            "artifact_summary": sorted(execution.artifacts),
            "bundle_summary": bundle.summary,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        primary_metric_name = _pick_primary_metric_name(objective_spec)
        best_score = execution.metrics.get(primary_metric_name)
        if best_score is None:
            for name in ("accuracy", "f1", "loss"):
                if name in execution.metrics:
                    best_score = execution.metrics[name]
                    break

        if progress_callback:
            progress_callback({
                "step": "agentic_log",
                "agent_stage": "completed" if execution.status == "completed" else "failed",
                "message": "远端训练执行完成" if execution.status == "completed" else f"远端训练失败: {execution.error_message}",
                "log_entry": execution.logs[-300:] if execution.logs else "",
            })

        return TrainingResult(
            job_id=job_id,
            status=execution.status,
            best_model_name=f"planned_{plan.get('modality', dataset_spec.data_type)}",
            best_metric_score=best_score,
            final_model_path=model_path,
            preprocessor_path=preprocessor_path,
            train_metrics=metrics_payload.get("history", [{}])[-1] if isinstance(metrics_payload.get("history"), list) and metrics_payload.get("history") else execution.metrics,
            val_metrics=metrics_payload.get("test_metrics", execution.metrics) if isinstance(metrics_payload, dict) else execution.metrics,
            training_duration=execution.elapsed_seconds,
            error_message=execution.error_message,
        )


def _resolve_scan_root(dataset_spec: DataSpec) -> Path:
    root = Path(dataset_spec.storage_path)
    extracted = root / "extracted"
    return extracted if extracted.exists() else root


def _pick_primary_metric_name(objective_spec: ObjectiveSpec) -> str:
    value = getattr(objective_spec.primary_metric, "value", str(objective_spec.primary_metric))
    if value in {"accuracy", "f1", "precision", "recall", "loss"}:
        return value
    return "accuracy"


def _stage_directory_bytes(root: Path, max_files: int = 4000) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    if not root.exists():
        return files

    count = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_path = str(Path("dataset") / path.relative_to(root))
        files[rel_path] = path.read_bytes()
        count += 1
        if count >= max_files:
            break
    return files
