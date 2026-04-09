import base64
import io
import os
import zipfile
from pathlib import Path

import pandas as pd

from backend.compiler import LabelDefinition, ObjectiveMetric, ObjectiveSpec, TaskFamily
from backend.data_manager import data_manager
from backend.training_router import build_training_plan, execute_training_plan
from backend.agentic_training import TrainingBundleGenerator


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z6uoAAAAASUVORK5CYII="
)


def test_upload_zip_infers_image_classification_structure():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("train/cat/1.png", _PNG_1X1)
        zf.writestr("train/dog/1.png", _PNG_1X1)
        zf.writestr("test/cat/1.png", _PNG_1X1)
        zf.writestr("test/dog/1.png", _PNG_1X1)

    spec = data_manager.upload_file(
        content=buffer.getvalue(),
        filename="images.zip",
    )

    assert spec.data_type == "image_classification"
    assert spec.exploration["statistics"]["splits"]["train"] == 2
    assert sorted(spec.exploration["statistics"]["classes"]) == ["cat", "dog"]


def test_text_agentic_training_executes_with_local_fallback(monkeypatch):
    monkeypatch.setenv("MODAL_ENABLED", "0")
    monkeypatch.setenv("MODAL_ALLOW_LOCAL_FALLBACK", "1")

    df = pd.DataFrame(
        {
            "text": [
                "great product and fast shipping",
                "bad quality and broken item",
                "excellent experience highly recommend",
                "terrible support and refund delay",
                "love it will buy again",
                "worst purchase ever made",
            ],
            "label": ["pos", "neg", "pos", "neg", "pos", "neg"],
        }
    )

    spec = data_manager.upload_file(
        content=df.to_csv(index=False).encode("utf-8"),
        filename="text_reviews.csv",
        target_hint="label",
    )
    assert spec.data_type == "text_classification"

    objective = ObjectiveSpec(
        raw_intent="训练一个文本情感分类模型",
        task_family=TaskFamily.BINARY_CLASSIFICATION,
        label=LabelDefinition(target_column="label", task_family=TaskFamily.BINARY_CLASSIFICATION),
        primary_metric=ObjectiveMetric.F1,
        max_training_time=120,
        max_trials=3,
    )

    plan = build_training_plan(
        objective_spec=objective,
        data_spec=spec,
        config={"dataset_id": spec.dataset_id, "max_training_time": 120},
    )
    result = execute_training_plan(
        job_id="test_text_agentic",
        plan=plan,
        objective_spec=objective,
    )

    assert result.status == "completed"
    assert result.final_model_path
    assert Path(result.final_model_path).exists()
    assert result.best_metric_score is not None


def test_image_training_plan_uses_exploration_as_strong_input():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("train/0/1.png", _PNG_1X1)
        zf.writestr("train/1/1.png", _PNG_1X1)
        zf.writestr("test/0/1.png", _PNG_1X1)
        zf.writestr("test/1/1.png", _PNG_1X1)

    spec = data_manager.upload_file(
        content=buffer.getvalue(),
        filename="mnist_like.zip",
    )
    spec.exploration["business_summary"] = "MNIST 灰度图像分类数据集"
    spec.exploration["data_understanding"]["summary"] = "28x28 grayscale handwritten digits"

    objective = ObjectiveSpec(
        raw_intent="训练一个mnist分类网络",
        task_family=TaskFamily.MULTICLASS_CLASSIFICATION,
        label=LabelDefinition(task_family=TaskFamily.MULTICLASS_CLASSIFICATION),
        primary_metric=ObjectiveMetric.ACCURACY,
        max_trials=3,
    )

    plan = build_training_plan(
        objective_spec=objective,
        data_spec=spec,
        config={"dataset_id": spec.dataset_id},
    )

    training_spec = plan.inputs["training_spec"]
    assert training_spec["input_channels"] == 1
    assert training_spec["color_mode"] == "L"
    assert training_spec["dataset_layout"] == "imagefolder_with_splits"

    bundle = TrainingBundleGenerator().build(plan.to_dict(), spec, objective)
    assert "self.in_channels = 1" in bundle.files["data_pipeline.py"]
    assert 'self.mode = "L"' in bundle.files["data_pipeline.py"]
    assert "num_output_channels=1" in bundle.files["data_pipeline.py"]
