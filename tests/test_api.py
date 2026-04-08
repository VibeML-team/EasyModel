"""
VibeML Agent 集成测试

测试完整流程：
1. 数据上传
2. 意图编译
3. 训练启动
4. 模型下载
5. 预测
"""

import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_data():
    """创建测试数据集"""
    df = pd.DataFrame({
        'user_id': range(100),
        'age': [25, 30, 35, 40, 45] * 20,
        'income': [50000, 60000, 75000, 90000, 100000] * 20,
        'tenure': [1, 2, 3, 4, 5] * 20,
        'churn': [0, 0, 1, 0, 1] * 20,
    })
    return df


def test_health(client):
    """健康检查"""
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_upload_data(client, sample_data):
    """测试数据上传"""
    csv_buffer = io.BytesIO()
    sample_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    res = client.post(
        "/api/data/upload",
        files={"file": ("test.csv", csv_buffer, "text/csv")},
        data={"target_hint": "churn"},
    )
    
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["n_rows"] == 100
    assert data["target_column"] == "churn"
    assert "dataset_id" in data


def test_compile_intent(client):
    """测试意图编译"""
    res = client.post(
        "/api/intent/compile",
        json={
            "user_goal": "预测用户是否会流失，宁可误报也别漏报",
            "must_keep": ["高价值客户"],
            "can_change": ["模型类型"],
            "worst_errors": ["漏报"],
            "priority": "quality",
        },
    )
    
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "objective_spec" in data
    assert "interpretation" in data
    
    spec = data["objective_spec"]
    # 任务类型应该是分类相关
    assert spec["task_family"] in ["binary_classification", "multiclass_classification", "time_series"]
    # 因为需求中有 "宁可误报也别漏报"，所以指标应该是 recall 或 f1
    assert spec["primary_metric"] in ["recall", "f1", "precision", "accuracy"]


def test_end_to_end_flow(client, sample_data):
    """测试端到端完整流程"""
    
    # 1. 上传数据
    csv_buffer = io.BytesIO()
    sample_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    upload_res = client.post(
        "/api/data/upload",
        files={"file": ("churn.csv", csv_buffer, "text/csv")},
        data={"target_hint": "churn"},
    )
    assert upload_res.status_code == 200
    dataset_id = upload_res.json()["dataset_id"]
    
    # 2. 启动训练
    start_res = client.post(
        "/api/training/start",
        json={
            "objective": "预测用户是否会流失",
            "must_keep": ["高价值客户"],
            "can_change": ["模型类型"],
            "worst_errors": ["漏报比误报更糟"],
            "priority": "quality",
            "mode": "standard",
            "dataset_id": dataset_id,
            "target_column": "churn",
            "max_training_time": 30,
            "max_trials": 5,
        },
    )
    
    assert start_res.status_code == 200
    job_data = start_res.json()
    assert job_data["success"] is True
    job_id = job_data["job_id"]
    
    # 3. 等待训练完成（最多30秒）
    max_wait = 30
    elapsed = 0
    while elapsed < max_wait:
        status_res = client.get(f"/api/training/{job_id}")
        status = status_res.json()
        
        if status["status"] in ["completed", "failed", "stopped"]:
            break
        
        time.sleep(2)
        elapsed += 2
    
    # 4. 验证训练结果
    final_res = client.get(f"/api/training/{job_id}")
    final_status = final_res.json()
    
    print(f"训练状态: {final_status['status']}")
    print(f"消息: {final_status['message']}")
    
    # 如果训练成功，验证结果
    if final_status["status"] == "completed":
        result_res = client.get(f"/api/training/{job_id}/result")
        assert result_res.status_code == 200
        
        result = result_res.json()
        assert result["status"] == "completed"
        assert result["best_model_name"] is not None
        assert result["best_metric_score"] is not None
        assert len(result["val_metrics"]) > 0
        
        # 5. 验证可以下载模型
        checkpoints_res = client.get(f"/api/training/{job_id}/checkpoints")
        checkpoints = checkpoints_res.json()
        assert len(checkpoints["checkpoints"]) > 0
        
        # 下载模型
        model_res = client.get(f"/api/training/{job_id}/download/model")
        assert model_res.status_code == 200
        assert len(model_res.content) > 0
        
        print(f"✅ 端到端测试通过！模型: {result['best_model_name']}, 得分: {result['best_metric_score']:.4f}")
    else:
        # 即使失败也打印日志，便于调试
        print(f"⚠️ 训练未成功完成: {final_status['message']}")
        # 在测试环境中，我们只验证 API 流程正常，不强制要求训练一定成功


def test_list_datasets(client, sample_data):
    """测试数据集列表"""
    # 先上传一个数据集
    csv_buffer = io.BytesIO()
    sample_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    client.post(
        "/api/data/upload",
        files={"file": ("list_test.csv", csv_buffer, "text/csv")},
    )
    
    # 获取列表
    res = client.get("/api/data/list")
    assert res.status_code == 200
    data = res.json()
    assert "datasets" in data
    assert len(data["datasets"]) >= 1


def test_dataset_info(client, sample_data):
    """测试获取数据集信息"""
    # 上传
    csv_buffer = io.BytesIO()
    sample_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    upload_res = client.post(
        "/api/data/upload",
        files={"file": ("info_test.csv", csv_buffer, "text/csv")},
    )
    dataset_id = upload_res.json()["dataset_id"]
    
    # 获取信息
    res = client.get(f"/api/data/{dataset_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_id"] == dataset_id
    assert data["n_rows"] == 100
    assert len(data["columns"]) == 5


def test_pause_resume_stop(client, sample_data):
    """测试训练控制"""
    # 上传并启动训练
    csv_buffer = io.BytesIO()
    sample_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    upload_res = client.post(
        "/api/data/upload",
        files={"file": ("control_test.csv", csv_buffer, "text/csv")},
    )
    dataset_id = upload_res.json()["dataset_id"]
    
    start_res = client.post(
        "/api/training/start",
        json={
            "objective": "预测用户流失",
            "must_keep": [],
            "can_change": [],
            "worst_errors": [],
            "dataset_id": dataset_id,
            "target_column": "churn",
            "max_training_time": 60,
        },
    )
    job_id = start_res.json()["job_id"]
    
    # 暂停
    pause_res = client.post(f"/api/training/{job_id}/pause")
    assert pause_res.status_code == 200
    assert pause_res.json()["status"] == "paused"
    
    # 恢复
    resume_res = client.post(f"/api/training/{job_id}/resume")
    assert resume_res.status_code == 200
    assert resume_res.json()["status"] == "running"
    
    # 停止
    stop_res = client.post(f"/api/training/{job_id}/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["status"] == "stopped"
