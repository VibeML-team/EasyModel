"""
测试中间 Checkpoint 下载功能
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_list_checkpoints():
    """测试获取 checkpoint 列表"""
    # 先启动一个训练
    response = client.post("/v2/compile", json={
        "intent": "预测用户是否会流失",
        "domain": "general"
    })
    session_id = response.json()["session_id"]
    
    response = client.post("/v2/train/start", json={
        "session_id": session_id
    })
    job_id = response.json()["job_id"]
    
    # 获取 checkpoint 列表
    response = client.get(f"/v2/train/checkpoints/{job_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert "checkpoints" in data
    assert "count" in data
    assert data["job_id"] == job_id


def test_pause_and_download():
    """测试暂停并获取 checkpoint 信息"""
    # 先启动一个训练
    response = client.post("/v2/compile", json={
        "intent": "预测用户是否会流失",
        "domain": "general"
    })
    session_id = response.json()["session_id"]
    
    response = client.post("/v2/train/start", json={
        "session_id": session_id
    })
    job_id = response.json()["job_id"]
    
    # 暂停并获取 checkpoint
    response = client.post(f"/v2/train/pause-and-download/{job_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert "checkpoints" in data
    assert "message" in data


def test_pause_resume_with_checkpoint():
    """测试暂停恢复流程中包含 checkpoint 下载"""
    # 编译并启动训练
    response = client.post("/v2/compile", json={
        "intent": "预测用户购买意向",
        "domain": "general"
    })
    session_id = response.json()["session_id"]
    
    response = client.post("/v2/train/start", json={
        "session_id": session_id
    })
    job_id = response.json()["job_id"]
    
    # 暂停（可能训练还未开始运行，但 API 应该返回成功）
    response = client.post(f"/v2/train/pause/{job_id}")
    assert response.status_code == 200
    # 不强制要求 success，因为训练可能还没开始
    
    # 获取 checkpoint 列表（即使为空也应该成功）
    response = client.get(f"/v2/train/checkpoints/{job_id}")
    assert response.status_code == 200
    assert "checkpoints" in response.json()
    
    # 恢复
    response = client.post(f"/v2/train/resume/{job_id}")
    assert response.status_code == 200
    # 不强制要求 success


def test_checkpoint_info_structure():
    """测试 checkpoint 信息结构"""
    from backend.v2_training_controller import CheckpointInfo
    
    info = CheckpointInfo(
        checkpoint_id="ckpt_1",
        step=10,
        metric=0.85,
        metric_name="f1_score",
        status="available"
    )
    
    data = info.to_dict()
    assert data["id"] == "ckpt_1"
    assert data["step"] == 10
    assert data["metric"] == 0.85
    assert data["metric_name"] == "f1_score"
    assert data["status"] == "available"
