"""
测试意图澄清流程
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_ambiguity_detection():
    """测试模糊性检测"""
    # 模糊的意图
    response = client.post("/v2/compile", json={
        "intent": "预测",
        "domain": "general"
    })
    assert response.status_code == 200
    data = response.json()
    
    # 验证模糊性检测结果
    clarification = data["intent_card"]["clarification"]
    assert clarification["is_ambiguous"] is True
    assert clarification["ambiguity_score"] > 0.5
    assert len(clarification["suggested_questions"]) > 0
    assert clarification["can_proceed"] is False
    
    # 验证问题格式
    question = clarification["suggested_questions"][0]
    assert "id" in question
    assert "text" in question
    assert "type" in question


def test_clarification_flow():
    """测试完整澄清流程"""
    # 1. 提交模糊意图
    response = client.post("/v2/compile", json={
        "intent": "预测用户",
        "domain": "general"
    })
    data = response.json()
    session_id = data["session_id"]
    
    # 2. 回答澄清问题
    response = client.post("/v2/clarify", json={
        "session_id": session_id,
        "answers": [
            {"question_id": "q_desc_detail", "answer": "预测用户是否会购买商品"},
            {"question_id": "q_error_pref", "answer": "宁可误报"},
        ]
    })
    assert response.status_code == 200
    data = response.json()
    
    # 验证澄清响应
    assert "clarification_complete" in data
    assert "remaining_questions" in data
    assert data["intent_card"]["clarification"]["is_ambiguous"] is False


def test_clear_intent():
    """测试清晰意图被正确识别"""
    response = client.post("/v2/compile", json={
        "intent": "预测用户是否会流失，宁可误报也别漏报",
        "domain": "general"
    })
    assert response.status_code == 200
    data = response.json()
    
    clarification = data["intent_card"]["clarification"]
    # 清晰意图应该有较低的模糊度
    assert clarification["ambiguity_score"] < 1.0


def test_compile_with_data_preview():
    """测试带数据预览的编译"""
    response = client.post("/v2/compile", json={
        "intent": "预测",
        "domain": "general",
        "data_preview": {"total": 1000, "train": 800, "val": 200}
    })
    assert response.status_code == 200
    data = response.json()
    
    # 有数据预览时，应该减少数据相关问题
    clarification = data["intent_card"]["clarification"]
    data_questions = [q for q in clarification["suggested_questions"] 
                      if "数据" in q.get("context", "")]
    # 有数据预览时，数据相关问题应该较少
    assert len(data_questions) == 0


def test_clarification_updates_intent():
    """测试澄清会更新意图卡"""
    # 1. 编译模糊意图
    response = client.post("/v2/compile", json={
        "intent": "预测",
        "domain": "general"
    })
    data = response.json()
    session_id = data["session_id"]
    
    # 2. 回答错误偏好和优先级问题
    response = client.post("/v2/clarify", json={
        "session_id": session_id,
        "answers": [
            {"question_id": "q_desc_detail", "answer": "预测用户购买行为"},
            {"question_id": "q_error_pref", "answer": "宁可误报"},
            {"question_id": "q_priority", "answer": "质量优先"},
        ]
    })
    data = response.json()
    
    # 验证意图卡被更新
    intent_card = data["intent_card"]
    assert intent_card["constraints"]["error_preference"] == "prefer_fp"
    assert intent_card["constraints"]["priority"] == "quality"
    # 回答核心问题后，is_ambiguous 应该变为 False
    assert intent_card["clarification"]["is_ambiguous"] is False
