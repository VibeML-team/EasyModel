from fastapi.testclient import TestClient

from backend.main import app


def test_health() -> None:
    client = TestClient(app)
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_clarify_and_pause_download_flow() -> None:
    client = TestClient(app)

    clarify = client.post(
        "/api/intent/clarify",
        json={
            "user_goal": "客服工单转成稳定JSON",
            "must_keep": ["JSON结构完整"],
            "can_change": ["措辞"],
            "worst_errors": ["格式错误"],
            "priority": "quality",
        },
    )
    assert clarify.status_code == 200

    start = client.post(
        "/api/training/start",
        json={
            "objective": "客服工单转成稳定JSON",
            "must_keep": ["JSON结构完整"],
            "can_change": ["措辞"],
            "worst_errors": ["格式错误"],
            "priority": "quality",
            "mode": "standard",
            "sample_notes": "A类样本重要",
        },
    )
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    paused = client.post(f"/api/training/{job_id}/pause")
    assert paused.status_code == 200
    filename = paused.json()["checkpoint"]["file"]

    downloaded = client.get(f"/api/training/{job_id}/download/{filename}")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"].startswith("application/json")
