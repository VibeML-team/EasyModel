"""分块上传断点续传测试。

主要验证 4 个韧性保证：
1. 同一分块重复上传是幂等的（skipped）。
2. 进程重启（清空 in-memory dict）后，会话能从磁盘 lazy load 回来。
3. 同一 client_token 再次 init 会返回原会话的 received 列表（断点续传）。
4. 全部分块到位后 upload-complete 能正确拼接并触发数据处理。
"""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import httpx
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.main import app, _chunk_uploads  # type: ignore[attr-defined]


class SyncASGITestClient:
    def __init__(self, target_app):
        self.app = target_app
        self.base_url = "http://testserver"

    async def _request(self, method: str, url: str, **kwargs):
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url=self.base_url) as client:
            return await client.request(method, url, **kwargs)

    def request(self, method: str, url: str, **kwargs):
        return asyncio.run(self._request(method, url, **kwargs))

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


@pytest.fixture
def client():
    return SyncASGITestClient(app)


def _build_csv_bytes(rows: int = 80) -> bytes:
    df = pd.DataFrame({
        "user_id": range(rows),
        "age": [25, 30, 35, 40, 45] * (rows // 5),
        "income": [50000, 60000, 75000, 90000, 100000] * (rows // 5),
        "churn": [0, 0, 1, 0, 1] * (rows // 5),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _split_into_chunks(payload: bytes, chunk_size: int) -> list[bytes]:
    return [payload[i:i + chunk_size] for i in range(0, len(payload), chunk_size)]


def test_chunked_upload_idempotent_resume_and_restart(client):
    """覆盖：幂等重传 + 模拟容器重启（清空 in-memory）+ 同 token 续传 + 完成拼接。"""
    payload = _build_csv_bytes(rows=120)
    chunk_size = 256  # 故意小，便于切多块
    chunks = _split_into_chunks(payload, chunk_size)
    total_chunks = len(chunks)
    assert total_chunks >= 4

    # 1) init —— 带 client_token
    init = client.post(
        "/api/data/upload-init",
        json={
            "filename": "resume_test.csv",
            "total_size": len(payload),
            "total_chunks": total_chunks,
            "client_token": "fingerprint-resume-test-1",
        },
    )
    assert init.status_code == 200, init.text
    upload_id = init.json()["upload_id"]
    dataset_id = init.json()["dataset_id"]
    assert init.json()["resumed"] is False
    assert init.json()["received"] == []

    # 2) 上传前 3 块
    for i in range(3):
        r = client.post(
            "/api/data/upload-chunk",
            data={"upload_id": upload_id, "chunk_index": str(i)},
            files={"chunk": (f"chunk_{i}", chunks[i], "application/octet-stream")},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["received"] == i
        assert body["skipped"] is False

    # 3) 幂等重传第 1 块 → skipped=True
    r = client.post(
        "/api/data/upload-chunk",
        data={"upload_id": upload_id, "chunk_index": "1"},
        files={"chunk": (f"chunk_1", chunks[1], "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["skipped"] is True

    # 4) 状态查询
    s = client.get(f"/api/data/upload-status?upload_id={upload_id}")
    assert s.status_code == 200, s.text
    assert sorted(s.json()["received"]) == [0, 1, 2]

    # 5) 模拟容器重启：清空 in-memory 字典
    _chunk_uploads.clear()

    # 6) 状态查询应仍能成功（lazy load from disk）
    s2 = client.get(f"/api/data/upload-status?upload_id={upload_id}")
    assert s2.status_code == 200, s2.text
    assert sorted(s2.json()["received"]) == [0, 1, 2]

    # 7) 同 client_token 再次 init —— 应复用旧会话
    reinit = client.post(
        "/api/data/upload-init",
        json={
            "filename": "resume_test.csv",
            "total_size": len(payload),
            "total_chunks": total_chunks,
            "client_token": "fingerprint-resume-test-1",
        },
    )
    assert reinit.status_code == 200
    rj = reinit.json()
    assert rj["resumed"] is True
    assert rj["upload_id"] == upload_id
    assert rj["dataset_id"] == dataset_id
    assert sorted(rj["received"]) == [0, 1, 2]

    # 8) 再清一次内存 + 上传剩余分块
    _chunk_uploads.clear()
    for i in range(3, total_chunks):
        r = client.post(
            "/api/data/upload-chunk",
            data={"upload_id": upload_id, "chunk_index": str(i)},
            files={"chunk": (f"chunk_{i}", chunks[i], "application/octet-stream")},
        )
        assert r.status_code == 200, r.text

    # 9) 完成拼接 —— 应触发数据分析并返回 dataset_id
    done = client.post(
        "/api/data/upload-complete",
        data={"upload_id": upload_id, "target_hint": "churn"},
    )
    assert done.status_code == 200, done.text
    j = done.json()
    assert j["success"] is True
    assert j["dataset_id"] == dataset_id
    assert j["file_size"] == len(payload)


def test_upload_chunk_returns_404_for_unknown_session(client):
    """会话不存在时返回 404（前端据此自动重建会话）。"""
    r = client.post(
        "/api/data/upload-chunk",
        data={"upload_id": "deadbeefdead", "chunk_index": "0"},
        files={"chunk": ("c", b"hello", "application/octet-stream")},
    )
    assert r.status_code == 404


def test_upload_status_returns_404_for_unknown_session(client):
    r = client.get("/api/data/upload-status?upload_id=deadbeefdead")
    assert r.status_code == 404


def test_upload_chunk_rejects_out_of_range_index(client):
    init = client.post(
        "/api/data/upload-init",
        json={"filename": "rng.csv", "total_size": 100, "total_chunks": 2},
    )
    assert init.status_code == 200
    upload_id = init.json()["upload_id"]

    bad = client.post(
        "/api/data/upload-chunk",
        data={"upload_id": upload_id, "chunk_index": "5"},
        files={"chunk": ("c", b"x" * 50, "application/octet-stream")},
    )
    assert bad.status_code == 400


def test_upload_abort_cleans_session(client):
    init = client.post(
        "/api/data/upload-init",
        json={
            "filename": "abort.csv",
            "total_size": 50,
            "total_chunks": 1,
            "client_token": "fingerprint-abort-test",
        },
    )
    upload_id = init.json()["upload_id"]

    abort = client.post(
        "/api/data/upload-abort",
        data={"upload_id": upload_id},
    )
    assert abort.status_code == 200
    assert abort.json()["ok"] is True

    s = client.get(f"/api/data/upload-status?upload_id={upload_id}")
    assert s.status_code == 404
