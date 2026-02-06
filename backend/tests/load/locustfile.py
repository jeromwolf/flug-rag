"""Main Locust load test for flux-rag API.

Run with:
    locust -f tests/load/locustfile.py --host http://localhost:8000
"""

import json
import random
import uuid
from pathlib import Path

from locust import HttpUser, between, events, task

# Load golden QA questions for realistic chat requests
_GOLDEN_QA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "golden_qa.json"
_QUESTIONS: list[dict] = []

try:
    with open(_GOLDEN_QA_PATH, "r", encoding="utf-8") as f:
        _QUESTIONS = json.load(f)
except FileNotFoundError:
    # Fallback questions if golden_qa.json is not available
    _QUESTIONS = [
        {"question": "안전관리 총괄책임자는 누구인가요?"},
        {"question": "일일 점검은 하루에 몇 회 실시하나요?"},
        {"question": "비상대피 시 1차 집결지는 어디인가요?"},
        {"question": "설비 점검 절차는 어떤 단계로 구성되어 있나요?"},
        {"question": "가스 누출 심각도 기준은 어떻게 분류되나요?"},
    ]

# Sample text file for upload tests
_SAMPLE_TEXT = "테스트 문서입니다. 가스 안전 관리 규정에 대한 내용입니다.\n" * 50


class ChatUser(HttpUser):
    """Simulates a typical user interacting with the flux-rag platform."""

    wait_time = between(1, 5)

    def on_start(self):
        """Initialize user session."""
        self.session_id = None
        self.message_ids = []

        # Health check
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure("Health check failed")

    @task(5)
    def send_chat_message(self):
        """Send a chat message using a random golden QA question."""
        qa = random.choice(_QUESTIONS)
        payload = {
            "message": qa["question"],
            "session_id": self.session_id,
            "mode": "rag",
        }

        with self.client.post(
            "/api/chat",
            json=payload,
            catch_response=True,
            name="/api/chat",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                self.session_id = data.get("session_id", self.session_id)
                msg_id = data.get("message_id")
                if msg_id:
                    self.message_ids.append(msg_id)
                resp.success()
            else:
                resp.failure(f"Chat failed: {resp.status_code}")

    @task(1)
    def upload_document(self):
        """Upload a small test document."""
        filename = f"test_{uuid.uuid4().hex[:8]}.txt"
        files = {"file": (filename, _SAMPLE_TEXT.encode("utf-8"), "text/plain")}

        with self.client.post(
            "/api/documents/upload",
            files=files,
            catch_response=True,
            name="/api/documents/upload",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Upload failed: {resp.status_code}")

    @task(3)
    def list_documents(self):
        """List all documents."""
        with self.client.get(
            "/api/documents",
            catch_response=True,
            name="/api/documents",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"List docs failed: {resp.status_code}")

    @task(1)
    def get_admin_info(self):
        """Get admin system info."""
        with self.client.get(
            "/api/admin/info",
            catch_response=True,
            name="/api/admin/info",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Admin info failed: {resp.status_code}")

    @task(2)
    def submit_feedback(self):
        """Submit feedback for a previous message."""
        if not self.message_ids:
            return

        payload = {
            "message_id": random.choice(self.message_ids),
            "session_id": self.session_id or str(uuid.uuid4()),
            "rating": random.choice([-1, 0, 1]),
            "comment": "부하 테스트 피드백",
        }

        with self.client.post(
            "/api/feedback",
            json=payload,
            catch_response=True,
            name="/api/feedback",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Feedback failed: {resp.status_code}")


# --- Custom event listeners for metrics ---

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log request metrics for analysis."""
    if exception:
        print(f"[FAIL] {request_type} {name} - {response_time:.0f}ms - {exception}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print(f"[LOAD TEST] Starting against {environment.host}")
    print(f"[LOAD TEST] {len(_QUESTIONS)} golden QA questions loaded")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test ends."""
    print("[LOAD TEST] Test completed")
    stats = environment.runner.stats
    print(f"[LOAD TEST] Total requests: {stats.total.num_requests}")
    print(f"[LOAD TEST] Failure rate: {stats.total.fail_ratio:.2%}")
    if stats.total.num_requests > 0:
        print(f"[LOAD TEST] Avg response time: {stats.total.avg_response_time:.0f}ms")
        print(f"[LOAD TEST] p95 response time: {stats.total.get_response_time_percentile(0.95):.0f}ms")
