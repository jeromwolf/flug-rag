"""Stress test scenarios for flux-rag API.

Run with:
    locust -f tests/load/locustfile_stress.py --host http://localhost:8000
"""

import json
import random
import uuid
from pathlib import Path

from locust import HttpUser, constant, task

# Load golden QA questions
_GOLDEN_QA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "golden_qa.json"
_QUESTIONS: list[dict] = []

try:
    with open(_GOLDEN_QA_PATH, "r", encoding="utf-8") as f:
        _QUESTIONS = json.load(f)
except FileNotFoundError:
    _QUESTIONS = [
        {"question": "안전관리 총괄책임자는 누구인가요?"},
        {"question": "일일 점검은 하루에 몇 회 실시하나요?"},
        {"question": "설비 점검 절차는 어떤 단계로 구성되어 있나요?"},
    ]

_SAMPLE_TEXT = "스트레스 테스트 문서입니다. 가스 안전 관리 규정 관련 내용.\n" * 100


class StressUser(HttpUser):
    """Rapid-fire chat messages with no wait time between requests.

    Tests maximum throughput and how the system handles sustained high load.
    """

    wait_time = constant(0)  # No wait between requests

    def on_start(self):
        self.session_id = None

    @task
    def rapid_chat(self):
        """Send chat messages as fast as possible."""
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
            name="/api/chat [stress]",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                self.session_id = data.get("session_id", self.session_id)
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")


class ConcurrentUploadUser(HttpUser):
    """Parallel document upload stress test.

    Tests file I/O, ingestion pipeline, and vectorstore under upload pressure.
    """

    wait_time = constant(0)

    @task
    def upload_document(self):
        """Upload documents as fast as possible."""
        filename = f"stress_{uuid.uuid4().hex[:8]}.txt"
        files = {"file": (filename, _SAMPLE_TEXT.encode("utf-8"), "text/plain")}

        with self.client.post(
            "/api/documents/upload",
            files=files,
            catch_response=True,
            name="/api/documents/upload [stress]",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")


class LongConversationUser(HttpUser):
    """Multi-turn conversation stress test.

    Simulates users having extended conversations with history accumulation.
    Tests memory usage growth and session management under load.
    """

    wait_time = constant(0.5)  # Slight delay to simulate typing

    def on_start(self):
        self.session_id = None
        self.turn_count = 0
        self.max_turns = random.randint(10, 30)

    @task
    def conversation_turn(self):
        """Send the next turn in a long conversation."""
        if self.turn_count >= self.max_turns:
            # Start a new conversation
            self.session_id = None
            self.turn_count = 0
            self.max_turns = random.randint(10, 30)

        qa = random.choice(_QUESTIONS)
        # Add turn context to make questions unique
        message = f"[턴 {self.turn_count + 1}] {qa['question']}"

        payload = {
            "message": message,
            "session_id": self.session_id,
            "mode": "rag",
        }

        with self.client.post(
            "/api/chat",
            json=payload,
            catch_response=True,
            name="/api/chat [long-conv]",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                self.session_id = data.get("session_id", self.session_id)
                self.turn_count += 1
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(1)
    def check_history(self):
        """Periodically check session history size."""
        if self.session_id:
            with self.client.get(
                f"/api/sessions/{self.session_id}/messages",
                catch_response=True,
                name="/api/sessions/{id}/messages [long-conv]",
            ) as resp:
                if resp.status_code in (200, 404):
                    resp.success()
                else:
                    resp.failure(f"Status {resp.status_code}")
