"""Load test scenario definitions with configurable parameters and thresholds."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LoadScenario:
    """Configuration for a load test scenario.

    Attributes:
        name: Human-readable scenario name.
        description: What this scenario tests.
        users: Total number of simulated users.
        spawn_rate: Users spawned per second.
        duration_seconds: Total test duration in seconds.
        expected_rps: Expected requests per second (minimum).
        max_response_time_p95: Maximum acceptable p95 response time in ms.
        max_failure_rate: Maximum acceptable failure rate (0.0 to 1.0).
        locustfile: Which locustfile to use.
        user_classes: Optional list of specific user classes to run.
    """

    name: str
    description: str
    users: int
    spawn_rate: int
    duration_seconds: int
    expected_rps: float
    max_response_time_p95: float  # milliseconds
    max_failure_rate: float = 0.05  # 5% default
    locustfile: str = "locustfile.py"
    user_classes: Optional[list[str]] = None


# --- Predefined scenarios ---

normal_load = LoadScenario(
    name="normal_load",
    description="Normal operating conditions: 10 concurrent users doing typical operations",
    users=10,
    spawn_rate=2,
    duration_seconds=300,  # 5 minutes
    expected_rps=2.0,
    max_response_time_p95=5000,  # 5 seconds
    max_failure_rate=0.02,
)

peak_load = LoadScenario(
    name="peak_load",
    description="Peak load: 50 concurrent users simulating busy period",
    users=50,
    spawn_rate=10,
    duration_seconds=600,  # 10 minutes
    expected_rps=10.0,
    max_response_time_p95=10000,  # 10 seconds
    max_failure_rate=0.05,
)

stress_test = LoadScenario(
    name="stress_test",
    description="Stress test: 100 users with rapid ramp-up to find breaking point",
    users=100,
    spawn_rate=20,
    duration_seconds=900,  # 15 minutes
    expected_rps=15.0,
    max_response_time_p95=30000,  # 30 seconds
    max_failure_rate=0.10,
    locustfile="locustfile_stress.py",
    user_classes=["StressUser"],
)

endurance_test = LoadScenario(
    name="endurance_test",
    description="Endurance test: moderate load sustained for 1 hour to detect memory leaks",
    users=20,
    spawn_rate=5,
    duration_seconds=3600,  # 60 minutes
    expected_rps=5.0,
    max_response_time_p95=8000,  # 8 seconds
    max_failure_rate=0.03,
)

concurrent_upload_test = LoadScenario(
    name="concurrent_upload",
    description="Concurrent document upload stress test",
    users=20,
    spawn_rate=5,
    duration_seconds=300,  # 5 minutes
    expected_rps=5.0,
    max_response_time_p95=15000,  # 15 seconds
    max_failure_rate=0.05,
    locustfile="locustfile_stress.py",
    user_classes=["ConcurrentUploadUser"],
)

long_conversation_test = LoadScenario(
    name="long_conversation",
    description="Long conversation test: multi-turn sessions to test memory growth",
    users=10,
    spawn_rate=2,
    duration_seconds=600,  # 10 minutes
    expected_rps=3.0,
    max_response_time_p95=10000,  # 10 seconds
    max_failure_rate=0.05,
    locustfile="locustfile_stress.py",
    user_classes=["LongConversationUser"],
)

# Registry of all scenarios
SCENARIOS: dict[str, LoadScenario] = {
    "normal_load": normal_load,
    "peak_load": peak_load,
    "stress_test": stress_test,
    "endurance_test": endurance_test,
    "concurrent_upload": concurrent_upload_test,
    "long_conversation": long_conversation_test,
}


def get_scenario(name: str) -> LoadScenario:
    """Get a scenario by name.

    Raises:
        KeyError: If scenario name is not found.
    """
    if name not in SCENARIOS:
        available = ", ".join(SCENARIOS.keys())
        raise KeyError(f"Unknown scenario: {name}. Available: {available}")
    return SCENARIOS[name]


def list_scenarios() -> list[dict]:
    """List all available scenarios with summary info."""
    return [
        {
            "name": s.name,
            "description": s.description,
            "users": s.users,
            "duration_seconds": s.duration_seconds,
            "expected_rps": s.expected_rps,
            "max_response_time_p95": s.max_response_time_p95,
        }
        for s in SCENARIOS.values()
    ]
