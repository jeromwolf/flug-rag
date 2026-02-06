#!/usr/bin/env python3
"""CLI runner for flux-rag load tests.

Usage:
    python -m tests.load.run_load_test --scenario normal_load --host http://localhost:8000
    python -m tests.load.run_load_test --scenario stress_test --headless --html-report report.html
    python -m tests.load.run_load_test --list
"""

import argparse
import sys
import time
from pathlib import Path

from tests.load.scenarios import SCENARIOS, get_scenario, list_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="flux-rag Load Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal_load",
        help=f"Test scenario name. Available: {', '.join(SCENARIOS.keys())}",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:8000",
        help="Target host URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the Locust web UI",
    )
    parser.add_argument(
        "--html-report",
        type=str,
        default=None,
        help="Path for HTML report output",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_scenarios",
        help="List all available scenarios and exit",
    )
    return parser.parse_args()


def print_scenarios() -> None:
    """Print all available scenarios."""
    scenarios = list_scenarios()
    print("\n--- Available Load Test Scenarios ---\n")
    for s in scenarios:
        print(f"  {s['name']:<20} | {s['users']:>3} users | {s['duration_seconds']:>4}s | p95 < {s['max_response_time_p95']:.0f}ms")
        print(f"  {'':20}   {s['description']}")
        print()


def run_locust(scenario_name: str, host: str, headless: bool, html_report: str | None) -> bool:
    """Run a Locust load test programmatically.

    Returns True if the test passed thresholds, False otherwise.
    """
    try:
        import locust.env
        import locust.stats
        from locust import events as locust_events
    except ImportError:
        print("ERROR: locust is not installed. Install with: pip install locust")
        return False

    scenario = get_scenario(scenario_name)
    locust_dir = Path(__file__).resolve().parent
    locustfile_path = locust_dir / scenario.locustfile

    if not locustfile_path.exists():
        print(f"ERROR: Locustfile not found: {locustfile_path}")
        return False

    print(f"\n{'=' * 60}")
    print(f"  flux-rag Load Test: {scenario.name}")
    print(f"{'=' * 60}")
    print(f"  Host:           {host}")
    print(f"  Users:          {scenario.users}")
    print(f"  Spawn rate:     {scenario.spawn_rate}/s")
    print(f"  Duration:       {scenario.duration_seconds}s")
    print(f"  Expected RPS:   >= {scenario.expected_rps}")
    print(f"  Max p95:        <= {scenario.max_response_time_p95}ms")
    print(f"  Max failure:    <= {scenario.max_failure_rate:.1%}")
    print(f"  Locustfile:     {scenario.locustfile}")
    print(f"{'=' * 60}\n")

    # Import the locustfile module dynamically
    import importlib.util

    spec = importlib.util.spec_from_file_location("locustfile_module", str(locustfile_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find user classes
    from locust import HttpUser as LocustHttpUser

    user_classes = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, LocustHttpUser) and attr is not LocustHttpUser:
            if scenario.user_classes is None or attr_name in scenario.user_classes:
                user_classes.append(attr)

    if not user_classes:
        print("ERROR: No user classes found in locustfile")
        return False

    print(f"User classes: {[c.__name__ for c in user_classes]}")

    # Create Locust environment
    env = locust.env.Environment(
        user_classes=user_classes,
        host=host,
    )

    if html_report:
        env.create_local_runner()
        locust.stats.stats_printer = lambda _: None

    # Start the test
    env.create_local_runner()

    if headless:
        print(f"\nStarting headless test for {scenario.duration_seconds}s...")
        env.runner.start(scenario.users, spawn_rate=scenario.spawn_rate)

        # Wait for duration
        start = time.time()
        try:
            while time.time() - start < scenario.duration_seconds:
                time.sleep(1)
                elapsed = int(time.time() - start)
                if elapsed % 30 == 0:  # Print status every 30 seconds
                    stats = env.runner.stats.total
                    print(
                        f"  [{elapsed}s] requests={stats.num_requests} "
                        f"failures={stats.num_failures} "
                        f"rps={stats.current_rps:.1f} "
                        f"avg_rt={stats.avg_response_time:.0f}ms"
                    )
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Stopping test...")

        env.runner.quit()

        # Generate report
        return _evaluate_results(env, scenario, html_report)
    else:
        # Web UI mode
        print(f"\nStarting Locust web UI... Open http://localhost:8089")
        env.create_web_ui("0.0.0.0", 8089)
        env.runner.start(scenario.users, spawn_rate=scenario.spawn_rate)
        try:
            env.runner.greenlet.join()
        except KeyboardInterrupt:
            pass
        return True


def _evaluate_results(env, scenario, html_report: str | None) -> bool:
    """Evaluate test results against scenario thresholds."""
    stats = env.runner.stats.total

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {scenario.name}")
    print(f"{'=' * 60}")

    total_requests = stats.num_requests
    total_failures = stats.num_failures
    failure_rate = stats.fail_ratio if total_requests > 0 else 0
    avg_rt = stats.avg_response_time
    p95_rt = stats.get_response_time_percentile(0.95) if total_requests > 0 else 0
    p99_rt = stats.get_response_time_percentile(0.99) if total_requests > 0 else 0
    rps = total_requests / scenario.duration_seconds if scenario.duration_seconds > 0 else 0

    print(f"  Total requests:    {total_requests}")
    print(f"  Total failures:    {total_failures}")
    print(f"  Failure rate:      {failure_rate:.2%}")
    print(f"  Avg response time: {avg_rt:.0f}ms")
    print(f"  p95 response time: {p95_rt:.0f}ms")
    print(f"  p99 response time: {p99_rt:.0f}ms")
    print(f"  Requests/sec:      {rps:.1f}")

    # Evaluate pass/fail
    passed = True
    checks = []

    if p95_rt > scenario.max_response_time_p95:
        checks.append(f"  FAIL: p95 ({p95_rt:.0f}ms) > threshold ({scenario.max_response_time_p95:.0f}ms)")
        passed = False
    else:
        checks.append(f"  PASS: p95 ({p95_rt:.0f}ms) <= threshold ({scenario.max_response_time_p95:.0f}ms)")

    if failure_rate > scenario.max_failure_rate:
        checks.append(f"  FAIL: failure rate ({failure_rate:.2%}) > threshold ({scenario.max_failure_rate:.2%})")
        passed = False
    else:
        checks.append(f"  PASS: failure rate ({failure_rate:.2%}) <= threshold ({scenario.max_failure_rate:.2%})")

    if rps < scenario.expected_rps and total_requests > 0:
        checks.append(f"  WARN: RPS ({rps:.1f}) < expected ({scenario.expected_rps:.1f})")
    else:
        checks.append(f"  PASS: RPS ({rps:.1f}) >= expected ({scenario.expected_rps:.1f})")

    print(f"\n  --- Threshold Checks ---")
    for check in checks:
        print(check)

    result = "PASSED" if passed else "FAILED"
    print(f"\n  Overall: {result}")
    print(f"{'=' * 60}\n")

    # Generate HTML report if requested
    if html_report:
        try:
            _generate_html_report(env, scenario, html_report, passed)
            print(f"  HTML report saved to: {html_report}")
        except Exception as e:
            print(f"  Warning: Could not generate HTML report: {e}")

    return passed


def _generate_html_report(env, scenario, output_path: str, passed: bool) -> None:
    """Generate a simple HTML report."""
    stats = env.runner.stats

    rows = []
    for entry in stats.entries.values():
        rows.append(
            f"<tr><td>{entry.method}</td><td>{entry.name}</td>"
            f"<td>{entry.num_requests}</td><td>{entry.num_failures}</td>"
            f"<td>{entry.avg_response_time:.0f}</td>"
            f"<td>{entry.get_response_time_percentile(0.95):.0f}</td></tr>"
        )

    result_class = "pass" if passed else "fail"
    result_text = "PASSED" if passed else "FAILED"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>flux-rag Load Test: {scenario.name}</title>
    <style>
        body {{ font-family: sans-serif; margin: 2em; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .pass {{ color: green; font-weight: bold; }}
        .fail {{ color: red; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>flux-rag Load Test Report</h1>
    <h2>Scenario: {scenario.name}</h2>
    <p>{scenario.description}</p>
    <p>Result: <span class="{result_class}">{result_text}</span></p>
    <h3>Endpoint Statistics</h3>
    <table>
        <tr><th>Method</th><th>Endpoint</th><th>Requests</th><th>Failures</th><th>Avg (ms)</th><th>p95 (ms)</th></tr>
        {''.join(rows)}
    </table>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")


def main():
    args = parse_args()

    if args.list_scenarios:
        print_scenarios()
        sys.exit(0)

    try:
        passed = run_locust(
            scenario_name=args.scenario,
            host=args.host,
            headless=args.headless,
            html_report=args.html_report,
        )
        sys.exit(0 if passed else 1)
    except KeyError as e:
        print(f"ERROR: {e}")
        print_scenarios()
        sys.exit(2)


if __name__ == "__main__":
    main()
