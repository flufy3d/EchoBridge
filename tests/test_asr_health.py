from __future__ import annotations

from echobridge.asr import AsrHealth


def test_asr_health_transitions_and_metrics():
    health = AsrHealth()
    assert health.available is True
    assert health.unavailable_total == 0

    health.mark_unavailable("ConnectError")
    assert health.available is False
    assert health.unavailable_total == 1
    assert health.last_error == "ConnectError"

    health.record_retry()
    health.record_retry_success()
    health.mark_ok()
    assert health.available is True
    assert health.last_ok > 0
    assert health.last_error == ""
    assert health.retry_total == 1
    assert health.retry_success_total == 1

    snapshot = health.snapshot()
    assert snapshot["available"] is True
    assert snapshot["retry_total"] == 1

    lines = "\n".join(health.metrics_lines())
    assert "echobridge_asr_available 1" in lines
    assert "echobridge_asr_unavailable_total 1" in lines
    assert "echobridge_asr_retry_total 1" in lines
    assert "echobridge_asr_retry_success_total 1" in lines
    assert "echobridge_asr_last_ok_timestamp_seconds" in lines


def test_asr_health_non_retryable_error_keeps_available():
    health = AsrHealth()
    health.mark_error("HTTP 400")
    assert health.available is True
    assert health.error_total == 1
    assert health.last_error == "HTTP 400"
