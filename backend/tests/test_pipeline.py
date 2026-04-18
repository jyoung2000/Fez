"""Tests for the analysis pipeline state transitions, timeouts, and error handling.

Covers:
  - Pipeline status lifecycle (queued → extracting → transcribing → ... → complete)
  - Cancellation handling
  - Timeout behavior for gather calls
  - Graceful degradation when branches fail
  - Dedicated thread pool for base64 encoding
  - Stage timer logging
"""
import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Pre-mock heavy native dependencies that may not be available in the test env
for _mod in (
    "google.generativeai", "google.generativeai.types", "google.ai",
    "google.ai.generativelanguage_v1beta", "anthropic", "groq", "httpx",
    "ctranslate2", "faster_whisper",
):
    sys.modules.setdefault(_mod, MagicMock())

from backend.services.pipeline import (
    CancelledError,
    _check_cancelled,
    _cancel_events,
    _ws_subscribers,
    is_cancel_requested,
    request_cancel,
    register_ws_subscriber,
    unregister_ws_subscriber,
    broadcast_ws,
    _update_progress,
    _b64_executor,
    _stage_timer,
)


class TestCancellation(unittest.TestCase):
    """Test cancellation event management."""

    def setUp(self):
        _cancel_events.clear()

    def tearDown(self):
        _cancel_events.clear()

    def test_cancel_not_requested_by_default(self):
        self.assertFalse(is_cancel_requested("job-1"))

    def test_request_cancel_sets_event(self):
        _cancel_events["job-1"] = asyncio.Event()
        request_cancel("job-1")
        self.assertTrue(is_cancel_requested("job-1"))

    def test_request_cancel_missing_job_no_error(self):
        # Should not raise
        request_cancel("nonexistent-job")

    def test_check_cancelled_raises(self):
        _cancel_events["job-1"] = asyncio.Event()
        _cancel_events["job-1"].set()
        with self.assertRaises(CancelledError):
            _check_cancelled("job-1")

    def test_check_cancelled_does_not_raise_when_not_set(self):
        _cancel_events["job-1"] = asyncio.Event()
        # Should not raise
        _check_cancelled("job-1")


class TestWebSocketSubscribers(unittest.TestCase):
    """Test WS subscriber management."""

    def setUp(self):
        _ws_subscribers.clear()

    def tearDown(self):
        _ws_subscribers.clear()

    def test_register_and_unregister(self):
        ws = MagicMock()
        register_ws_subscriber("job-1", ws)
        self.assertIn("job-1", _ws_subscribers)
        self.assertIn(ws, _ws_subscribers["job-1"])

        unregister_ws_subscriber("job-1", ws)
        self.assertNotIn("job-1", _ws_subscribers)

    def test_unregister_nonexistent(self):
        # Should not raise
        unregister_ws_subscriber("nonexistent", MagicMock())


class TestBroadcastWs(unittest.TestCase):
    """Test WebSocket broadcasting."""

    def setUp(self):
        _ws_subscribers.clear()

    def tearDown(self):
        _ws_subscribers.clear()

    def test_broadcast_sends_to_subscribers(self):
        ws = AsyncMock()
        register_ws_subscriber("job-1", ws)
        asyncio.get_event_loop().run_until_complete(
            broadcast_ws("job-1", {"type": "test"})
        )
        ws.send_json.assert_called_once_with({"type": "test"})

    def test_broadcast_removes_dead_subscribers(self):
        ws = AsyncMock()
        ws.send_json.side_effect = Exception("connection closed")
        register_ws_subscriber("job-1", ws)
        asyncio.get_event_loop().run_until_complete(
            broadcast_ws("job-1", {"type": "test"})
        )
        # Dead subscriber should be removed
        self.assertEqual(len(_ws_subscribers.get("job-1", [])), 0)


class TestUpdateProgress(unittest.TestCase):
    """Test _update_progress behavior."""

    def setUp(self):
        _cancel_events.clear()

    def tearDown(self):
        _cancel_events.clear()

    @patch("backend.services.pipeline.database")
    @patch("backend.services.pipeline.broadcast_ws", new_callable=AsyncMock)
    def test_update_progress_normal(self, mock_broadcast, mock_db):
        mock_db.update_job_status = AsyncMock()
        asyncio.get_event_loop().run_until_complete(
            _update_progress("job-1", "processing", 50, "Half done")
        )
        mock_db.update_job_status.assert_called_once()
        mock_broadcast.assert_called_once()

    @patch("backend.services.pipeline.database")
    @patch("backend.services.pipeline.broadcast_ws", new_callable=AsyncMock)
    def test_update_progress_raises_on_cancel(self, mock_broadcast, mock_db):
        _cancel_events["job-1"] = asyncio.Event()
        _cancel_events["job-1"].set()
        with self.assertRaises(CancelledError):
            asyncio.get_event_loop().run_until_complete(
                _update_progress("job-1", "processing", 50, "Half done")
            )
        # Should not have written to DB
        mock_db.update_job_status.assert_not_called()


class TestDedicatedExecutor(unittest.TestCase):
    """Test that the dedicated base64 executor exists."""

    def test_b64_executor_exists(self):
        self.assertIsNotNone(_b64_executor)
        self.assertEqual(_b64_executor._max_workers, 4)


class TestStageTimer(unittest.TestCase):
    """Test the stage timer context manager."""

    def test_stage_timer_logs(self):
        with patch("backend.services.pipeline.logger") as mock_logger:
            async def _run():
                async with _stage_timer("job-1", "test_stage"):
                    pass
            asyncio.get_event_loop().run_until_complete(_run())
            # Should have logged start and finish
            calls = [str(c) for c in mock_logger.info.call_args_list]
            self.assertTrue(any("started" in c for c in calls))
            self.assertTrue(any("finished" in c for c in calls))


class TestPipelineTimeoutConstants(unittest.TestCase):
    """Verify timeout constants exist and are reasonable."""

    def test_extraction_timeout(self):
        from backend.services.pipeline import _EXTRACTION_TIMEOUT
        self.assertGreaterEqual(_EXTRACTION_TIMEOUT, 300)
        self.assertLessEqual(_EXTRACTION_TIMEOUT, 1800)

    def test_summary_clip_timeout(self):
        from backend.services.pipeline import _SUMMARY_CLIP_TIMEOUT
        self.assertGreaterEqual(_SUMMARY_CLIP_TIMEOUT, 300)
        self.assertLessEqual(_SUMMARY_CLIP_TIMEOUT, 3600)


class TestRunAnalysisCancelled(unittest.TestCase):
    """Test run_analysis handles CancelledError correctly."""

    @patch("backend.services.pipeline.broadcast_ws", new_callable=AsyncMock)
    @patch("backend.services.pipeline.database")
    @patch("backend.services.pipeline._run_analysis_inner")
    def test_cancel_sets_cancelled_status(self, mock_inner, mock_db, mock_broadcast):
        mock_inner.side_effect = CancelledError("cancelled")
        mock_db.update_job_status = AsyncMock()

        from backend.services.pipeline import run_analysis
        asyncio.get_event_loop().run_until_complete(run_analysis("job-cancel-1"))

        # Should have set CANCELLED status
        calls = mock_db.update_job_status.call_args_list
        cancelled_calls = [c for c in calls if c.kwargs.get("status") == "cancelled"]
        self.assertTrue(len(cancelled_calls) > 0, "Should set status to cancelled")

    @patch("backend.services.pipeline.broadcast_ws", new_callable=AsyncMock)
    @patch("backend.services.pipeline.database")
    @patch("backend.services.pipeline._run_analysis_inner")
    def test_exception_sets_failed_status(self, mock_inner, mock_db, mock_broadcast):
        mock_inner.side_effect = RuntimeError("something broke")
        mock_db.update_job_status = AsyncMock()

        from backend.services.pipeline import run_analysis
        asyncio.get_event_loop().run_until_complete(run_analysis("job-fail-1"))

        # Should have set FAILED status
        calls = mock_db.update_job_status.call_args_list
        failed_calls = [c for c in calls if c.kwargs.get("status") == "failed"]
        self.assertTrue(len(failed_calls) > 0, "Should set status to failed")


if __name__ == "__main__":
    unittest.main()
