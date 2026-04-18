"""Tests for the AI orchestrator — circuit breaker and provider fallback.

Covers:
  - CircuitBreaker state transitions (healthy → degraded → recovered)
  - Provider chain construction (active chain skips degraded)
  - Timeout constants existence in each provider
"""
import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Pre-mock heavy native dependencies
for _mod in (
    "google.generativeai", "google.generativeai.types", "google.ai",
    "google.ai.generativelanguage_v1beta", "anthropic", "groq", "httpx",
    "openai",
):
    sys.modules.setdefault(_mod, MagicMock())

from backend.services.ai_orchestrator import _CircuitBreaker


class TestCircuitBreakerInit(unittest.TestCase):
    """Circuit breaker starts clean."""

    def test_initial_state_not_degraded(self):
        cb = _CircuitBreaker()
        self.assertFalse(cb.is_degraded("openrouter"))
        self.assertFalse(cb.is_degraded("anthropic"))
        self.assertFalse(cb.is_degraded("gemini"))

    def test_unknown_provider_not_degraded(self):
        cb = _CircuitBreaker()
        self.assertFalse(cb.is_degraded("nonexistent"))


class TestCircuitBreakerFailures(unittest.TestCase):
    """Test failure tracking and degradation."""

    def test_single_failure_not_degraded(self):
        cb = _CircuitBreaker()
        cb.record_failure("openrouter")
        self.assertFalse(cb.is_degraded("openrouter"))

    def test_two_failures_not_degraded(self):
        cb = _CircuitBreaker()
        cb.record_failure("openrouter")
        cb.record_failure("openrouter")
        self.assertFalse(cb.is_degraded("openrouter"))

    def test_three_failures_triggers_degraded(self):
        cb = _CircuitBreaker()
        cb.record_failure("openrouter")
        cb.record_failure("openrouter")
        cb.record_failure("openrouter")
        self.assertTrue(cb.is_degraded("openrouter"))

    def test_degraded_only_affects_failing_provider(self):
        cb = _CircuitBreaker()
        cb.record_failure("openrouter")
        cb.record_failure("openrouter")
        cb.record_failure("openrouter")
        self.assertTrue(cb.is_degraded("openrouter"))
        self.assertFalse(cb.is_degraded("anthropic"))


class TestCircuitBreakerRecovery(unittest.TestCase):
    """Test recovery from degraded state."""

    def test_success_clears_failures(self):
        cb = _CircuitBreaker()
        cb.record_failure("openrouter")
        cb.record_failure("openrouter")
        cb.record_success("openrouter")
        # Should not be degraded, failure count reset
        self.assertFalse(cb.is_degraded("openrouter"))
        # Another failure should only count as 1
        cb.record_failure("openrouter")
        self.assertFalse(cb.is_degraded("openrouter"))

    def test_success_clears_degraded(self):
        cb = _CircuitBreaker()
        cb.record_failure("openrouter")
        cb.record_failure("openrouter")
        cb.record_failure("openrouter")
        self.assertTrue(cb.is_degraded("openrouter"))
        cb.record_success("openrouter")
        self.assertFalse(cb.is_degraded("openrouter"))

    def test_degraded_expires_after_timeout(self):
        """Degraded state should expire after 15 minutes (900s)."""
        cb = _CircuitBreaker()
        cb.record_failure("openrouter")
        cb.record_failure("openrouter")
        cb.record_failure("openrouter")
        self.assertTrue(cb.is_degraded("openrouter"))
        # Simulate time passing by manipulating the internal state
        cb._degraded_until["openrouter"] = time.monotonic() - 1  # Already expired
        self.assertFalse(cb.is_degraded("openrouter"))


class TestCircuitBreakerWindowExpiry(unittest.TestCase):
    """Test that failures outside the 10-min window are pruned."""

    def test_old_failures_pruned(self):
        cb = _CircuitBreaker()
        # Simulate old failures by manipulating timestamps
        old_time = time.monotonic() - 700  # > 600s ago
        cb._failures["openrouter"] = [old_time, old_time + 1]
        # Record a new failure — old ones should be pruned
        cb.record_failure("openrouter")
        # Only 1 recent failure, so not degraded
        self.assertFalse(cb.is_degraded("openrouter"))


class TestProviderTimeoutConstants(unittest.TestCase):
    """Verify that provider timeout constants exist and are reasonable."""

    def test_openrouter_has_timeout(self):
        from backend.services.providers.openrouter_provider import OpenRouterProvider
        self.assertTrue(hasattr(OpenRouterProvider, '_API_TIMEOUT'))
        self.assertGreaterEqual(OpenRouterProvider._API_TIMEOUT, 60)

    def test_anthropic_has_timeout(self):
        from backend.services.providers.anthropic_provider import _VISION_TIMEOUT, _TEXT_TIMEOUT
        self.assertGreaterEqual(_VISION_TIMEOUT, 60)
        self.assertGreaterEqual(_TEXT_TIMEOUT, 60)

    def test_gemini_has_timeout(self):
        from backend.services.providers.gemini_provider import _VISION_TIMEOUT, _TEXT_TIMEOUT
        self.assertGreaterEqual(_VISION_TIMEOUT, 60)
        self.assertGreaterEqual(_TEXT_TIMEOUT, 60)

    def test_groq_has_timeout(self):
        from backend.services.providers.groq_provider import _API_TIMEOUT
        self.assertGreaterEqual(_API_TIMEOUT, 30)


class TestOrchestratorActiveChain(unittest.TestCase):
    """Test that AIOrchestrator builds an active chain correctly."""

    @patch("backend.services.ai_orchestrator.settings")
    @patch("backend.services.ai_orchestrator._build_provider")
    def test_active_chain_skips_degraded(self, mock_build, mock_settings):
        mock_settings.active_provider_chain = ["openrouter", "gemini"]

        mock_provider_or = MagicMock()
        mock_provider_or.provider_name = "openrouter"
        mock_provider_gem = MagicMock()
        mock_provider_gem.provider_name = "gemini"

        def _build_side_effect(name):
            if name == "openrouter":
                return mock_provider_or
            if name == "gemini":
                return mock_provider_gem
            return None
        mock_build.side_effect = _build_side_effect

        from backend.services.ai_orchestrator import AIOrchestrator
        orch = AIOrchestrator()

        # Both should be in chain initially
        chain = orch._get_active_chain()
        self.assertEqual(len(chain), 2)

        # Degrade openrouter
        orch._circuit_breaker.record_failure("openrouter")
        orch._circuit_breaker.record_failure("openrouter")
        orch._circuit_breaker.record_failure("openrouter")

        # Only gemini should remain
        chain = orch._get_active_chain()
        self.assertEqual(len(chain), 1)
        self.assertEqual(chain[0].provider_name, "gemini")


if __name__ == "__main__":
    unittest.main()
