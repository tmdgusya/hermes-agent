"""Clap detection for the /jarvis slash command.

Two layers:

1. ``ClapAnalyzer`` — pure-Python state machine. Given int16 audio chunks
   and a monotonic timestamp, it reports when a double-clap pattern
   (two short impulsive events within ``window_seconds``, separated by at
   least ``cooldown_seconds``) completes. No sounddevice dependency, so
   unit-testable with synthetic numpy arrays.

2. ``ClapDetector.listen()`` — thin sounddevice.InputStream adapter that
   feeds chunks to the analyzer and blocks until a double-clap is
   detected or a timeout expires. Added in Task 2.

Tunables are module constants; override via ``ClapAnalyzer(**kwargs)``
when tuning for a noisier environment.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables — chosen so that:
#   * Speech/ambient RMS (0-2000) does NOT trigger.
#   * A clap's brief peak (15k-30k on int16) clearly exceeds threshold.
#   * Two claps must be >=cooldown apart and <=window apart.
# ---------------------------------------------------------------------------
CLAP_RMS_THRESHOLD = 3000          # int16 RMS; must exceed this for a chunk to count as impulse
CLAP_WINDOW_SECONDS = 3.0          # second clap must arrive within this
CLAP_COOLDOWN_SECONDS = 0.3        # min gap between two claps — also rejects one clap spread over chunks


class ClapAnalyzer:
    """Pure state machine; no hardware I/O.

    States::

        IDLE        -- waiting for first clap
        ARMED       -- saw first clap; waiting for second within window

    ``process_chunk`` advances the machine and returns ``True`` exactly
    once per completed double-clap, then resets to IDLE. If a caller
    keeps feeding chunks after a trigger, subsequent impulses re-arm
    as a fresh first-clap — i.e. three rapid claps trigger once (on
    the 2nd) and leave the 3rd armed. The ``/jarvis`` handler stops
    listening on the first ``True``, so this only matters for
    long-running reuse.
    """

    def __init__(
        self,
        rms_threshold: int = CLAP_RMS_THRESHOLD,
        window_seconds: float = CLAP_WINDOW_SECONDS,
        cooldown_seconds: float = CLAP_COOLDOWN_SECONDS,
    ) -> None:
        self._rms_threshold = int(rms_threshold)
        self._window = float(window_seconds)
        self._cooldown = float(cooldown_seconds)
        self._first_clap_at: Optional[float] = None
        self._last_impulse_at: Optional[float] = None

    def reset(self) -> None:
        """Clear armed state; next impulse becomes a fresh first-clap."""
        self._first_clap_at = None
        self._last_impulse_at = None

    def process_chunk(self, samples: np.ndarray, now: float) -> bool:
        """Feed one audio chunk.

        Args:
            samples: int16 1-D numpy array, typically 50-200ms at 16kHz.
            now: Monotonic timestamp in seconds.

        Returns:
            ``True`` exactly when the double-clap pattern completes on this
            chunk. ``False`` otherwise.
        """
        if samples.size == 0:
            return False

        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

        # Expire an old first-clap outside the window so that a fresh clap
        # starts a brand-new pair.
        if self._first_clap_at is not None and now - self._first_clap_at > self._window:
            logger.debug("clap: window expired; resetting")
            self.reset()

        if rms < self._rms_threshold:
            return False

        # Cooldown rejects the tail of the same clap spilling into the next chunk.
        if self._last_impulse_at is not None and now - self._last_impulse_at < self._cooldown:
            return False

        self._last_impulse_at = now

        if self._first_clap_at is None:
            self._first_clap_at = now
            logger.debug("clap: first impulse armed at %.3f", now)
            return False

        # Second impulse inside window + past cooldown — double-clap complete.
        logger.info("clap: double-clap detected (gap=%.2fs)", now - self._first_clap_at)
        self.reset()
        return True
