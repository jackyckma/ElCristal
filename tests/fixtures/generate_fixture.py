"""Generate a 5-second mono WAV test fixture using only stdlib + numpy."""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "test_5s_mono_44100.wav"

SR = 44_100
DURATION_S = 5
FREQUENCY_HZ = 440.0  # A4
AMPLITUDE = 0.3


def generate() -> Path:
    n_samples = SR * DURATION_S
    try:
        import numpy as np  # noqa: PLC0415

        t = np.linspace(0, DURATION_S, n_samples, endpoint=False, dtype=np.float32)
        signal = (AMPLITUDE * np.sin(2 * math.pi * FREQUENCY_HZ * t) * 32767).astype(np.int16)
        raw = signal.tobytes()
    except ImportError:
        # Stdlib-only fallback
        raw = bytearray()
        for i in range(n_samples):
            t = i / SR
            sample = int(AMPLITUDE * math.sin(2 * math.pi * FREQUENCY_HZ * t) * 32767)
            raw += struct.pack("<h", max(-32768, min(32767, sample)))

    with wave.open(str(FIXTURE_PATH), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SR)
        wf.writeframes(raw)

    print(f"Generated fixture: {FIXTURE_PATH} ({FIXTURE_PATH.stat().st_size // 1024} KB)")
    return FIXTURE_PATH


if __name__ == "__main__":
    generate()
