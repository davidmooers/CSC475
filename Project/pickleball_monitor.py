"""
Pickleball Waveform + FFT Monitor (Matplotlib version)
======================================================

No PySide6 or pyqtgraph required.

Expected Arduino packets:
    STATUS,READY,sensor,vref
    LIVE,time_us,sensor,vref,recent_pp
    EVENT,event_id,event_baseline,sample_period_us,raw0,...,raw255

Install:
    python -m pip install pyserial numpy matplotlib

Run:
    python pickleball_monitor_matplotlib.py
"""

import csv
import time
from pathlib import Path

import numpy as np
import serial
import matplotlib.pyplot as plt


PORT = "COM3"
BAUD = 1_000_000

EXPECTED_SAMPLES = 256
PRE_TRIGGER_SAMPLES = 64

EVENTS_FOLDER = Path("events_waveform")


def save_event_csv(event):
    EVENTS_FOLDER.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    ms = int(time.time() * 1000) % 1000

    path = EVENTS_FOLDER / (
        f"event_{event['event_id']:04d}_{stamp}_{ms:03d}.csv"
    )

    raw = event["raw"]
    baseline = event["baseline"]
    period_us = event["sample_period_us"]

    trigger_index = PRE_TRIGGER_SAMPLES - 1

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "time_ms", "raw", "centered"])

        for i, value in enumerate(raw):
            t_ms = (i - trigger_index) * period_us / 1000.0

            writer.writerow([
                i,
                f"{t_ms:.6f}",
                int(value),
                int(value - baseline),
            ])

    return path


def analyze_event(event):
    raw = np.asarray(event["raw"], dtype=float)
    baseline = float(event["baseline"])
    period_us = float(event["sample_period_us"])

    centered = raw - baseline

    # Remove DC for FFT.
    ac = centered - np.mean(centered)

    peak_to_peak = float(np.max(raw) - np.min(raw))
    peak_abs = float(np.max(np.abs(centered)))
    rms = float(np.sqrt(np.mean(ac ** 2)))

    sample_rate = (
        1_000_000.0 / period_us
        if period_us > 0
        else 0.0
    )

    window = np.hanning(len(ac))
    spectrum = np.abs(np.fft.rfft(ac * window))

    freqs = np.fft.rfftfreq(
        len(ac),
        d=(1.0 / sample_rate) if sample_rate > 0 else 1.0,
    )

    dominant = 0.0

    if len(spectrum) > 1:
        useful = spectrum.copy()
        useful[0] = 0.0
        dominant = float(freqs[np.argmax(useful)])

    trigger_index = PRE_TRIGGER_SAMPLES - 1

    time_ms = (
        np.arange(len(raw)) - trigger_index
    ) * period_us / 1000.0

    return {
        "time_ms": time_ms,
        "centered": centered,
        "freqs": freqs,
        "spectrum": spectrum,
        "peak_to_peak": peak_to_peak,
        "peak_abs": peak_abs,
        "rms": rms,
        "dominant": dominant,
        "sample_rate": sample_rate,
    }


def setup_plot():
    plt.ion()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
    )

    wave_ax = axes[0]
    fft_ax = axes[1]

    wave_ax.set_title("Latest Event Waveform")
    wave_ax.set_xlabel("Time relative to trigger (ms)")
    wave_ax.set_ylabel("ADC relative to baseline")
    wave_ax.axvline(0)

    fft_ax.set_title("Latest Event Frequency Spectrum")
    fft_ax.set_xlabel("Frequency (Hz)")
    fft_ax.set_ylabel("Magnitude")
    fft_ax.set_xlim(0, 4000)

    fig.tight_layout()

    plt.show(block=False)

    return fig, wave_ax, fft_ax


def draw_event(fig, wave_ax, fft_ax, event):
    analysis = analyze_event(event)

    wave_ax.clear()
    wave_ax.set_title(
        f"Event #{event['event_id']} Waveform"
    )
    wave_ax.set_xlabel("Time relative to trigger (ms)")
    wave_ax.set_ylabel("ADC relative to baseline")
    wave_ax.axvline(0)

    wave_ax.plot(
        analysis["time_ms"],
        analysis["centered"],
    )

    fft_ax.clear()
    fft_ax.set_title(
        "Frequency Spectrum"
    )
    fft_ax.set_xlabel("Frequency (Hz)")
    fft_ax.set_ylabel("Magnitude")
    fft_ax.set_xlim(0, 4000)

    mask = analysis["freqs"] <= 4000

    fft_ax.plot(
        analysis["freqs"][mask],
        analysis["spectrum"][mask],
    )

    fig.suptitle(
        f"p-p {analysis['peak_to_peak']:.0f} ADC   "
        f"peak {analysis['peak_abs']:.0f} ADC   "
        f"RMS {analysis['rms']:.1f}   "
        f"dominant {analysis['dominant']:.0f} Hz"
    )

    fig.tight_layout()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    return analysis


def process_line(line, fig, wave_ax, fft_ax):
    if not line:
        return

    parts = line.split(",")

    if not parts:
        return

    command = parts[0]

    if command == "STATUS":
        print("STATUS:", line)
        return

    if command == "LIVE":
        # Keep terminal quiet. Uncomment if you want live diagnostics.
        # print(line)
        return

    if command != "EVENT":
        return

    expected_fields = 4 + EXPECTED_SAMPLES

    if len(parts) != expected_fields:
        print(
            f"Rejected incomplete EVENT: "
            f"expected {expected_fields}, got {len(parts)}"
        )
        return

    try:
        event_id = int(parts[1])
        baseline = int(parts[2])
        sample_period_us = float(parts[3])

        raw = np.asarray(
            [int(v) for v in parts[4:]],
            dtype=int,
        )

    except ValueError as exc:
        print("Malformed EVENT:", exc)
        return

    event = {
        "event_id": event_id,
        "baseline": baseline,
        "sample_period_us": sample_period_us,
        "raw": raw,
    }

    analysis = draw_event(
        fig,
        wave_ax,
        fft_ax,
        event,
    )

    path = save_event_csv(event)

    print()
    print(
        f"EVENT #{event_id} | "
        f"p-p={analysis['peak_to_peak']:.0f} | "
        f"peak={analysis['peak_abs']:.0f} | "
        f"RMS={analysis['rms']:.1f} | "
        f"dominant={analysis['dominant']:.0f} Hz"
    )
    print(f"Saved: {path}")
    print()


def main():
    print(f"Opening {PORT} at {BAUD:,} baud...")

    fig, wave_ax, fft_ax = setup_plot()

    with serial.Serial(
        PORT,
        BAUD,
        timeout=0.05,
    ) as ser:

        # Mega resets when serial port opens.
        time.sleep(1.5)
        ser.reset_input_buffer()

        print("Connected.")
        print("Waiting for events. Ctrl+C to stop.\n")

        buffer = bytearray()

        while plt.fignum_exists(fig.number):
            chunk = ser.read(
                ser.in_waiting or 1
            )

            if chunk:
                buffer.extend(chunk)

            if len(buffer) > 65536:
                print("Serial buffer overflow; clearing.")
                buffer.clear()

            while b"\n" in buffer:
                raw_line, _, remainder = (
                    buffer.partition(b"\n")
                )

                buffer = bytearray(remainder)

                line = raw_line.decode(
                    "ascii",
                    errors="ignore",
                ).rstrip("\r")

                process_line(
                    line,
                    fig,
                    wave_ax,
                    fft_ax,
                )

            plt.pause(0.001)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except serial.SerialException as exc:
        print(f"\nSerial error: {exc}")
