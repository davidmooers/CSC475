"""
Interactive Piezo Event Trainer + Real-Time Classifier
======================================================

Purpose
-------
One program for the full workflow:

1. RECORD MODE
   - Choose a label, e.g. "finger"
   - Every detected Arduino EVENT is saved under that label
   - Switch labels whenever you want
   - Add more samples to an existing label later

2. PREDICT MODE
   - Retrains from every saved label
   - Predicts one of the known labels OR "other"
   - Logs predictions

Dependencies
------------
    python -m pip install numpy pyserial matplotlib

Arduino protocol expected
-------------------------
    EVENT,event_id,event_baseline,sample_period_us,raw0,...,raw255

Useful terminal commands
------------------------
    help
    labels
    addlabel finger
    addlabel pen
    label finger
    record
    predict
    retrain
    pause
    status
    quit

Typical workflow
----------------
    addlabel finger
    label finger
    record
    # do many finger taps

    addlabel pen
    label pen
    # do many pen taps

    predict
    # new events are classified as finger / pen / other

    record
    label finger
    # add more finger examples

    predict
    # automatically retrains including the new samples

Training data layout
--------------------
    training_data/
        finger/
            event_....csv
        pen/
            event_....csv

Prediction log
--------------
    prediction_log.csv
"""

import argparse
import csv
import queue
import threading
import time
from pathlib import Path

import numpy as np
import serial
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

EXPECTED_SAMPLES = 256
PRE_TRIGGER_SAMPLES = 64

TRAINING_ROOT = Path("training_data")
PREDICTION_LOG = Path("prediction_log.csv")

# "Other" rejection.
# Larger DISTANCE_MARGIN = more willing to accept known labels.
# Larger AMBIGUITY_RATIO = more willing to choose between two close labels.
DISTANCE_MARGIN = 1.50
AMBIGUITY_RATIO = 0.92

# Need at least this many samples in EACH label before prediction.
MIN_SAMPLES_PER_LABEL = 5

FEATURE_NAMES = [
    "peak_to_peak",
    "rms",
    "peak_abs",
    "crest_factor",
    "spectral_centroid",
    "spectral_bandwidth",
    "dominant_freq_over_100",
    "late_to_early_energy",
    "energy_0_300",
    "energy_300_600",
    "energy_600_1000",
    "energy_1000_1500",
    "energy_1500_2500",
    "energy_2500_4000",
]


# ============================================================
# STATE
# ============================================================

class AppState:
    def __init__(self):
        self.mode = "pause"       # pause | record | predict
        self.current_label = None
        self.model = None
        self.running = True

        self.recorded_this_session = 0
        self.predicted_this_session = 0


# ============================================================
# TERMINAL COMMAND THREAD
# ============================================================

def command_reader(command_queue):
    while True:
        try:
            command = input("> ").strip()
        except EOFError:
            return

        command_queue.put(command)

        if command.lower() == "quit":
            return


# ============================================================
# EVENT CSV HELPERS
# ============================================================

def save_training_event(event, label):
    folder = TRAINING_ROOT / label
    folder.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    ms = int(time.time() * 1000) % 1000

    path = folder / (
        f"event_{event['event_id']:04d}_{stamp}_{ms:03d}.csv"
    )

    trigger_index = PRE_TRIGGER_SAMPLES - 1

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sample", "time_ms", "raw", "centered"]
        )

        for i, raw in enumerate(event["raw"]):
            time_ms = (
                (i - trigger_index)
                * event["sample_period_us"]
                / 1000.0
            )

            writer.writerow([
                i,
                f"{time_ms:.6f}",
                int(raw),
                int(raw - event["baseline"]),
            ])

    return path


def read_training_csv(path):
    rows = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    if len(rows) != EXPECTED_SAMPLES:
        return None

    try:
        centered = np.asarray(
            [float(r["centered"]) for r in rows],
            dtype=float,
        )

        time_ms = np.asarray(
            [float(r["time_ms"]) for r in rows],
            dtype=float,
        )
    except (KeyError, ValueError):
        return None

    if len(time_ms) < 2:
        return None

    dt_ms = float(np.median(np.diff(time_ms)))

    if dt_ms <= 0:
        return None

    sample_rate = 1000.0 / dt_ms

    return centered, sample_rate


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(centered, sample_rate):
    centered = np.asarray(centered, dtype=float)

    ac = centered - np.mean(centered)

    peak_to_peak = float(
        np.max(centered) - np.min(centered)
    )

    rms = float(
        np.sqrt(np.mean(ac ** 2))
    )

    baseline_estimate = float(
        np.median(centered[:32])
    )

    peak_abs = float(
        np.max(
            np.abs(
                centered - baseline_estimate
            )
        )
    )

    crest_factor = peak_abs / (rms + 1e-9)

    window = np.hanning(len(ac))

    power = (
        np.abs(
            np.fft.rfft(ac * window)
        ) ** 2
    )

    freqs = np.fft.rfftfreq(
        len(ac),
        d=1.0 / sample_rate,
    )

    # Remove DC.
    power[0] = 0.0

    total_power = (
        float(np.sum(power)) + 1e-12
    )

    spectral_centroid = float(
        np.sum(freqs * power)
        / total_power
    )

    spectral_bandwidth = float(
        np.sqrt(
            np.sum(
                ((freqs - spectral_centroid) ** 2)
                * power
            )
            / total_power
        )
    )

    over_100 = freqs >= 100.0

    if np.any(over_100):
        dominant_freq_over_100 = float(
            freqs[over_100][
                np.argmax(power[over_100])
            ]
        )
    else:
        dominant_freq_over_100 = 0.0

    first_half_energy = (
        float(np.mean(ac[:128] ** 2))
        + 1e-9
    )

    second_half_energy = (
        float(np.mean(ac[128:] ** 2))
        + 1e-9
    )

    late_to_early_energy = (
        second_half_energy
        / first_half_energy
    )

    bands = [
        (0, 300),
        (300, 600),
        (600, 1000),
        (1000, 1500),
        (1500, 2500),
        (2500, 4000),
    ]

    band_fractions = []

    for low, high in bands:
        mask = (
            (freqs >= low)
            & (freqs < high)
        )

        band_fractions.append(
            float(
                np.sum(power[mask])
                / total_power
            )
        )

    return np.asarray(
        [
            peak_to_peak,
            rms,
            peak_abs,
            crest_factor,
            spectral_centroid,
            spectral_bandwidth,
            dominant_freq_over_100,
            late_to_early_energy,
            *band_fractions,
        ],
        dtype=float,
    )


# ============================================================
# TRAINING
# ============================================================

def discover_labels():
    TRAINING_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sorted(
        path.name
        for path in TRAINING_ROOT.iterdir()
        if path.is_dir()
    )


def count_samples(label):
    folder = TRAINING_ROOT / label

    if not folder.exists():
        return 0

    return len(
        list(folder.glob("*.csv"))
    )


def load_all_training_data():
    labels = discover_labels()

    feature_groups = {}
    unusable_counts = {}

    for label in labels:
        folder = TRAINING_ROOT / label

        features = []
        unusable = 0

        for path in sorted(
            folder.glob("*.csv")
        ):
            try:
                parsed = read_training_csv(path)

                if parsed is None:
                    unusable += 1
                    continue

                centered, sample_rate = parsed

                features.append(
                    extract_features(
                        centered,
                        sample_rate,
                    )
                )

            except Exception:
                unusable += 1

        if features:
            feature_groups[label] = np.vstack(
                features
            )

        unusable_counts[label] = unusable

    return feature_groups, unusable_counts


def train_model():
    feature_groups, unusable = (
        load_all_training_data()
    )

    if len(feature_groups) < 2:
        print(
            "\nNeed at least TWO labels with usable samples "
            "before prediction.\n"
        )
        return None

    insufficient = []

    for label, rows in feature_groups.items():
        if len(rows) < MIN_SAMPLES_PER_LABEL:
            insufficient.append(
                (label, len(rows))
            )

    if insufficient:
        print()
        print(
            f"Need at least {MIN_SAMPLES_PER_LABEL} "
            "samples per label:"
        )

        for label, count in insufficient:
            print(
                f"  {label}: {count}"
            )

        print()
        return None

    labels = sorted(
        feature_groups.keys()
    )

    X = np.vstack(
        [
            feature_groups[label]
            for label in labels
        ]
    )

    mean = X.mean(axis=0)
    std = X.std(axis=0)

    std[std < 1e-9] = 1.0

    standardized_groups = {
        label: (
            feature_groups[label] - mean
        ) / std
        for label in labels
    }

    centroids = np.vstack(
        [
            standardized_groups[label].mean(
                axis=0
            )
            for label in labels
        ]
    )

    distance_limits = []

    for index, label in enumerate(labels):
        group = standardized_groups[label]

        distances = np.linalg.norm(
            group - centroids[index],
            axis=1,
        )

        # Use 95th percentile so one extreme training
        # sample does not completely define the class.
        limit = (
            float(
                np.percentile(
                    distances,
                    95,
                )
            )
            * DISTANCE_MARGIN
        )

        distance_limits.append(limit)

    model = {
        "labels": labels,
        "mean": mean,
        "std": std,
        "centroids": centroids,
        "distance_limits": np.asarray(
            distance_limits
        ),
    }

    print()
    print("Model trained")
    print("-------------")

    for i, label in enumerate(labels):
        print(
            f"{label:20s} "
            f"samples={len(feature_groups[label]):4d} "
            f"limit={distance_limits[i]:.2f}"
        )

    print()

    return model


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_event(model, features):
    x = (
        features - model["mean"]
    ) / model["std"]

    distances = np.linalg.norm(
        model["centroids"] - x,
        axis=1,
    )

    order = np.argsort(distances)

    best_index = int(order[0])
    best_distance = float(
        distances[best_index]
    )

    label = model["labels"][
        best_index
    ]

    limit = float(
        model["distance_limits"][
            best_index
        ]
    )

    if len(order) > 1:
        second_distance = float(
            distances[int(order[1])]
        )

        ambiguity_ratio = (
            best_distance
            / (second_distance + 1e-9)
        )
    else:
        second_distance = float("inf")
        ambiguity_ratio = 0.0

    is_too_far = (
        best_distance > limit
    )

    is_ambiguous = (
        ambiguity_ratio
        > AMBIGUITY_RATIO
    )

    if is_too_far or is_ambiguous:
        prediction = "other"
    else:
        prediction = label

    return {
        "prediction": prediction,
        "nearest_label": label,
        "distance": best_distance,
        "distance_limit": limit,
        "second_distance": second_distance,
        "ambiguity_ratio": ambiguity_ratio,
        "distances": {
            model["labels"][i]:
                float(distances[i])
            for i in range(
                len(model["labels"])
            )
        },
    }


# ============================================================
# PREDICTION LOG
# ============================================================

def ensure_prediction_log():
    if PREDICTION_LOG.exists():
        return

    with PREDICTION_LOG.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "event_id",
            "prediction",
            "nearest_label",
            "distance",
            "distance_limit",
            "second_distance",
            "ambiguity_ratio",
            *FEATURE_NAMES,
        ])


def log_prediction(
    event,
    result,
    features,
):
    ensure_prediction_log()

    with PREDICTION_LOG.open(
        "a",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            event["event_id"],
            result["prediction"],
            result["nearest_label"],
            f"{result['distance']:.6f}",
            f"{result['distance_limit']:.6f}",
            f"{result['second_distance']:.6f}",
            f"{result['ambiguity_ratio']:.6f}",
            *[
                f"{value:.8f}"
                for value in features
            ],
        ])


# ============================================================
# SERIAL PARSING
# ============================================================

def parse_event_line(line):
    parts = line.split(",")

    if not parts or parts[0] != "EVENT":
        return None

    if len(parts) != (
        4 + EXPECTED_SAMPLES
    ):
        return None

    try:
        event_id = int(parts[1])
        baseline = int(parts[2])
        sample_period_us = float(
            parts[3]
        )

        raw = np.asarray(
            [
                int(value)
                for value in parts[4:]
            ],
            dtype=float,
        )
    except ValueError:
        return None

    if sample_period_us <= 0:
        return None

    sample_rate = (
        1_000_000.0
        / sample_period_us
    )

    centered = raw - baseline

    return {
        "event_id": event_id,
        "baseline": baseline,
        "sample_period_us":
            sample_period_us,
        "sample_rate": sample_rate,
        "raw": raw,
        "centered": centered,
    }


# ============================================================
# DISPLAY
# ============================================================

def setup_plot():
    plt.ion()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
    )

    fig.canvas.manager.set_window_title(
        "Piezo Trainer / Classifier"
    )

    plt.show(block=False)

    return fig, axes[0], axes[1]


def draw_event(
    fig,
    wave_ax,
    fft_ax,
    event,
    title,
):
    centered = event["centered"]

    trigger_index = (
        PRE_TRIGGER_SAMPLES - 1
    )

    time_ms = (
        np.arange(len(centered))
        - trigger_index
    ) * event[
        "sample_period_us"
    ] / 1000.0

    ac = (
        centered
        - np.mean(centered)
    )

    spectrum = np.abs(
        np.fft.rfft(
            ac * np.hanning(len(ac))
        )
    )

    freqs = np.fft.rfftfreq(
        len(ac),
        d=1.0 / event["sample_rate"],
    )

    wave_ax.clear()

    wave_ax.plot(
        time_ms,
        centered,
    )

    wave_ax.axvline(0)

    wave_ax.set_title(
        f"Event #{event['event_id']} waveform"
    )

    wave_ax.set_xlabel(
        "Time relative to trigger (ms)"
    )

    wave_ax.set_ylabel(
        "ADC relative to baseline"
    )

    fft_ax.clear()

    mask = freqs <= 4000

    fft_ax.plot(
        freqs[mask],
        spectrum[mask],
    )

    fft_ax.set_xlim(0, 4000)
    fft_ax.set_title("FFT")
    fft_ax.set_xlabel("Frequency (Hz)")
    fft_ax.set_ylabel("Magnitude")

    fig.suptitle(title)

    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


# ============================================================
# COMMANDS
# ============================================================

def print_help():
    print("""
Commands
--------
help
    Show this list.

labels
    Show labels and number of saved samples.

addlabel NAME
    Create a new label.

label NAME
    Select the label used by RECORD mode.

record
    Start saving each new EVENT under the selected label.

predict
    Retrain from all saved labels and classify new EVENTs.

retrain
    Retrain immediately without changing mode.

pause
    Receive/display events but do not save or predict them.

status
    Show current mode, selected label, and counts.

quit
    Exit.
""")


def print_labels():
    labels = discover_labels()

    if not labels:
        print(
            "\nNo labels yet. "
            "Use: addlabel finger\n"
        )
        return

    print("\nLabels:")

    for label in labels:
        print(
            f"  {label:20s} "
            f"{count_samples(label)} samples"
        )

    print()


def process_command(
    command,
    state,
):
    command = command.strip()

    if not command:
        return

    pieces = command.split(
        maxsplit=1
    )

    action = pieces[0].lower()

    argument = (
        pieces[1].strip()
        if len(pieces) > 1
        else None
    )

    if action == "help":
        print_help()

    elif action == "labels":
        print_labels()

    elif action == "addlabel":
        if not argument:
            print(
                "Usage: addlabel NAME"
            )
            return

        label = argument.replace(
            " ",
            "_",
        )

        (
            TRAINING_ROOT / label
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        state.current_label = label

        print(
            f"Created/selected label: "
            f"{label}"
        )

    elif action == "label":
        if not argument:
            print(
                "Usage: label NAME"
            )
            return

        label = argument.replace(
            " ",
            "_",
        )

        folder = (
            TRAINING_ROOT / label
        )

        if not folder.exists():
            print(
                f"Label '{label}' does not exist. "
                f"Use: addlabel {label}"
            )
            return

        state.current_label = label

        print(
            f"Selected label: {label}"
        )

    elif action == "record":
        if state.current_label is None:
            print(
                "Select a label first with "
                "'label NAME' or "
                "'addlabel NAME'."
            )
            return

        state.mode = "record"

        print(
            f"\nRECORD MODE -> "
            f"{state.current_label}\n"
        )

    elif action == "predict":
        model = train_model()

        if model is None:
            print(
                "Prediction mode not started."
            )
            return

        state.model = model
        state.mode = "predict"

        print(
            "\nPREDICT MODE\n"
        )

    elif action == "retrain":
        model = train_model()

        if model is not None:
            state.model = model
            print("Model replaced.")

    elif action == "pause":
        state.mode = "pause"
        print("\nPAUSED\n")

    elif action == "status":
        print()
        print(
            f"mode:          {state.mode}"
        )
        print(
            f"current label: "
            f"{state.current_label}"
        )
        print(
            f"recorded:      "
            f"{state.recorded_this_session}"
        )
        print(
            f"predicted:     "
            f"{state.predicted_this_session}"
        )
        print()

    elif action == "quit":
        state.running = False

    else:
        print(
            f"Unknown command: {action}"
        )
        print(
            "Type 'help' for commands."
        )


# ============================================================
# EVENT HANDLING
# ============================================================

def handle_event(
    event,
    state,
    fig,
    wave_ax,
    fft_ax,
):
    if state.mode == "record":
        if state.current_label is None:
            return

        path = save_training_event(
            event,
            state.current_label,
        )

        state.recorded_this_session += 1

        print(
            f"RECORDED event "
            f"{event['event_id']} "
            f"-> {state.current_label} "
            f"({count_samples(state.current_label)} total)"
        )

        draw_event(
            fig,
            wave_ax,
            fft_ax,
            event,
            (
                f"RECORDING: "
                f"{state.current_label}"
            ),
        )

    elif state.mode == "predict":
        if state.model is None:
            return

        features = extract_features(
            event["centered"],
            event["sample_rate"],
        )

        result = classify_event(
            state.model,
            features,
        )

        log_prediction(
            event,
            result,
            features,
        )

        state.predicted_this_session += 1

        distances_text = " | ".join(
            f"{label}={distance:.2f}"
            for label, distance
            in result["distances"].items()
        )

        print(
            f"PREDICT event "
            f"{event['event_id']} "
            f"-> {result['prediction']} "
            f"| nearest={result['nearest_label']} "
            f"| dist={result['distance']:.2f}/"
            f"{result['distance_limit']:.2f} "
            f"| ratio={result['ambiguity_ratio']:.2f}"
        )

        print(
            f"         {distances_text}"
        )

        draw_event(
            fig,
            wave_ax,
            fft_ax,
            event,
            (
                f"PREDICTION: "
                f"{result['prediction']} "
                f"| nearest "
                f"{result['nearest_label']}"
            ),
        )

    else:
        draw_event(
            fig,
            wave_ax,
            fft_ax,
            event,
            "PAUSED - event not saved",
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        default="COM3",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=1_000_000,
    )

    args = parser.parse_args()

    TRAINING_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = AppState()

    command_queue = queue.Queue()

    thread = threading.Thread(
        target=command_reader,
        args=(command_queue,),
        daemon=True,
    )

    thread.start()

    fig, wave_ax, fft_ax = setup_plot()

    def on_key(event):
        if event.key == "q":
            state.running = False
            plt.close(fig)

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    print()
    print(
        "Interactive Piezo Trainer / Classifier"
    )
    print(
        "Type 'help' for commands."
    )
    print()

    with serial.Serial(
        args.port,
        args.baud,
        timeout=0.05,
    ) as ser:

        print(
            f"Opening {args.port} "
            f"at {args.baud:,} baud..."
        )

        time.sleep(1.5)
        ser.reset_input_buffer()

        print("Connected.")
        print("Mode starts PAUSED.\n")

        buffer = bytearray()

        while (
            state.running
            and plt.fignum_exists(
                fig.number
            )
        ):
            while True:
                try:
                    command = (
                        command_queue.get_nowait()
                    )
                except queue.Empty:
                    break

                process_command(
                    command,
                    state,
                )

            chunk = ser.read(
                ser.in_waiting or 1
            )

            if chunk:
                buffer.extend(chunk)

            if len(buffer) > 65536:
                print(
                    "Serial buffer overflow; "
                    "clearing."
                )
                buffer.clear()

            while b"\n" in buffer:
                raw_line, _, remainder = (
                    buffer.partition(b"\n")
                )

                buffer = bytearray(
                    remainder
                )

                line = raw_line.decode(
                    "ascii",
                    errors="ignore",
                ).rstrip("\r")

                event = parse_event_line(
                    line
                )

                if event is not None:
                    handle_event(
                        event,
                        state,
                        fig,
                        wave_ax,
                        fft_ax,
                    )

            plt.pause(0.001)

    print("\nFinished.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except serial.SerialException as exc:
        print(
            f"\nSerial error: {exc}"
        )
