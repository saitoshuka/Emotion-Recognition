import argparse
import time

import numpy as np
from pylsl import StreamInfo, StreamOutlet


def parse_args():
    p = argparse.ArgumentParser(description="Dummy EEG LSL stream for DEAP realtime demo")
    p.add_argument("--stream-name", type=str, default="OpenSignals")
    p.add_argument("--stream-type", type=str, default="EEG")
    p.add_argument("--n-channels", type=int, default=32)
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--noise-std", type=float, default=0.08)
    p.add_argument("--state-sec", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_state(phase):
    states = [
        [(10.0, 1.1), (18.0, 0.35), (5.0, 0.25)],
        [(20.0, 1.0), (28.0, 0.55), (9.0, 0.25)],
        [(2.0, 1.2), (6.0, 0.8), (10.0, 0.15)],
        [(14.0, 0.8), (24.0, 0.7), (32.0, 0.4)],
    ]
    return states[phase % len(states)]


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    info = StreamInfo(
        name=args.stream_name,
        type=args.stream_type,
        channel_count=args.n_channels,
        nominal_srate=args.fs,
        channel_format="float32",
        source_id="dummy_eeg_seediv_deap_demo",
    )
    outlet = StreamOutlet(info)

    channel_gain = np.linspace(0.7, 1.3, args.n_channels).astype(np.float32)
    channel_phase = rng.uniform(0.0, 2 * np.pi, size=args.n_channels).astype(np.float32)

    dt = 1.0 / args.fs
    t = 0.0
    phase = 0
    next_switch = args.state_sec

    print(
        f"[DUMMY LSL] stream='{args.stream_name}', channels={args.n_channels}, fs={args.fs:.1f}Hz, "
        f"switch every {args.state_sec:.1f}s"
    )
    print("[DUMMY LSL] press Ctrl+C to stop")

    try:
        while True:
            if t >= next_switch:
                phase += 1
                next_switch += args.state_sec
                print(f"[DUMMY LSL] state -> {phase % 4}")

            mix = build_state(phase)
            sig = np.zeros(args.n_channels, dtype=np.float32)
            for freq, amp in mix:
                sig += (amp * np.sin(2 * np.pi * freq * t + channel_phase)).astype(np.float32)

            noise = rng.normal(0.0, args.noise_std, size=args.n_channels).astype(np.float32)
            sample = (sig * channel_gain + noise).astype(np.float32)
            outlet.push_sample(sample.tolist())

            t += dt
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\n[DUMMY LSL] stopped.")


if __name__ == "__main__":
    main()
