import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import asyncio
import json
import threading
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from pylsl import StreamInlet
try:
    from pylsl import resolve_stream
except ImportError:
    from pylsl import resolve_byprop as resolve_stream
from scipy.signal import welch
import websockets

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


EMOTION_MAP = [
    ("sleepy", 0.01, -1.00), ("tired", -0.01, -1.00), ("afraid", -0.12, 0.79),
    ("angry", -0.40, 0.79), ("calm", 0.78, -0.67), ("relaxed", 0.71, -0.65),
    ("content", 0.81, -0.55), ("depressed", -0.81, -0.48), ("discontent", -0.68, -0.32),
    ("determined", 0.89, 0.35), ("happy", 0.89, 0.17), ("anxious", -0.72, -0.80),
    ("good", 0.78, 0.35), ("pensive", 0.03, -0.60), ("impressed", 0.39, 0.06),
    ("frustrated", -0.60, 0.40), ("disappointed", -0.80, -0.03), ("bored", -0.35, -0.78),
    ("annoyed", -0.44, 0.76), ("enraged", -0.18, 0.83), ("excited", 0.70, 0.71),
    ("melancholy", -0.65, -0.65), ("satisfied", 0.77, -0.63), ("distressed", -0.76, 0.83),
    ("uncomfortable", -0.68, -0.37), ("worried", -0.07, 0.32), ("amused", 0.55, -0.02),
    ("apathetic", -0.20, -0.12), ("peaceful", 0.55, -0.60), ("contemplative", 0.58, -0.60),
    ("embarrassed", -0.31, -0.61), ("sad", -0.81, -0.40), ("hopeful", 0.61, 0.40),
    ("pleased", 0.89, -0.10),
]


class EEGNetRegressor(nn.Module):
    def __init__(self, chans=32, bands=5, f1=16, d=2, f2=32, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            z = torch.zeros(1, 1, chans, bands)
            z = self.block1(z)
            z = self.block2(z)
            flat = z.numel()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


def map_0_9_to_neg1_1(x):
    return (x / 4.5) - 1.0


def nearest_emotion_label(v, a):
    best_name, best_d = "unknown", 1e9
    for name, ev, ea in EMOTION_MAP:
        d = (v - ev) ** 2 + (a - ea) ** 2
        if d < best_d:
            best_d = d
            best_name = name
    return best_name


def compute_de_features(window_data, fs):
    bands = [(4, 8), (8, 13), (13, 20), (20, 30), (0.5, 4)]
    chans = window_data.shape[0]
    out = np.zeros((chans, 5), dtype=np.float32)
    for ch in range(chans):
        sig = window_data[ch]
        nperseg = min(len(sig), int(fs * 2))
        if nperseg < 8:
            nperseg = len(sig)
        freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
        for bi, (lo, hi) in enumerate(bands):
            mask = (freqs >= lo) & (freqs <= hi)
            out[ch, bi] = np.log(np.sum(psd[mask]) + 1e-8)
    return out


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    chans = int(ckpt.get("chans", 32))
    bands = int(ckpt.get("bands", 5))
    label_space = ckpt.get("label_space", "0_9")
    model_channel_indices = ckpt.get("channel_indices", list(range(chans)))

    model = EEGNetRegressor(chans=chans, bands=bands).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, chans, label_space, model_channel_indices


def load_scaler(path):
    d = np.load(path)
    return d["mean"], d["std"]


async def ws_client_loop(uri, shared, stop_event):
    while not stop_event.is_set():
        try:
            async with websockets.connect(uri) as ws:
                print(f"[WS] connected: {uri}")
                while not stop_event.is_set():
                    outbound = None
                    with shared["lock"]:
                        if shared["pending_send"] is not None:
                            outbound = shared["pending_send"]
                            shared["pending_send"] = None
                    if outbound is not None:
                        await ws.send(json.dumps(outbound))

                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                        data = json.loads(msg)
                        if data.get("type") == "update":
                            d = data.get("data", {})
                        else:
                            d = data
                        src = str(d.get("source", "")).lower()
                        if src == "morphcast":
                            with shared["lock"]:
                                shared["morphcast"] = {
                                    "valence": float(d.get("valence", 0.0)),
                                    "arousal": float(d.get("arousal", 0.0)),
                                    "emotion": str(d.get("emotion", "")),
                                }
                    except asyncio.TimeoutError:
                        pass
        except Exception as e:
            print(f"[WS] reconnect after error: {e}")
            await asyncio.sleep(1.0)


def lsl_loop(args, model, scaler, label_space, device, shared, stop_event):
    print(f"[LSL] resolving stream by name={args.stream_name}")
    streams = resolve_stream("name", args.stream_name)
    if not streams:
        raise RuntimeError(f"No LSL stream found for name={args.stream_name}")
    inlet = StreamInlet(streams[0])

    maxlen = int(args.window_sec * args.fs)
    step = max(1, int(args.step_sec * args.fs))
    buf = deque(maxlen=maxlen)
    since = 0

    center_n = max(1, int(args.center_sec / args.step_sec))
    norm_n = max(1, int(args.norm_sec / args.step_sec))
    hist_len = max(center_n, norm_n)
    pred_hist = deque(maxlen=hist_len)
    demo_smooth_alpha = float(np.clip(args.demo_smooth_alpha, 0.0, 1.0))
    demo_state = None

    while not stop_event.is_set():
        sample, ts = inlet.pull_sample(timeout=1.0)
        if sample is None:
            continue
        vals = [float(sample[i]) for i in args.channel_indices]
        buf.append(vals)
        since += 1

        if len(buf) < maxlen or since < step:
            continue
        since = 0

        arr = np.asarray(buf, dtype=np.float32).T
        feat = compute_de_features(arr, fs=args.fs)
        mean, std = scaler
        feat = (feat - mean) / (std + 1e-8)
        feat = np.asarray(feat, dtype=np.float32)
        feat = np.squeeze(feat)
        if feat.ndim != 2:
            raise ValueError(f"Unexpected feature shape after scaling: {feat.shape}")

        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).squeeze(0).cpu().numpy()

        v_raw, a_raw = float(pred[0]), float(pred[1])
        if label_space == "0_9":
            v = map_0_9_to_neg1_1(v_raw)
            a = map_0_9_to_neg1_1(a_raw)
        else:
            v, a = v_raw, a_raw

        if args.real_output:
            v_out = float(np.clip(v, -1.0, 1.0))
            a_out = float(np.clip(a, -1.0, 1.0))
        else:
            pred_hist.append((v, a))
            hist_arr = np.asarray(pred_hist, dtype=np.float32)
            center_arr = hist_arr[-center_n:]
            norm_arr = hist_arr[-norm_n:]
            mu = center_arr.mean(axis=0)
            sigma = norm_arr.std(axis=0) + 1e-6
            z = (np.array([v, a], dtype=np.float32) - mu) / sigma
            v_out = float(np.tanh(args.demo_gain * float(z[0])))
            a_out = float(np.tanh(args.demo_gain * float(z[1])))

            if demo_smooth_alpha > 0.0:
                current = np.array([v_out, a_out], dtype=np.float32)
                if demo_state is None:
                    demo_state = current
                else:
                    demo_state = (1.0 - demo_smooth_alpha) * demo_state + demo_smooth_alpha * current
                v_out = float(demo_state[0])
                a_out = float(demo_state[1])

        emo = nearest_emotion_label(v_out, a_out)

        payload = {
            "source": "EEG",
            "timestamp": float(ts),
            "valence": v_out,
            "arousal": a_out,
            "emotion": emo,
            "raw_valence": v_raw,
            "raw_arousal": a_raw,
            "mapped_valence": float(v),
            "mapped_arousal": float(a),
            "mode": "real" if args.real_output else "demo",
        }
        with shared["lock"]:
            shared["eeg"] = {"valence": v_out, "arousal": a_out, "emotion": emo}
            shared["pending_send"] = payload

        print(f"[EEG:{'real' if args.real_output else 'demo'}] v={v_out:.3f}, a={a_out:.3f}, emotion={emo}")


def run_plot(shared, stop_event):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="gray", lw=1)
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("Real-time Circumplex: EEG (red) vs Morphcast (blue)")

    for _, ev, ea in EMOTION_MAP:
        ax.scatter(ev, ea, s=16, color="lightgray")

    eeg_dot = ax.scatter([0], [0], s=120, c="tab:red", label="EEG")
    morph_dot = ax.scatter([0], [0], s=120, c="tab:blue", label="Morphcast")
    txt = ax.text(-1.0, 1.02, "", fontsize=10)
    ax.legend(loc="lower right")

    def update(_):
        with shared["lock"]:
            eeg = shared.get("eeg")
            morph = shared.get("morphcast")
        if eeg:
            eeg_dot.set_offsets(np.array([[eeg["valence"], eeg["arousal"]]]))
        if morph:
            morph_dot.set_offsets(np.array([[morph["valence"], morph["arousal"]]]))
        eemo = eeg["emotion"] if eeg else "-"
        memo = morph["emotion"] if morph else "-"
        txt.set_text(f"EEG: {eemo} | Morphcast: {memo}")
        return eeg_dot, morph_dot, txt

    ani = FuncAnimation(fig, update, interval=100, blit=False)

    def on_close(_event):
        stop_event.set()

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()
    return ani


def parse_args():
    p = argparse.ArgumentParser(description="Realtime EEGNet circumplex with websocket compare (DEAP)")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--scaler-path", type=str, required=True)
    p.add_argument("--stream-name", type=str, default="OpenSignals")
    p.add_argument("--fs", type=int, default=100)
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument("--step-sec", type=float, default=0.25)
    p.add_argument("--channel-indices", type=str, default="")
    p.add_argument("--ws-uri", type=str, default="ws://localhost:8767")
    p.add_argument("--demo-gain", type=float, default=8.0, help="Visualization gain for valence/arousal (demo mode only)")
    p.add_argument("--center-sec", type=float, default=20.0, help="Sliding window seconds for mean-centering (demo mode)")
    p.add_argument("--norm-sec", type=float, default=20.0, help="Sliding window seconds for std-normalization (demo mode)")
    p.add_argument("--demo-smooth-alpha", type=float, default=0.18, help="EWMA smoothing strength for demo output, 0 disables")
    p.add_argument("--real-output", action="store_true", help="Disable demo post-processing and output real mapped model values")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, n_chans, label_space, model_channel_indices = load_model(args.model_path, device)

    if args.channel_indices:
        args.channel_indices = [int(x.strip()) for x in args.channel_indices.split(",") if x.strip()]
    else:
        args.channel_indices = [int(x) for x in model_channel_indices]

    if n_chans != len(args.channel_indices):
        raise ValueError(
            f"Model expects {n_chans} channels, but channel-indices has {len(args.channel_indices)}: {args.channel_indices}"
        )
    scaler = load_scaler(args.scaler_path)

    shared = {
        "lock": threading.Lock(),
        "eeg": None,
        "morphcast": None,
        "pending_send": None,
    }
    stop_event = threading.Event()

    ws_loop = asyncio.new_event_loop()

    def ws_thread_fn():
        asyncio.set_event_loop(ws_loop)
        ws_loop.run_until_complete(ws_client_loop(args.ws_uri, shared, stop_event))

    ws_thread = threading.Thread(target=ws_thread_fn, daemon=True)
    ws_thread.start()

    lsl_thread = threading.Thread(
        target=lsl_loop,
        args=(args, model, scaler, label_space, device, shared, stop_event),
        daemon=True,
    )
    lsl_thread.start()

    run_plot(shared, stop_event)
    stop_event.set()
    time.sleep(0.2)


if __name__ == "__main__":
    main()








