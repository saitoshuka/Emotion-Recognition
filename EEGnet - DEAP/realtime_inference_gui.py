import numpy as np
import torch
import torch.nn as nn
import json
import asyncio
import websockets
from pylsl import StreamInlet, resolve_byprop
from collections import deque
import os

# ==========================================
# 1. 定义模型结构 (必须与训练时完全一致)
# ==========================================
class DE_Net(nn.Module):
    def __init__(self, nb_classes=2, chans=32, bands=5):
        super(DE_Net, self).__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, (chans, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(16 * bands, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, nb_classes)
        )
    def forward(self, x):
        x = self.spatial_conv(x); x = self.flatten(x); x = self.fc(x)
        return x

# ==========================================
# 2. 平滑器与映射逻辑
# ==========================================
class EmotionSmoother:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.v = 0.0; self.a = 0.0
    def update(self, v_raw, a_raw):
        self.v = self.alpha * v_raw + (1 - self.alpha) * self.v
        self.a = self.alpha * a_raw + (1 - self.alpha) * self.a
        return self.v, self.a

def prob_to_minus1_1(probs):
    """
    probs_2: [p_low, p_high]
    return: [-1, 1]
    """
    p_high = float(probs[1])
    return 2.0 * p_high - 1.0

# ==========================================
# 3. 实时处理核心
# ==========================================
# 加载你训练好的 s15 冠军模型 (假设你保存为 best_model.pth)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device =", device)

MODEL_V_PATH = r"C:\Users\xinji\checkpoints\s15.dat_valence.pth"
MODEL_A_PATH = r"C:\Users\xinji\checkpoints\s15.dat_arousal.pth"

if not os.path.exists(MODEL_V_PATH) or not os.path.exists(MODEL_A_PATH):
    raise FileNotFoundError(
        f"找不到模型文件：\n{MODEL_V_PATH}\n{MODEL_A_PATH}\n请确认路径/文件名"
    )

def load_checkpoint_state(path: str):
    ckpt = torch.load(path, map_location=device)
    # 你现在的 .pth 是 dict，真正权重在 model_state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt
    # 兼容：如果有人存的是纯 state_dict
    return ckpt, {"note": "pure_state_dict"}

model_v = DE_Net().to(device)
model_a = DE_Net().to(device)
state_v, meta_v = load_checkpoint_state(MODEL_V_PATH)
state_a, meta_a = load_checkpoint_state(MODEL_A_PATH)
model_v.load_state_dict(state_v, strict=True)
model_a.load_state_dict(state_a, strict=True)

model_v.eval()

model_a.eval()
print("✅ Valence ckpt info:", {k: meta_v.get(k) for k in ["subject", "task", "best_fold_acc"] if isinstance(meta_v, dict)})
print("✅ Arousal  ckpt info:", {k: meta_a.get(k) for k in ["subject", "task", "best_fold_acc"] if isinstance(meta_a, dict)})

smoother = EmotionSmoother(alpha=0.2) # alpha越小越丝滑

# ==========================================
# 4. LSL 输入（可选）
# ==========================================
def init_lsl():
    print("🔎 正在寻找 LSL OpenSignals 流...")
    streams = resolve_byprop("name", "OpenSignals")
    inlet = StreamInlet(streams[0])
    print("✅ LSL 已连接")
    return inlet

# ==========================================
# 5. 单步推理（先 dummy 特征，后面换真实特征）
# ==========================================
logit_mean_v = 0.0
logit_mean_a = 0.0

@torch.no_grad()
def infer_from_feat(feat_32x5: np.ndarray):
    global logit_mean_v, logit_mean_a

    feat = feat_32x5.astype(np.float32)
    feat = np.clip(feat, 0.0, None)
    feat = np.log1p(feat)   # 保留 log 压缩（可选但通常有用）
    # ⚠️ 关键：先把全局 z-score 去掉，否则你在 while 里怎么调制都被抹平
    # feat = (feat - feat.mean()) / (feat.std() + 1e-6)

    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    logits_v = model_v(x)  # (1,2)
    logits_a = model_a(x)

    # 1) 用 logit 差（high - low）作为连续值
    dv = (logits_v[0, 1] - logits_v[0, 0]).item()
    da = (logits_a[0, 1] - logits_a[0, 0]).item()

    # 2) 用 EMA 估计 baseline，并去掉（让它围绕0摆动）
    beta = 0.01  # 越小越慢，越稳定
    logit_mean_v = (1 - beta) * logit_mean_v + beta * dv
    logit_mean_a = (1 - beta) * logit_mean_a + beta * da

    dv_c = dv - logit_mean_v
    da_c = da - logit_mean_a

    # 3) 映射到 [-1,1]：tanh(gain * centered_logit)
    gain_v = 1.8
    gain_a = 1.8
    v_raw = float(np.tanh(gain_v * dv_c))
    a_raw = float(np.tanh(gain_a * da_c))

    v_final, a_final = smoother.update(v_raw, a_raw)

    # debug：你会看到 dv_c/da_c 正负来回，a_raw 才会去负数
    # print("dv, dv_c, v_raw:", dv, dv_c, v_raw, "| da, da_c, a_raw:", da, da_c, a_raw)

    return v_final, a_final, None, None

def get_emotion_label(v, a):
    """
    根据 Valence (x) 和 Arousal (y) 映射出情感标签
    阈值 0.1 用于判定是否为中性 (Neutral)
    """
    threshold = 0.1
    
    # 1. 如果都在原点附近，判定为中性
    if abs(v) < threshold and abs(a) < threshold:
        return "Neutral"
    
    # 2. 第一象限 (Valence > 0, Arousal > 0) -> 开心/兴奋
    if v >= 0 and a >= 0:
        return "Happy"
    
    # 3. 第二象限 (Valence < 0, Arousal > 0) -> 愤怒/紧张
    if v < 0 and a >= 0:
        return "Anxious" # 或者 "Angry"
    
    # 4. 第三象限 (Valence < 0, Arousal < 0) -> 悲伤/沮丧
    if v < 0 and a < 0:
        return "Sad"
    
    # 5. 第四象限 (Valence > 0, Arousal < 0) -> 放松/平静
    if v >= 0 and a < 0:
        return "Relaxed"
        
    return "Unknown"

def simulate_de_features_32x5(t: float) -> np.ndarray:
    """
    生成更像真实 EEG 的(32,5)特征：非负、慢变化、带噪声、不同频段不同能量
    """
    # 5 bands: delta theta alpha beta gamma
    band_scale = np.array([2.0, 1.5, 1.2, 0.9, 0.6], dtype=np.float32)

    # 慢变化：全局 arousal/valence-like 因子（不是最终坐标，只是控制能量）
    arousal_factor = 1.0 + 0.25 * np.sin(t * 0.2)
    valence_factor = 1.0 + 0.20 * np.cos(t * 0.17)

    # 通道差异：每个通道固定一个权重（像电极不同）
    ch = 32
    channel_gain = (0.8 + 0.4 * np.sin(np.linspace(0, 2*np.pi, ch, endpoint=False))).astype(np.float32)

    # 组合：外积 -> (32,5)
    base = np.outer(channel_gain, band_scale)  # >=0

    # 加一点相关噪声（避免完全平滑）
    noise = 0.15 * np.random.randn(32, 5).astype(np.float32)

    feat = base * arousal_factor * valence_factor + noise

    # bandpower 类特征一般 >=0，clip一下
    feat = np.clip(feat, 0.0, None)

    return feat.astype(np.float32)
        
# 6. WebSocket Server：前端来连 ws://localhost:8767
# ==========================================
async def ws_handler(websocket):
    print("✅ 前端已连接 WebSocket:", websocket.remote_address)

    # 如果你要真 LSL，把下面两行取消注释
    # inlet = init_lsl()
    # eeg_buffer = deque(maxlen=512)  # 4秒@128Hz（仅示意）

    while True:
        t = asyncio.get_event_loop().time()
        # 每 8 秒换一个象限：0,1,2,3
        q = int(t // 8) % 4
        # 目标象限对应 (valence_sign, arousal_sign)
        # 右上(+,+), 左上(-,+), 左下(-,-), 右下(+,-)
        targets = [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]
        sv, sa = targets[q]

        # 生成一个“基础能量型”特征（非负）
        feat = np.abs(np.random.randn(32, 5)).astype(np.float32)
        band_scale = np.array([2.0, 1.5, 1.2, 0.9, 0.6], dtype=np.float32)
        feat *= band_scale[None, :]

        # 用调制因子让模型输出跨过0.5：
        # sv/sa 决定往 high 或 low 推
        # 这一步不依赖具体模型权重，所以是“强制让它能切换”的 demo 手段
        valence_gain = 1.0 + 0.35 * sv   # +1 -> 1.35, -1 -> 0.65
        arousal_gain = 1.0 + 0.35 * sa

        feat *= (0.8 * valence_gain + 0.2 * arousal_gain)

        # 加少量时间变化和噪声（更像真实）
        feat *= (1.0 + 0.15 * np.sin(t * 0.7))
        feat += 0.05 * np.random.randn(32, 5).astype(np.float32)
        feat = np.clip(feat, 0.0, None)
        # feat = simulate_de_features_32x5(t)
        # -------------------------
        # A) 先用 dummy 直接验证链路
        # -------------------------
        # feat = np.random.randn(32, 5).astype(np.float32)
        # feat = np.random.normal(loc=0, scale=0.01, size=(32, 5)).astype(np.float32)
        # -------------------------
        # B) 真 LSL 时你要做的：
        # sample, ts = inlet.pull_sample()
        # eeg_buffer.append(sample)  # 这里 sample 的通道数要和你特征提取一致
        # if len(eeg_buffer) >= 512:
        #     data = np.array(eeg_buffer).T
        #     feat = extract_de_features_32x5(data)  # 你自己实现
        # -------------------------

        v_final, a_final, _, _ = infer_from_feat(feat)
        # v_final, a_final = smoother.update(feat)
        
# === 【新增】计算情感标签 ===
        # emotion_text = get_emotion_label(v_final, a_final) 

        # === 【修改】Payload 增加 "emotion" 字段 ===
        payload = {
            "valence": float(v_final),
            "arousal": float(a_final),
            "emotion": "EEG",    # <--- 这里对应前端的 data.emotion
            "source": "EEG",
            "status": "connected"       # 建议保留这个状态字段
        }
        await websocket.send(json.dumps(payload))
        # print(f"Sent: V={v_final:.2f}, A={a_final:.2f}")

        await asyncio.sleep(0.1)  # 10Hz

async def main():
    print("🛰️ WS Server listening on ws://localhost:8767")
    async with websockets.serve(ws_handler, "0.0.0.0", 8767):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())