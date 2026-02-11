import os
import random
from typing import Any, Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

import matplotlib.pyplot as plt
import math

# ==========================
# 0) Environment & Utilities
# ==========================
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_WINDOWS = (os.name == "nt")
DEFAULT_NUM_WORKERS = 0 if IS_WINDOWS else 4

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errors = y_pred - y_true
    errors = np.clip(errors, -500, 500) 
    
    scores = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0
    )
    return float(np.sum(scores))


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def apply_scaler_inplace(df: pd.DataFrame, feat_cols: List[str], scaler: StandardScaler) -> None:
    df[feat_cols] = scaler.transform(df[feat_cols]).astype(np.float32)


# ==========================
# 1) Sliding Windows & Datasets
# ==========================
def build_windows(
    df: pd.DataFrame,
    feat_cols: List[str],
    target_col: str,
    window_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xs, ys, unit_ids = [], [], []

    for unit_num, group in df.groupby("unit_number", sort=True):
        features = group[feat_cols].to_numpy(dtype=np.float32)
        targets = group[target_col].to_numpy(dtype=np.float32)

        if len(features) < window_size:
            continue

        for i in range(len(features) - window_size + 1):
            window_feats = features[i:i + window_size]
            window_target = targets[i + window_size - 1]

            Xs.append(window_feats)
            ys.append(window_target)
            unit_ids.append(unit_num)

    if not Xs:
        return (
            np.zeros((0, window_size, len(feat_cols)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.array([], dtype=np.int32)
        )

    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(unit_ids, dtype=np.int32)
    )


def build_test_windows(
    df_test: pd.DataFrame,
    feat_cols: List[str],
    window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    Xs, unit_ids = [], []

    for unit_num, group in df_test.groupby("unit_number", sort=True):
        features = group[feat_cols].to_numpy(dtype=np.float32)

        padded_window = np.zeros((window_size, features.shape[1]), dtype=np.float32)
        actual_len = min(len(features), window_size)

        if actual_len > 0:
            padded_window[-actual_len:] = features[-actual_len:]

        Xs.append(padded_window)
        unit_ids.append(unit_num)

    if not Xs:
        return (
            np.zeros((0, window_size, len(feat_cols)), dtype=np.float32),
            np.array([], dtype=np.int32)
        )

    return (
        np.asarray(Xs, dtype=np.float32),
        np.asarray(unit_ids, dtype=np.int32)
    )


class RULDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32).permute(0, 2, 1).contiguous()
        self.y = torch.as_tensor(y, dtype=torch.float32).contiguous()

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class CAEDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32).permute(0, 2, 1).contiguous()

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        return x, x


def make_loader(
    dataset: Dataset,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = DEFAULT_NUM_WORKERS
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False
    )


# ==========================
# 2) Models
# ==========================
class CAE(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(128, num_features, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class RULModelLSTM(nn.Module):
    def __init__(self, encoder, lstm_units=128, dropout=0.4, bidirectional=False):
        super().__init__()
        self.encoder = encoder
        nd = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.attention_dense = nn.Linear(lstm_units * nd, 1)

        self.fc = nn.Sequential(
            nn.Linear(lstm_units * nd, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.permute(0, 2, 1).contiguous()
        
        lstm_out, _ = self.lstm(h)

        if self.lstm.bidirectional:
            context = lstm_out[:, -1, :]
        else:
            attention_scores = torch.tanh(self.attention_dense(lstm_out))
            attention_weights = torch.softmax(attention_scores, dim=1)
            context = (attention_weights * lstm_out).sum(dim=1)

        output = self.fc(context)
        return output.squeeze(1)

class RULModelGRU(nn.Module):
    def __init__(self, encoder, gru_units=128, dropout=0.4, bidirectional=False, attention=False):
        super().__init__()
        self.encoder = encoder
        self.attention = attention        
        
        self.num_directions = 2 if bidirectional else 1
        
        self.gru = nn.GRU(
            input_size=256, 
            hidden_size=gru_units, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=bidirectional
        )
        
        if attention:
            self.attn = nn.Linear(gru_units * self.num_directions, 1)

        self.fc = nn.Sequential(
            nn.Linear(gru_units * self.num_directions, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        h = self.encoder(x).permute(0, 2, 1).contiguous()
        out, h_n = self.gru(h)
        
        
        if self.attention:
            scores = self.attn(out)
            weights = torch.softmax(torch.tanh(scores), dim=1)
            ctx = (weights * out).sum(dim=1)
        else:
            if self.gru.bidirectional:
                ctx = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                ctx = out[:, -1, :]
            
        return self.fc(ctx).squeeze(1)
    
# ==========================
# 3) Label mappings
# ==========================

def _create_baseline_mapper(p: Dict[str, Any]):
    CAP = float(p.get("CAP", 125.0))

    def phi(y):
        y = np.asarray(y, float)
        return np.clip(y, 0.0, CAP)

    def inv(v):
        v = np.asarray(v, float)
        return np.clip(v, 0.0, CAP)

    return phi, inv, f"Baseline(CAP={int(CAP)})"


def _create_exp_concave_mapper(p: Dict[str, Any]):
    CAP = float(p.get("CAP", 125.0))
    k = float(p.get("k", 1.5))
    eps = 1e-12

    if abs(k) < 1e-8:
        def phi(y):
            y = np.asarray(y, float)
            return np.clip(y, 0.0, CAP)

        def inv(v):
            v = np.asarray(v, float)
            return np.clip(v, 0.0, CAP)

        return phi, inv, f"Exp-Concave≈Id(CAP={int(CAP)},k~0)"

    denom = -np.expm1(-k) + eps

    def phi(y):
        y = np.asarray(y, float)
        u = np.clip(y, 0.0, CAP) / CAP
        num = -np.expm1(-k * u)
        return CAP * (num / denom)

    def inv(v):
        v = np.asarray(v, float)
        w = np.clip(v, 0.0, CAP) / CAP

        arg = -w * denom
        arg = np.clip(arg, -1.0 + 1e-7, None)

        return (CAP / (k + eps)) * (-np.log1p(arg))

    return phi, inv, f"Exp-Concave(CAP={int(CAP)},k={k:g})"


def create_rul_mapper(p: Dict[str, Any]):
    t = str(p.get("type", "baseline")).lower().replace("-", "_").strip()

    if t in ("baseline", "cap", "clipped"):
        return _create_baseline_mapper(p)
    if t in ("exp_concave", "exp_shape"):
        return _create_exp_concave_mapper(p)

    # 그 외 알 수 없는 타입은 기본값(Baseline)으로 처리
    print(f"[create_rul_mapper] unknown '{t}', fallback Baseline.")
    return _create_baseline_mapper(p)

# ==========================
# 4) Train/Eval helpers
# ==========================

# 1. (RMSE, MAE, NASA Score)
def calculate_metrics(y_true, y_pred, cap_eval=None):
    
    if cap_eval is not None:
        y_true = np.minimum(y_true, cap_eval)
        y_pred = np.minimum(y_pred, cap_eval)

    
    rmse = rmse_np(y_true, y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))  
    nasa = nasa_score(y_true, y_pred)
    
    return {"rmse": rmse, "mae": mae, "nasa": nasa}


# 2. Epoch 학습
def train_epoch(model, loader, loss_fn, opt):
    model.train()
    tot = 0.0
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True).squeeze(-1)
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += float(loss.item())
    return tot / max(1, len(loader))


# 3. [전체 시점 평가]
@torch.no_grad()
def eval_all_metrics(model, X, y_raw, inv_phi, cap_eval=None):
    model.eval()
    xb = torch.from_numpy(X).permute(0, 2, 1).contiguous().to(DEVICE)
    pred_lbl = model(xb).detach().cpu().numpy()

    # 역변환 및 후처리
    yhat = inv_phi(pred_lbl)
    yhat = np.nan_to_num(yhat, nan=0.0, posinf=1e9, neginf=0.0)
    yhat = np.clip(yhat, 0, None)

    # 공통 함수로 점수 계산
    return calculate_metrics(y_raw, yhat, cap_eval)


# 4. [EOL 시점 평가]
@torch.no_grad()
def eval_eol_metrics(model, X, y_raw, units, inv_phi, cap_eval=None):
    model.eval()
    
    # EOL(각 엔진의 마지막 시점) 인덱스 추출
    df = pd.DataFrame({"i": np.arange(len(units)), "u": units})
    last_idx = df.groupby("u", sort=False)["i"].max().values

    # EOL 데이터만 필터링
    X_eol = X[last_idx]
    y_eol = y_raw[last_idx]

    # 예측
    xb = torch.from_numpy(X_eol).permute(0, 2, 1).contiguous().to(DEVICE)
    pred_lbl = model(xb).detach().cpu().numpy()

    # 역변환 및 후처리
    yhat = inv_phi(pred_lbl)
    yhat = np.nan_to_num(yhat, nan=0.0, posinf=1e9, neginf=0.0)
    yhat = np.clip(yhat, 0, None)

    # 공통 함수로 점수 계산
    return calculate_metrics(y_eol, yhat, cap_eval)


# ==========================
# 5) CAE pretraining
# ==========================
def pretrain_cae(
    train_df_proc: pd.DataFrame,
    feat_cols: List[str],
    window_size: int = 30,
    epochs: int = 25,
    batch_size: int = 512,
    lr: float = 1e-3,
    tag: str = "",
    seed: int = 42,
    do_visualize: bool = False,       
    save_path: str = "cae_result.png" 
):
    # 시드 고정 및 초기 설정 로그 출력
    set_seed(seed, deterministic=True)
    
    header = f" CAE PRE-TRAINING PHASE {tag} "
    print("\n" + "=" * 80)
    print(f"{header.center(80, '=')}")
    print(f"│ Configuration : Seed={seed} | Epochs={epochs} | Batch={batch_size} | LR={lr}")
    print(f"│ Device        : {DEVICE}")
    print("=" * 80)

    Xs = []
    for _, group in train_df_proc.groupby("unit_number", sort=True):
        features = group[feat_cols].to_numpy(dtype=np.float32)
        if len(features) < window_size:
            continue
        for i in range(len(features) - window_size + 1):
            Xs.append(features[i:i + window_size])

    X_cae = np.asarray(Xs, dtype=np.float32)
    print(f" Input Shape: {X_cae.shape}")

    
    cae = CAE(num_features=len(feat_cols)).to(DEVICE)
    opt = torch.optim.Adam(cae.parameters(), lr=lr)
    mse = nn.MSELoss()
    
    dataset = CAEDataset(X_cae)
    loader = make_loader(dataset, batch_size=batch_size, shuffle=True)

    print(f" Optimization Started")
    
    for epoch in range(epochs):
        cae.train()
        total_loss = 0.0
        
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                xb = batch[0]
            else:
                xb = batch
            
            xb = xb.to(DEVICE)
            
            # --- Forward & Backward ---
            pred = cae(xb)       
            loss = mse(pred, xb) 
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(loader))
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            print(f"│ [EPOCH {epoch+1:02d}/{epochs}] Reconstruction Loss: {avg_loss:.6f}")

    
    if do_visualize:
        print(f"│ [VIS] Saving reconstruction visualization -> {save_path}")
        visualize_cae_result(cae, loader, feat_cols, save_path=save_path)

    
    print(f"│ [INFO] Pre-training Complete. Encoder weights extracted.")
    print("=" * 80 + "\n")
    
    return {k: v.detach().clone() for k, v in cae.encoder.state_dict().items()}


def make_model(
    model_type: str, 
    encoder: nn.Module, 
    units: int = 128, 
    dropout: float = 0.4
) -> nn.Module:
    
    mt = model_type.lower()
    
    if mt == "lstm":
        model = RULModelLSTM(encoder, lstm_units=units, dropout=dropout, bidirectional=False)
    elif mt == "bilstm":
        model = RULModelLSTM(encoder, lstm_units=units, dropout=dropout, bidirectional=True)
    elif mt == "gru":
        model = RULModelGRU(encoder, gru_units=units, dropout=dropout, bidirectional=False)
    elif mt == "bigru":
        model = RULModelGRU(encoder, gru_units=units, dropout=dropout, bidirectional=True)
    elif mt == "abgru":
        model = RULModelGRU(encoder, gru_units=units, dropout=dropout, bidirectional=True, attention=True)
    else:
        raise ValueError(f"[ERROR] Unknown model_type: {model_type}")

    return model.to(DEVICE)


# ==========================
# 6) Single experiment
# ==========================
def run_single_experiment(
    train_df_proc,
    val_df_proc,
    test_df_proc,
    feat_cols,
    base_encoder_state,
    model_type,
    phi_params,
    window_size=30,
    lr=1e-3,
    epochs=25,
    patience=8,
    cap_eval_val=None,
    do_test=False,
    exp_tag: str = "",
    seed: int = 42,
    val_mode: str = "all"
):
    set_seed(seed, deterministic=True)
    phi, inv_phi, name = create_rul_mapper(phi_params)

    # 1. Encoder 설정
    encoder = CAE(num_features=len(feat_cols)).encoder.to(DEVICE)
    encoder.load_state_dict(base_encoder_state)
    encoder.requires_grad_(False)
    encoder.eval()

    # 2. 데이터셋 생성
    X_tr, y_tr_raw, _ = build_windows(train_df_proc, feat_cols, "RUL_raw_unclipped", window_size)
    y_tr_lbl = phi(y_tr_raw).astype(np.float32)

    X_va, y_va_raw, u_va = build_windows(val_df_proc, feat_cols, "RUL_raw_unclipped", window_size)

    # Late 구간 정의 (CAP인 125.0 이하를 Late로 간주)
    mask_late = (y_va_raw <= 125.0)

    # 3. 모델 및 학습 설정
    model = make_model(model_type, encoder, units=128, dropout=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()
    loader_tr = make_loader(RULDataset(X_tr, y_tr_lbl), batch_size=512, shuffle=True)

    tag_str = f"[{exp_tag}] " if exp_tag else ""

    best_res = {
        "rmse_all": float("inf"),
        "rmse_late": float("inf"),
        "mae_all": float("inf"), 
        "nasa": float("inf")
    }
    best_state = None
    wait = 0

    for ep in range(epochs):
        train_loss = train_epoch(model, loader_tr, mse, opt)

        model.eval()
        with torch.no_grad():
            xb_val = torch.from_numpy(X_va).permute(0, 2, 1).contiguous().to(DEVICE)
            pred_lbl = model(xb_val).cpu().numpy()

        yhat_val = inv_phi(pred_lbl)
        yhat_val = np.nan_to_num(yhat_val, nan=0.0, posinf=125.0, neginf=0.0)
        
        yhat_val_capped = np.clip(yhat_val, 0, 125.0)
        y_va_capped = np.minimum(y_va_raw, 125.0)

        # A. 전체 구간 (All)
        res_all = calculate_metrics(y_va_capped, yhat_val_capped)

        # B. Late 구간
        if mask_late.sum() > 0:
            res_late = calculate_metrics(y_va_raw[mask_late], yhat_val_capped[mask_late])
            rmse_late = res_late["rmse"]
        else:
            rmse_late = res_all["rmse"]

        # (3) Early Stopping (Late RMSE 기준)
        improved = rmse_late < best_res["rmse_late"] - 1e-4

        if improved:
            best_res["rmse_all"] = res_all["rmse"]
            best_res["rmse_late"] = rmse_late
            best_res["mae_all"] = res_all["mae"] 
            best_res["nasa"] = res_all["nasa"]
            
            wait = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

        if (ep + 1) % 5 == 0 or (ep + 1) == 1:
            print(
                f"{tag_str}[{model_type}] Ep {ep + 1:02d} TL={train_loss:.4f} "
                f"VR_All={res_all['rmse']:.4f} VR_Late={rmse_late:.4f} {'★' if improved else ''}"
            )

    
    if best_state is not None:
        model.load_state_dict(best_state)

    
    print(f"{tag_str}>> [Fine-tuning] Unfreezing Encoder & Tuning for 5 epochs...")
    
    for param in model.encoder.parameters():
        param.requires_grad = True
        
    opt_ft = torch.optim.Adam(model.parameters(), lr=lr * 0.1)

    model.train()
    for ft_ep in range(5):
        loss_ft = train_epoch(model, loader_tr, mse, opt_ft) 
        print(f"{tag_str}   Fine-tune Ep {ft_ep+1}/5 | Loss: {loss_ft:.6f}") 

    model.eval()
    with torch.no_grad():
        xb_val = torch.from_numpy(X_va).permute(0, 2, 1).contiguous().to(DEVICE)
        pred_lbl = model(xb_val).cpu().numpy()

    yhat_val = inv_phi(pred_lbl)
    yhat_val = np.nan_to_num(yhat_val, nan=0.0, posinf=125.0, neginf=0.0)
    yhat_val_capped = np.clip(yhat_val, 0, 125.0)
    y_va_capped = np.minimum(y_va_raw, 125.0)

    res_final = calculate_metrics(y_va_capped, yhat_val_capped)
    
    if mask_late.sum() > 0:
        res_late_final = calculate_metrics(y_va_raw[mask_late], yhat_val_capped[mask_late])
        rmse_late_final = res_late_final["rmse"]
    else:
        rmse_late_final = res_final["rmse"]

    best_res["rmse_all"] = res_final["rmse"]
    best_res["rmse_late"] = rmse_late_final
    best_res["mae_all"] = res_final["mae"]
    best_res["nasa"] = res_final["nasa"]

    print(f"{tag_str}>> [Final Result] VR_All={best_res['rmse_all']:.4f} VR_Late={best_res['rmse_late']:.4f}")

    # === Return Dictionary Construction ===
    ret = {
        "label": name,
        "model": model_type,
        "best_val_rmse_all": float(best_res["rmse_all"]),
        "best_val_rmse_late": float(best_res["rmse_late"]),
        "best_val_mae_all": float(best_res["mae_all"]), 
        "best_val_nasa": float(best_res["nasa"]),
        "model_obj": model,
        "inv_phi": inv_phi,
        "phi_params": phi_params,
        "rmse_test": None,
        "mae_test": None,
        "s_test": None,
        "rmse_test_cap125": None,
        "mae_test_cap125": None,
        "s_test_cap125": None,
    }

    # === Test Set Evaluation ===
    if do_test and (test_df_proc is not None) and (not test_df_proc.empty):
        model.eval()
        X_te, _ = build_test_windows(test_df_proc, feat_cols, window_size)

        with torch.no_grad():
            xb = torch.from_numpy(X_te).permute(0, 2, 1).contiguous().to(DEVICE)
            pred_lbl = model(xb).cpu().numpy()

        y_pred = inv_phi(pred_lbl)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=125.0, neginf=0.0)
        y_pred = np.clip(y_pred, 0, None)

        truth_path = os.path.join(os.getcwd(), "RUL_FD001.txt")
        if os.path.exists(truth_path):
            y_true = pd.read_csv(truth_path, sep=r"\s+", header=None).values.ravel().astype(float)
            
            # (A) (CAP @125) 
            res_c = calculate_metrics(y_true, y_pred, cap_eval=125.0)

            # (B) 전체 관점 평가 (Raw)
            res_raw = calculate_metrics(y_true, y_pred, cap_eval=None)

            print(f"[TEST] {model_type} RMSE(All)={res_raw['rmse']:.3f} | RMSE(@125)={res_c['rmse']:.3f} | MAE(@125)={res_c['mae']:.3f}")

            ret.update({
                "rmse_test": res_raw["rmse"],
                "mae_test": res_raw["mae"],
                "s_test": res_raw["nasa"],
                "rmse_test_cap125": res_c["rmse"],
                "mae_test_cap125": res_c["mae"],
                "s_test_cap125": res_c["nasa"],
                "y_pred_test": y_pred,
            })
        else:
            print(f"[Warning] True RUL file not found at {truth_path}")

    return ret


# ==========================
# 7) Ensemble & Utilities
# ==========================
def ensemble_predict(
    models: List[nn.Module],
    X: np.ndarray,
    inv_phi_funcs: List[Callable],
    weights: Optional[np.ndarray] = None,
    cap_at: float = 125.0
) -> np.ndarray:
    if weights is None:
        w = np.full(len(models), 1.0 / len(models), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(models), 1.0 / len(models), dtype=float)

    X = X.astype(np.float32, copy=False)
    xb = torch.from_numpy(X).permute(0, 2, 1).contiguous().to(DEVICE)

    preds = []
    with torch.inference_mode():
        for model, inv_phi in zip(models, inv_phi_funcs):
            model.eval()
            pred_lbl = model(xb).cpu().numpy()
            y = inv_phi(pred_lbl)
            y = np.nan_to_num(y, nan=0.0, posinf=cap_at, neginf=0.0)
            y = np.clip(y, 0, cap_at)
            preds.append(y)

    return np.average(preds, axis=0, weights=w)

def visualize_cae_result(model, loader, feat_cols, save_path="cae_result.png"):
    model.eval()
    try:
        xb = next(iter(loader))
        if isinstance(xb, (list, tuple)): xb = xb[0]
    except StopIteration:
        return
    
    xb = xb.to(DEVICE)
    with torch.no_grad():
        recon = model(xb)
        loss = nn.MSELoss()(recon, xb).item()
        
    original = xb[0].cpu().numpy()
    reconstructed = recon[0].cpu().numpy()
    
    if original.shape[0] != len(feat_cols):
        original = original.T
        reconstructed = reconstructed.T
        
    cols = 3
    rows = math.ceil(len(feat_cols) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows), constrained_layout=True)
    fig.suptitle(f"CAE Reconstruction (Loss: {loss:.5f})", fontsize=16)
    
    if isinstance(axes, np.ndarray): axes = axes.flatten()
    else: axes = [axes]

    for i, col_name in enumerate(feat_cols):
        ax = axes[i]
        ax.plot(original[i], color="gray", alpha=0.8, label="Original")
        ax.plot(reconstructed[i], color="red", linestyle="--", label="Recon")
        ax.set_title(col_name)
        if i == 0: ax.legend()
    
    plt.savefig(save_path)
    plt.close(fig)
    print(f"✅ Saved visualization to {save_path}")


if __name__ == "__main__":
    # 1. 환경 설정 및 시드 고정
    set_seed(42, deterministic=True)

    current_file_path = os.path.abspath(__file__)
    code_dir = os.path.dirname(current_file_path)
    project_root = os.path.dirname(code_dir)
    DATA_DIR = os.path.join(project_root, "CMAPSSData")
    RESULTS_DIR = os.path.join(project_root, "results")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ----------------------------------
    print(f"[Environment] {DEVICE}", end="")
    if DEVICE.type == "cuda":
        print(f" | {torch.cuda.get_device_name(0)}")
    else:
        print()

    # --- Step 1: Data load & basic prep ---
    print(f"\n[Step 1] Loading data from: {DATA_DIR}")

    cols = ["unit_number", "time_in_cycles"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]

    train_path = os.path.join(DATA_DIR, "train_FD001.txt")
    test_path = os.path.join(DATA_DIR, "test_FD001.txt")
    rul_path = os.path.join(DATA_DIR, "RUL_FD001.txt")

    if not os.path.exists(train_path):
        print(f"Error: '{train_path}' 파일을 찾을 수 없습니다. CMAPSSData 폴더를 확인하세요.")
        exit(1)
    if not os.path.exists(test_path):
        print(f"Error: '{test_path}' 파일을 찾을 수 없습니다. CMAPSSData 폴더를 확인하세요.")
        exit(1)

    train_df_raw = pd.read_csv(train_path, sep=r"\s+", header=None, names=cols)
    test_df_raw = pd.read_csv(test_path, sep=r"\s+", header=None, names=cols)

    max_cyc = train_df_raw.groupby("unit_number")["time_in_cycles"].max().rename("max_cycles")
    train_df_raw = train_df_raw.merge(max_cyc, on="unit_number").sort_values(["unit_number", "time_in_cycles"])
    train_df_raw["RUL_raw_unclipped"] = train_df_raw["max_cycles"] - train_df_raw["time_in_cycles"]

    FEATURE_COLS = [f"s{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
    
    print(f"  Train: {len(train_df_raw)} samples, {train_df_raw['unit_number'].nunique()} units")
    print(f"  Test:  {len(test_df_raw)} samples, {test_df_raw['unit_number'].nunique()} units")

    # --- Step 2: k-search Logic ---
    N_SPLITS = 5
    all_units = train_df_raw["unit_number"].unique()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    MODEL_TYPES = ["gru","abgru","bigru","lstm","bilstm"]
    SEEDS = [0, 1, 2, 3, 4]

    # K_CANDIDATES = np.arange(0.0, 3.1, 0.1).round(1).tolist()
    K_CANDIDATES = [1.6] 

    best_k_per_model: Dict[str, float] = {}
    
    # === 후보 개수에 따른 분기 처리 ===
    if len(K_CANDIDATES) > 1:
        print("\n[Step 2] Finding best 'k' for each model via 5-Fold Cross-Validation...")
        print("          CRITERIA: RMSE (LATE) minimizing on ALL validation data")
        
        perf: Dict[float, Dict[str, Dict[str, List[float]]]] = {
            k: {m: {"rmse_all": [], "rmse_late": [], "nasa": []} for m in MODEL_TYPES}
            for k in K_CANDIDATES
        }

        # K-Fold 수행
        for fold, (tr_idx, va_idx) in enumerate(kf.split(all_units), 1):
            tr_units = all_units[tr_idx]
            va_units = all_units[va_idx]

            tr_df_fold = train_df_raw[train_df_raw["unit_number"].isin(tr_units)].copy()
            va_df_fold = train_df_raw[train_df_raw["unit_number"].isin(va_units)].copy()

            scaler_fold = StandardScaler().fit(tr_df_fold[FEATURE_COLS])
            apply_scaler_inplace(tr_df_fold, FEATURE_COLS, scaler_fold)
            apply_scaler_inplace(va_df_fold, FEATURE_COLS, scaler_fold)

            fold_tag = f"FOLD={fold}"
            cae_seed = 1000 + fold 

            print(f"\n>> [K-FOLD] {fold_tag} | Pretraining CAE once... (seed={cae_seed})")

            # CAE Pretrain
            encoder_state_fold = pretrain_cae(
                tr_df_fold,
                FEATURE_COLS,
                window_size=30,
                epochs=25,
                batch_size=512,
                tag=fold_tag,
                seed=cae_seed,
                do_visualize=True,                  
                save_path=os.path.join(RESULTS_DIR, f"cae_result_{fold_tag}.png")
            )

            # k Search
            for k in K_CANDIDATES:
                phi_params = {"type": "exp_concave", "CAP": 125.0, "k": float(k)}
                print(f"   - Evaluating k={k:.2f} on {fold_tag}")

                for seed in SEEDS:
                    for m in MODEL_TYPES:
                        exp_tag = f"KCV | {fold_tag} | k={k:.2f} | model={m} | seed={seed}"
                        out = run_single_experiment(
                            tr_df_fold, va_df_fold, pd.DataFrame(),
                            FEATURE_COLS, encoder_state_fold,
                            m, phi_params,
                            window_size=30, lr=1e-3, epochs=50, patience=10,
                            cap_eval_val=125.0, do_test=False, exp_tag=exp_tag, seed=seed,
                            val_mode="all",
                        )
                        perf[k][m]["rmse_all"].append(out["best_val_rmse_all"])
                        perf[k][m]["rmse_late"].append(out["best_val_rmse_late"])
                        perf[k][m]["nasa"].append(out["best_val_nasa"])

        # 결과 집계
        print("\n[Step 2 Completed] Aggregating Results (By Late RMSE)...")
        for m in MODEL_TYPES:
            print(f"\n--- {m.upper()} Results ---")
            summary_rows = []
            for k in sorted(K_CANDIDATES):
                rmse_vals = np.array(perf[k][m]["rmse_late"], dtype=float)
                nasa_vals = np.array(perf[k][m]["nasa"], dtype=float)
                summary_rows.append({
                    "k": k,
                    "rmse_mean": float(rmse_vals.mean()) if rmse_vals.size else float("nan"),
                    "rmse_std": float(rmse_vals.std()) if rmse_vals.size else float("nan"),
                    "nasa_mean": float(nasa_vals.mean()) if nasa_vals.size else float("nan"),
                })
            df_res = pd.DataFrame(summary_rows)
            df_sorted = df_res.sort_values(by=["rmse_mean"], ascending=True)

            best_k = float(df_sorted.iloc[0]["k"])
            best_k_per_model[m] = best_k
            print(df_sorted.head(3))
            print(f"-> Best k (Late RMSE): {best_k}")
    else:
        # 후보가 1개 이하일 경우 K-Fold 스킵
        print(f"\n[Step 2] Skipping K-Fold CV (Only 1 candidate: {K_CANDIDATES})")
        print("          Use the single candidate directly for Final Training.")
        
        fixed_k = float(K_CANDIDATES[0])
        for m in MODEL_TYPES:
            best_k_per_model[m] = fixed_k

    # --- Step 3: Final Training & Testing ---
    print("\n[Step 3] Final training & testing...")
    final_train_units, final_val_units = train_test_split(all_units, test_size=0.1, random_state=42)

    tr_final_raw = train_df_raw[train_df_raw["unit_number"].isin(final_train_units)].copy()
    va_final_raw = train_df_raw[train_df_raw["unit_number"].isin(final_val_units)].copy()
    te_final_raw = test_df_raw.copy()

    final_scaler = StandardScaler().fit(tr_final_raw[FEATURE_COLS])
    apply_scaler_inplace(tr_final_raw, FEATURE_COLS, final_scaler)
    apply_scaler_inplace(va_final_raw, FEATURE_COLS, final_scaler)
    apply_scaler_inplace(te_final_raw, FEATURE_COLS, final_scaler)

    final_encoder_state = pretrain_cae(
        tr_final_raw, FEATURE_COLS, window_size=30, epochs=25, batch_size=512,
        tag="FINAL", seed=4242,
        do_visualize=True,                  
        save_path=os.path.join(RESULTS_DIR, "cae_result_FINAL.png")
    )
    
    final_results = []
    for m in MODEL_TYPES:
        # 1. Baseline
        base_res = run_single_experiment(
            tr_final_raw, va_final_raw, te_final_raw, FEATURE_COLS, final_encoder_state,
            m, {"type": "baseline", "CAP": 125.0},
            window_size=30, lr=1e-3, epochs=50, patience=12,
            cap_eval_val=125.0, do_test=True,
            exp_tag=f"FINAL|Base|{m}", seed=42, val_mode="all"
        )
        final_results.append(base_res)

        # 2. Concave (Best k 사용)
        bk = best_k_per_model[m]
        conc_res = run_single_experiment(
            tr_final_raw, va_final_raw, te_final_raw, FEATURE_COLS, final_encoder_state,
            m, {"type": "exp_concave", "CAP": 125.0, "k": bk},
            window_size=30, lr=1e-3, epochs=50, patience=12,
            cap_eval_val=125.0, do_test=True,
            exp_tag=f"FINAL|Conc|{m}", seed=42, val_mode="all"
        )
        final_results.append(conc_res)

    # --- Step 4: Report Final Results ---
    print("\n" + "=" * 80)
    print(" FINAL TEST RESULTS ".center(80, "="))
    print("=" * 80)

    for model_type in MODEL_TYPES:
        print(f"\n{'=' * 80}")
        print(f" {model_type.upper()} ".center(80, "="))
        print(f"{'=' * 80}")

        baseline = [r for r in final_results if r["model"] == model_type and "Baseline" in r["label"]][0]
        concave = [r for r in final_results if r["model"] == model_type and "Exp-Concave" in r["label"]][0]

        print("\n[Baseline]")
        print(f"  Uncapped: RMSE={baseline['rmse_test']:.3f}, MAE={baseline['mae_test']:.3f}, S={baseline['s_test']:.0f}")
        print(
            f"  CAP=125:  RMSE={baseline['rmse_test_cap125']:.3f}, "
            f"MAE={baseline['mae_test_cap125']:.3f}, S={baseline['s_test_cap125']:.0f}"
        )

        print(f"\n[{concave['label']}]")
        print(f"  Uncapped: RMSE={concave['rmse_test']:.3f}, MAE={concave['mae_test']:.3f}, S={concave['s_test']:.0f}")
        print(
            f"  CAP=125:  RMSE={concave['rmse_test_cap125']:.3f}, "
            f"MAE={concave['mae_test_cap125']:.3f}, S={concave['s_test_cap125']:.0f}"
        )

        def ok(x: float) -> str:
            return "✅" if x > 0 else "❌"

        rmse_imp_uncap = (baseline["rmse_test"] - concave["rmse_test"]) / baseline["rmse_test"] * 100
        mae_imp_uncap = (baseline["mae_test"] - concave["mae_test"]) / baseline["mae_test"] * 100
        s_imp_uncap = (baseline["s_test"] - concave["s_test"]) / baseline["s_test"] * 100

        rmse_imp_cap = (baseline["rmse_test_cap125"] - concave["rmse_test_cap125"]) / baseline["rmse_test_cap125"] * 100
        mae_imp_cap = (baseline["mae_test_cap125"] - concave["mae_test_cap125"]) / baseline["mae_test_cap125"] * 100
        s_imp_cap = (baseline["s_test_cap125"] - concave["s_test_cap125"]) / baseline["s_test_cap125"] * 100

        print("\nImprovements (Original / @CAP=125)")
        print(f"  RMSE : {ok(rmse_imp_uncap)} {rmse_imp_uncap:+.2f}%  |  {ok(rmse_imp_cap)} {rmse_imp_cap:+.2f}%")
        print(f"  MAE  : {ok(mae_imp_uncap)} {mae_imp_uncap:+.2f}%  |  {ok(mae_imp_cap)} {mae_imp_cap:+.2f}%")
        print(f"  NASA : {ok(s_imp_uncap)} {s_imp_uncap:+.2f}%  |  {ok(s_imp_cap)} {s_imp_cap:+.2f}%")

    # --- Step 5: Best Overall (Concave @CAP=125 기준) ---
    print("\n" + "=" * 80)
    print(" BEST OVERALL MODEL ".center(80, "="))
    print("=" * 80)

    concave_results = [r for r in final_results if "Exp-Concave" in r["label"]]
    best_concave = min(concave_results, key=lambda x: x["rmse_test_cap125"])

    best_baseline = [r for r in final_results if r["model"] == best_concave["model"] and "Baseline" in r["label"]][0]

    print(f"\n🏆 Best Model: {best_concave['model'].upper()} with {best_concave['label']}")
    print(f"    Test RMSE (CAP=125): {best_concave['rmse_test_cap125']:.3f}")
    print(f"    vs Baseline: {best_baseline['rmse_test_cap125']:.3f}")

    improvement = (
        (best_baseline["rmse_test_cap125"] - best_concave["rmse_test_cap125"])
        / best_baseline["rmse_test_cap125"]
        * 100
    )
    print(f"    Improvement: {improvement:+.2f}%")

    # --- Step 6: Detailed analysis for best model & Saving ---
    print("\n[Step 6] Detailed analysis for best model & Saving")

    X_te_final, _ = build_test_windows(te_final_raw, FEATURE_COLS, window_size=30)
    
    if os.path.exists(rul_path):
        y_true = pd.read_csv(rul_path, sep=r"\s+", header=None).values.ravel().astype(float)

        baseline_pred_raw = ensemble_predict(
            [best_baseline["model_obj"]], X_te_final,
            [best_baseline["inv_phi"]], cap_at=1e9
        )
        concave_pred_raw = ensemble_predict(
            [best_concave["model_obj"]], X_te_final,
            [best_concave["inv_phi"]], cap_at=1e9
        )

        baseline_pred_cap = np.minimum(baseline_pred_raw, 125.0)
        concave_pred_cap = np.minimum(concave_pred_raw, 125.0)

        print("\n[Step 6b] Saving final predictions to file...")
        try:
            k_to_save = best_k_per_model.get(best_concave["model"], None)
            if k_to_save is None:
                k_to_save = float(best_concave.get("phi_params", {}).get("k", 1.6))

            save_npz_path = os.path.join(RESULTS_DIR, "final_predictions_for_plotting.npz")

            np.savez(
                save_npz_path,
                y_true=y_true,
                baseline_pred=baseline_pred_cap,
                concave_pred=concave_pred_cap,
                model_tag=np.array(best_concave["model"].upper()),
                k_used=np.array(k_to_save)
            )
            print(f" ✓ Final predictions saved to '{save_npz_path}'")
        except Exception as e:
            print(f" [!] Failed to save predictions: {e}")
    else:
        print(f"[Warning] True RUL file not found at {rul_path} (skip detailed analysis)")

    print("\n✅ All Done")
