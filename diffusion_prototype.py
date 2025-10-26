import numpy as np

# ----------------------
# Hyperparams (Octo small)
# ----------------------
time_dim = 32
hidden_dim = 256
num_blocks = 3
action_horizon = 4
action_dim = 7
A_flat = action_horizon * action_dim  # 28
obs_dim = 384  # transformer pooled embedding dim
diffusion_steps = 20

# ----------------------
# Utilities
# ----------------------
def swish(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = np.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0.0, 0.999)
    return betas

betas = cosine_beta_schedule(diffusion_steps)
alphas = 1.0 - betas
alpha_hats = np.cumprod(alphas)  # length T

# ----------------------
# Parameter initialization (random for demo)
# All Dense layers represented by (W, b) where out = x @ W.T + b
# ----------------------
rng = np.random.RandomState(0)

# FourierFeatures kernel: shape (time_dim//2, 1)
ff_kernel = rng.normal(0, 0.2, size=(time_dim // 2, 1))  # learnable

# cond_encoder MLP: input 32 -> 64 -> 32
W_cond_1 = rng.normal(0, 0.02, size=(64, time_dim))  # 32 -> 64
b_cond_1 = np.zeros((64,))
W_cond_2 = rng.normal(0, 0.02, size=(time_dim, 64))  # 64 -> 32
b_cond_2 = np.zeros((time_dim,))

# reverse_network initial proj: 444 -> 256
in_dim = time_dim + obs_dim + A_flat  # 32 + 384 + 28 = 444
W_init = rng.normal(0, 0.02, size=(hidden_dim, in_dim))
b_init = np.zeros((hidden_dim,))

# ResNet blocks parameters (per block)
W_block_1 = [rng.normal(0, 0.02, size=(hidden_dim * 4, hidden_dim)) for _ in range(num_blocks)]
b_block_1 = [np.zeros((hidden_dim * 4,)) for _ in range(num_blocks)]
W_block_2 = [rng.normal(0, 0.02, size=(hidden_dim, hidden_dim * 4)) for _ in range(num_blocks)]
b_block_2 = [np.zeros((hidden_dim,)) for _ in range(num_blocks)]
# If residual projection required we would have W_res but shapes match so skip.

# final out proj 256 -> out_dim (A_flat)
W_out = rng.normal(0, 0.02, size=(A_flat, hidden_dim))
b_out = np.zeros((A_flat,))

# ----------------------
# Module implementations (numpy)
# ----------------------
def fourier_features(x):  
    # x shape: (..., 1)
    # kernel shape: (time_dim//2, 1)
    # f = 2π * x @ kernel.T -> (..., time_dim//2)
    f = 2.0 * np.pi * np.matmul(x, ff_kernel.T)  # broadcasted matmul
    return np.concatenate([np.cos(f), np.sin(f)], axis=-1)  # (..., time_dim)

def cond_encoder_forward(t_ff):
    # t_ff shape: (..., time_dim)
    # layer1: 32 -> 64
    x = np.matmul(t_ff, W_cond_1.T) + b_cond_1  # (..., 64)
    x = swish(x)
    # layer2: 64 -> 32
    x = np.matmul(x, W_cond_2.T) + b_cond_2  # (..., 32)
    return x  # (..., 32)

def mlpresnet_forward(reverse_input):
    # reverse_input shape: (...prefix..., in_dim=444)
    x = np.matmul(reverse_input, W_init.T) + b_init  # (..., 256)
    # blocks
    for i in range(num_blocks):
        residual = x
        # Dense 256 -> 1024
        h = np.matmul(x, W_block_1[i].T) + b_block_1[i]  # (..., 1024)
        h = swish(h)
        # Dense 1024 -> 256
        h = np.matmul(h, W_block_2[i].T) + b_block_2[i]  # (..., 256)
        # residual add
        x = residual + h
    x = swish(x)
    out = np.matmul(x, W_out.T) + b_out  # (..., A_flat)
    return out  # pred_eps

# ----------------------
# Add noise (training) : x_t = sqrt(alpha_hat_t) * x0 + sqrt(1-alpha_hat_t) * eps
# ----------------------
def add_noise(actions, S=1):
    # actions shape: [B, W, H, a]
    # actions_flat shape: [B, W, A_flat]
    actions_flat = actions.reshape(actions.shape[0], actions.shape[1], -1)  # [B, W, 28]
    # sample time indices (S, B, W, 1)
    B, W = actions_flat.shape[:2]
    time = rng.randint(0, diffusion_steps, size=(S, B, W, 1))
    # sample noise (S, B, W, A_flat)
    noise = rng.normal(size=(S, B, W, A_flat)).astype(np.float32)
    alpha_sel = alpha_hats[time.squeeze(-1)]  # shape (S, B, W)
    alpha_sel = alpha_sel[..., None]  # (S,B,W,1)
    scale = np.sqrt(alpha_sel)
    std = np.sqrt(1.0 - alpha_sel)
    noisy = scale * actions_flat[None, ...] + std * noise
    # returns noisy_actions [S, B, W, A_flat], noise, time
    return noisy.astype(np.float32), noise.astype(np.float32), time

# ----------------------
# Training step example (single forward MSE loss)
# ----------------------
def training_step(transformer_embeddings, actions, timestep_pad_mask, action_pad_mask):
    """
    transformer_embeddings: [B, W, D]
    actions: [B, W, H, a]
    timestep_pad_mask: [B, W] boolean (True means valid)
    action_pad_mask: [B, W, H, a] boolean
    """
    # 1) add noise
    S = 1
    noisy_actions, noise, time = add_noise(actions, S=S)  # noisy_actions: [S,B,W,28]
    Sdim = noisy_actions.shape[0]

    # 2) prepare cond and obs
    # embeddings: [B, W, D]
    # replicate embeddings to match S dimension
    embeddings_bcast = np.broadcast_to(embeddings[None, ...], (Sdim,) + embeddings.shape)  # [S,B,W,D]

    # call ScoreActor: need to compute cond_enc from time
    t_ff = fourier_features(time.astype(np.float32))  # time: (S,B,W,1) -> t_ff: (S,B,W,32)
    cond_enc = cond_encoder_forward(t_ff)  # (S,B,W,32)

    # reverse_input = concat([cond_enc, obs_enc_bcast, noisy_actions], axis=-1)
    reverse_input = np.concatenate([cond_enc, embeddings_bcast, noisy_actions], axis=-1)  # (S,B,W,444)

    # pred_eps via reverse network
    pred_eps = mlpresnet_forward(reverse_input)  # (S,B,W,28)

    # compute mask: flatten to (B,W,28) and add S dim -> (S,B,W,28)
    mask = (timestep_pad_mask[:, :, None, None] & action_pad_mask).reshape(actions.shape[0], actions.shape[1], -1)
    mask = mask[None, ...]  # (S,B,W,28)

    # mse over masked entries
    se = (pred_eps - noise) ** 2
    masked_se = se * mask
    loss = masked_se.sum() / (mask.sum() + 1e-8)
    return loss, pred_eps

# ----------------------
# Predict (inference) : DDPM reverse loop (numpy)
# ----------------------
def predict_action(transformer_embeddings, rng_seed=1, sample_shape=()):
    """
    transformer_embeddings: [B, W, D]
    Returns: actions_last_step [*sample_shape, B, H, a]
    """
    np_rng = np.random.RandomState(rng_seed)
    B, W, Ddim = transformer_embeddings.shape

    # initialize noise: shape (*sample_shape, B, W, A_flat)
    sample_count = 1  # for simplicity ignore sample_shape complexity here
    current_x = np_rng.normal(size=(sample_count, B, W, A_flat)).astype(np.float32)

    # pre-broadcasted embeddings: we'll broadcast inside loop
    for t in range(diffusion_steps - 1, -1, -1):
        # build input_time shape (sample_count, B, W, 1)
        input_time = np.full((sample_count, B, W, 1), t, dtype=np.int32)
        # cond
        t_ff = fourier_features(input_time.astype(np.float32))
        cond_enc = cond_encoder_forward(t_ff)  # (sample_count, B, W, 32)
        embeddings_bcast = np.broadcast_to(transformer_embeddings[None, ...], (sample_count, B, W, Ddim))  # (sample_count,B,W,D)
        reverse_input = np.concatenate([cond_enc, embeddings_bcast, current_x], axis=-1)  # (sample_count,B,W,444)
        eps_pred = mlpresnet_forward(reverse_input)  # (sample_count,B,W,28)

        # coefficients
        alpha_t = alphas[t]
        alpha_hat_t = alpha_hats[t]
        alpha_1 = 1.0 / np.sqrt(alpha_t)
        alpha_2 = (1.0 - alpha_t) / np.sqrt(1.0 - alpha_hat_t)

        current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

        # add noise if t > 0
        if t > 0:
            z = np_rng.normal(size=current_x.shape).astype(np.float32)
            current_x = current_x + np.sqrt(betas[t]) * z
        # clip
        current_x = np.clip(current_x, -5.0, 5.0)  # max_action ~5

    # reshape to actions: (sample_count, B, W, H, a)
    actions = current_x.reshape((sample_count, B, W, action_horizon, action_dim))
    # return last window timestep actions for each batch (simplify: last in window)
    return actions[:, :, -1, :, :]  # shape (sample_count, B, H, a)

# ----------------------
# Demo run
# ----------------------
if __name__ == "__main__":
    # demo dims
    B = 1
    W = 1
    # fake transformer embeddings
    embeddings = rng.normal(size=(B, W, obs_dim)).astype(np.float32)
    # fake actions
    actions = rng.normal(size=(B, W, action_horizon, action_dim)).astype(np.float32)
    # masks
    timestep_pad_mask = np.ones((B, W), dtype=bool)
    action_pad_mask = np.ones((B, W, action_horizon, action_dim), dtype=bool)

    loss, pred_eps = training_step(embeddings, actions, timestep_pad_mask, action_pad_mask)
    print("Training step loss:", loss)
    sampled = predict_action(embeddings, rng_seed=42)
    print("Sampled actions shape:", sampled.shape)  