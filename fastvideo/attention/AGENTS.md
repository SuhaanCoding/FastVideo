# `fastvideo/attention/` — Attention Backends

**Generated:** 2026-05-02

Backend registry + selector wrapping FlashAttn / SageAttn / SageAttn3 / SDPA / VSA / VMoBA / SLA / BSA.

## Layout

```
attention/
├── __init__.py            # Exports DistributedAttention, LocalAttention, get_attn_backend
├── layer.py               # DistributedAttention, DistributedAttention_VSA, LocalAttention
├── selector.py            # get_attn_backend (cached) + attention_backend_scope
├── backends/
│   ├── abstract.py        #   AttentionBackend / AttentionMetadata / AttentionMetadataBuilder
│   ├── flash_attn.py      #   FA2/FA3
│   ├── sage_attn.py       #   SageAttention v1
│   ├── sage_attn3.py      #   SageAttention v3
│   ├── sdpa.py            #   torch SDPA fallback
│   ├── video_sparse_attn.py  # VSA (paper: Video Sparse Attention)
│   ├── vmoba.py           #   Video-MoBA
│   ├── sla.py             #   Sliding-window (STA)
│   └── bsa_attn.py        #   Block-sparse
└── utils/
    ├── flash_attn_cute.py
    └── flash_attn_no_pad.py
```

## Selection Order

`get_attn_backend()` reads every selection input, then resolves via, in
precedence order:

1. `global_force_attn_backend(...)` — deprecated process-global override.
2. The active `attention_backend_scope(...)` request (per component).
3. Env-var `FASTVIDEO_ATTENTION_BACKEND` (see `STR_BACKEND_ENV_VAR` in
   `fastvideo/utils.py`) — suppressed while a scope is active unless the
   scope passes `consult_env=True`.
4. The layer-declared `default_backend`.
5. Per-platform automatic selection from `fastvideo/platforms/`, which probes
   the *current device's* capability.

The result is cached on **all** of those inputs (plus component identity and
device index), so a changed request simply lands on a different cache key —
no `cache_clear()` is needed and none should be added.

Callers that need a specific backend for one component (a role model, a
teacher/critic pair, a test) use `attention_backend_scope(backend,
component=...)`: it is process-local, exception-safe, and nestable.
`attention_backend_scope(None)` means "automatic selection, ignore the
process-wide request". Never mutate `FASTVIDEO_ATTENTION_BACKEND`
mid-process.

## Adding a Backend

1. Subclass `AttentionBackend` in `backends/<name>.py`.
2. Implement `AttentionMetadata` + `AttentionMetadataBuilder` for the new path.
3. Register the enum value in `fastvideo/platforms/interface.py` (`AttentionBackendEnum`).
4. Wire string → class resolution in `selector.py`.
5. Verify the new backend works with `DistributedAttention` (sequence parallel)
   and `LocalAttention` (single-rank). If it cannot support SP, document the
   gap in the backend file's module docstring.

## Anti-Patterns

- Calling `torch.nn.functional.scaled_dot_product_attention` directly inside a
  model's forward — go through `DistributedAttention` / `LocalAttention`.
- Reading `os.environ[STR_BACKEND_ENV_VAR]` from arbitrary call sites. Use
  `get_env_variable_attn_backend()`.
- Caching backend instances per-module. The selector cache is process-wide; do
  not duplicate it.
