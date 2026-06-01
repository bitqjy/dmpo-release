# Decoupled MeanFlow on ReFlow

This note summarizes the DMF migration implemented in this repo.

## What DMF Adds

Decoupled MeanFlow treats a flow model as a flow map:

```text
x_r = x_t + (r - t) u_theta(x_t, t, r, cond)
```

The key architectural idea from the DMF paper is to keep the encoder conditioned on the current time
`t`, while conditioning the decoder on the next time `r`.  For ReinFlow MLP policies, the split is:

- encoder: observation encoder, current-time embedding, and the first velocity-head layer
- decoder: the remaining velocity-head layers plus a zero-initialized `r` projection

The zero initialization means an existing ReFlow checkpoint can be loaded into the DMF network with
`strict=False`; before fine-tuning, the new `r` path has no effect.

## Training Objective

`model.flow.dmf.DecoupledMeanFlow` uses two independent samples per batch, following DMF:

- boundary flow matching: `u(x_t, t, t)` predicts the ReFlow velocity `x_data - noise`
- flow-map loss: `u(x_t, t, r)` predicts the average velocity target
  `v + (r - t) du/dt`, with `du/dt` computed by JVP

Unlike the image DMF code, this implementation keeps ReinFlow's noise-to-data convention:

```text
x_t = t * x_data + (1 - t) * noise
t = 0 noise, t = 1 data
```

## Files

- `model/flow/mlp_dmf.py`: low-dimensional and image-conditioned decoupled networks
- `model/flow/dmf.py`: DMF loss and sampler
- `agent/pretrain/train_dmf_agent.py`: pretraining agent
- `agent/eval/eval_dmf_img_agent.py`: robomimic image eval agent
- `cfg/robomimic/pretrain/lift/pre_dmf_mlp_img.yaml`: example training config
- `cfg/robomimic/eval/lift/eval_dmf_mlp_img.yaml`: example evaluation config

## Warm-start From ReFlow

Set `base_policy_path` in the DMF pretrain config to a trained ReFlow checkpoint:

```yaml
base_policy_path: /path/to/reflow/checkpoint/state_2000.pt
resume_training_state: false
load_strict: false
load_optimizer: false
```

For decoder-only fine-tuning, also set:

```yaml
model:
  freeze_encoder: true
```

For best one-step behavior, joint fine-tuning is usually preferable after the warm start.
