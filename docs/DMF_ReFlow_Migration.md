# Decoupled MeanFlow on ReFlow

This note summarizes the DMF migration implemented in this repo.

## What DMF Adds

Decoupled MeanFlow treats a flow model as a flow map:

```text
x_r = x_t + (r - t) u_theta(x_t, t, r, cond)
```

The key architectural idea from the DMF paper is to keep the encoder conditioned on the current time
`t`, while conditioning the decoder on the next time `r`.  This repo now applies that idea on DiT
action-token blocks:

- ReFlow-DiT: all DiT blocks and the final layer are conditioned on `t`
- DMF-DiT: the first `dmf_depth` blocks are conditioned on `t`; the remaining blocks and final layer
  are conditioned on `r`

`VisionActionDiT` and `DecoupledVisionActionDiT` share the same base parameter names.  DMF-DiT adds
the official-release stability pieces used during fine-tuning: qk-normalized attention, a learned
log-variance head for logvar-weighted loss, `torch.func.jvp`, and gradient clipping.

The repo also includes a DMF+iMF-target variant, `DecoupledImprovedMeanFlow`.  It keeps the DMF
decoupled DiT architecture and boundary FM loss, but trains the flow-map branch with the iMF v-loss:
the JVP tangent is the predicted boundary velocity `u(x_t, t, t)`, and the `du/dt` term is
stop-grad'ed before regressing back to the instantaneous ReFlow velocity.

## Training Objective

`model.flow.dmf.DecoupledMeanFlow` uses two independent samples per batch, following DMF:

- boundary flow matching: `u(x_t, t, t)` predicts the ReFlow velocity `x_data - noise`
- flow-map loss: `u(x_t, t, r)` predicts the average velocity target
  `v + (r - t) du/dt`, with `du/dt` computed by JVP
- DMF+iMF-target loss: `u(x_t, t, r) - (r - t) stopgrad(du/dt)` predicts the ReFlow velocity, with
  JVP tangent `u(x_t, t, t)`
- by default, both losses use the official-style logvar weighting instead of raw MSE

Unlike the image DMF code, this implementation keeps ReinFlow's noise-to-data convention:

```text
x_t = t * x_data + (1 - t) * noise
t = 0 noise, t = 1 data
```

## Files

- `model/flow/dit.py`: low-dimensional and image-conditioned ReFlow-DiT / DMF-DiT networks
- `model/flow/dmf.py`: DMF loss and sampler
- `agent/pretrain/train_dmf_agent.py`: pretraining agent
- `agent/eval/eval_dmf_img_agent.py`: robomimic image eval agent
- `cfg/robomimic/pretrain/lift/pre_reflow_dit_img.yaml`: lift ReFlow-DiT training config
- `cfg/robomimic/eval/lift/eval_reflow_dit_img.yaml`: lift ReFlow-DiT evaluation config
- `cfg/robomimic/pretrain/transport/pre_reflow_dit_img.yaml`: transport ReFlow-DiT training config
- `cfg/robomimic/eval/transport/eval_reflow_dit_img.yaml`: transport ReFlow-DiT evaluation config
- `cfg/robomimic/pretrain/lift/pre_dmf_dit_img.yaml`: lift DMF-DiT training config
- `cfg/robomimic/eval/lift/eval_dmf_dit_img.yaml`: lift DMF-DiT evaluation config
- `cfg/robomimic/pretrain/transport/pre_dmf_dit_img.yaml`: transport DMF-DiT training config
- `cfg/robomimic/eval/transport/eval_dmf_dit_img.yaml`: transport DMF-DiT evaluation config
- `cfg/robomimic/pretrain/lift/pre_dmf_imf_dit_img.yaml`: lift DMF-DiT with iMF target
- `cfg/robomimic/eval/lift/eval_dmf_imf_dit_img.yaml`: lift DMF+iMF-target evaluation config
- `cfg/robomimic/pretrain/transport/pre_dmf_imf_dit_img.yaml`: transport DMF-DiT with iMF target
- `cfg/robomimic/eval/transport/eval_dmf_imf_dit_img.yaml`: transport DMF+iMF-target evaluation config

## Warm-start From ReFlow

Set `base_policy_path` in the DMF pretrain config to a trained ReFlow-DiT checkpoint:

```yaml
base_policy_path: /path/to/reflow/checkpoint/state_2000.pt
resume_training_state: false
load_strict: false
load_optimizer: false
```

`load_strict: false` is intentional: DMF-DiT may add `logvar_linear` and qk-norm parameters that are
not present in an older ReFlow-DiT checkpoint.  The shared ReFlow weights still warm-start normally.

For decoder-only fine-tuning, also set:

```yaml
model:
  freeze_encoder: true
```

For best one-step behavior, joint fine-tuning is usually preferable after the warm start.
