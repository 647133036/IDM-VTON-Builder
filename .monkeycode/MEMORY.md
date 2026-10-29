# User Instruction Memory

This file records user instructions, preferences, and teachings for reference in future interactions.

## Format

### User Instruction Entry
User instruction entries should follow this format:

[User Instruction Summary]
- Date: [YYYY-MM-DD]
- Context: [Mentioned scenario or time]
- Instructions:
  - [Content of user teaching or instruction, described line by line]

### Project Knowledge Entry
Entries discovered by the Agent during task execution should follow this format:

[Project Knowledge Summary]
- Date: [YYYY-MM-DD]
- Context: Discovered by Agent while performing [specific task description]
- Category: [Operations & Deployment|Build Methods|Testing Methods|Troubleshooting & Debugging|Workflow & Collaboration|Environment Configuration]
- Instructions:
  - [Specific knowledge points, described line by line]

## Deduplication Strategy
- Before adding a new entry, check for similar or identical instructions.
- If a duplicate is found, skip the new entry or merge it with the existing one.
- When merging, update the context or date information.
- This helps avoid redundant entries and keeps the memory file tidy.

## Entries

[Project Knowledge Summary]
- Date: 2026-10-27
- Context: Discovered by Agent while closing the second-round security review
- Category: Workflow & Collaboration
- Instructions:
  - Before pushing, all four checks must be green, with a baseline of 99 pytest +
    11 vitest and a clean ruff run:
      * `python3 -m ruff check backend/ training/` (from the repo root)
      * `python3 -m ruff format --check backend/ training/` (from the repo root)
      * `python3 -m pytest tests/ -q` (run from the `backend/` directory)
      * `npm test` (from the repo root; needs `npm install` first)
  - `npm run build` (from the repo root) MUST also pass before pushing: it is the only
    check that catches JS module syntax errors. `npm test` only covers `src/i18n.js`,
    not `src/main.js`, so a stray paren in `main.js` will not fail `npm test` but will
    blank out the SPA (vite dev server logs it, but nothing surfaces that log).
  - `ruff` only sees the repo layout when both `backend/` and `training/` are passed as
    explicit path arguments from the repo root; `training/` does not exist under `backend/`.
  - Security fixes must ship with a regression test that fails before the fix is applied
    (requestor requirement); add it to `backend/tests/test_api.py` unless the layer differs.

[Project Knowledge Summary]
- Date: 2026-10-28
- Context: Discovered by Agent while debugging IP-Adapter on the RTX 4090 (diffusers 0.42.0, SDXL ControlNet pipeline)
- Category: Troubleshooting & Debugging
- Instructions:
  - IP-Adapter Plus (`ip-adapter-plus_sdxl_vit-h`) is incompatible with diffusers 0.42:
    the plus `image_proj` returns a tuple the attention processor cannot consume, and
    plus needs last_hidden (patch) features while the pipeline's `encode_image` uses
    pooled embeddings. Use the NON-plus `ip-adapter_sdxl_vit-h` instead; its `image_proj`
    is a simple MLP (`image_proj.norm`+`image_proj.proj`), works with the h94 `models/
    image_encoder` (hidden 1280, proj 1024).
  - `enable_attention_slicing()` MUST NOT be called when IP-Adapter is loaded:
    `SlicedAttnProcessor` replaces the IPAdapter processors and crashes with
    `AttributeError: 'tuple' object has no attribute 'shape'` because `unet_2d_condition`
    (ip_image_proj branch) passes `encoder_hidden_states` as a `(text, image)` tuple.
  - `run_test.sh`/`run_app.sh` need `export HF_ENDPOINT=https://hf-mirror.com` first, and
    all local component loads use `local_files_only=True` to avoid HF network hangs.
  - The IP-Adapter weight dir is `app/models/sdxl/ip_adapter/`: weight file plus an
    `image_encoder/` (config.json + model.safetensors) must both exist for it to load.
