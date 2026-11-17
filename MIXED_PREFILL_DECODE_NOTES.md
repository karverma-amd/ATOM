# DSR-1 Mixed Prefill/Decode — Status & Resume Notes

Built from this image's own ATOM code (branch lineage `jiaolyu/dsr1-best-20261117`),
NOT derived from ROCm/ATOM PR #2037 or #3460. DSR-1 has no MTP/draft (spec off),
TBO off, so those paths stay inert.

## References
- Feature idea / prior art (DeepSeek-V4.1, do NOT copy — ours is from main/image):
  - ROCm/ATOM PR #2037 — mixed prefill/decode batching with DSpark (V4.1 core):
    https://github.com/ROCm/ATOM/pull/2037
  - ROCm/ATOM PR #3460 — TBO for mixed (V4.1 follow-up; N/A here, TBO off):
    https://github.com/ROCm/ATOM/pull/3460
- Image: `rocm/mlperf-inference:gfx1250-atom-20261117-dsr1` (README at
  `/opt/dsr1_best_20261117/README.md` inside the image). fusion B off
  (`ATOM_MLA_VUP_OPROJ_QUANT_FUSE=0`), 16-loop perf config
  `offline_mi455x_atom_team_test_perf.yaml` (70,208 samples).
- Why V4.1 graphs and we (initially) don't: V4.1's backend sets
  `supports_token_graph=True` + `build_for_token_graph_capture`; DSR-1 uses
  `AiterMLAMetadataBuilder` (`supports_token_graph=False`). The port below adds it.

## Files changed (all under `atom/`)
- `config.py`, `model_engine/arg_utils.py` — `enable_mixed_prefill_decode` flag.
- `model_engine/scheduler.py` — `_augment_with_decodes` (mixed batch formation).
- `utils/forward_context.py` — ForwardMode/Context mixed fields + `graphable`.
- `model_engine/model_runner.py` — `_build_mixed_metadata`, `_clone_attn_metadata`,
  `_stage_input_ids` mixed branch.
- `model_ops/attention_mla.py` — `MLAAttention._forward_mixed` (two-pass).
- `model_ops/embed_head.py` — mixed LM-head gather.
- `model_ops/attentions/aiter_mla.py` — `build_for_token_graph_capture` +
  `supports_token_graph` (token-graph port, currently disabled).

## Config knobs
- Flag: `enable_mixed_prefill_decode` (default False). Plumbed yaml → `EngineArgs`
  (`arg_utils.py`) → `Config` (`config.py`) → `Scheduler`.
- Perf config (verbatim README 16-loop + flag): `offline_mi455x_atom_team_test_perf_mixed.yaml`
- Accuracy config (+ flag): `offline_mi455x_atom_team_test_mixed.yaml`
- Smoke config: `offline_mixed_smoke.yaml` (max_num_seqs 256, caps up to 256,
  max_tokens 1500, total_sample_count 1024)
- Run inside container `dsr1_mixed`: `CONFIG=<name> TAG=<t> bash /app/mlperf-recipe-scripts/run_performance.sh`
  (accuracy: `run_accuracy.sh`). Results bind-mounted to `/home/karverma/dsr1_mixed/results`.
- GPU cleanup between runs (never docker stop): kill `main.py` / `python3 -c from` PIDs.

## LIVE STATE: validated eager-mixed (safe, working)
Mixed batches form in the scheduler, run as a single compiled (non-graphed)
forward, attention split into a decode pass + prefill pass. Validated:
- Accuracy inference completed 4388/4388, 0 errors (final exact_match score not
  captured — run was stopped in the eval/scoring phase to prioritize the graph port).
- Smoke: 32,418 tok/s total, 0 errors, mixed batches up to "224 decode + prefill".
- No decode-step regression: mixing only rides already-eager prefill steps; steady-state
  pure-decode steps stay fully CUDA-graphed.

### Key implementation points (eager-mixed)
- `scheduler.py::_augment_with_decodes`: splices running decodes IN FRONT of a
  scheduled prefill (decode-first row order). Two-phase: collect candidates
  (read-only `can_append`), clamp decode-token count to a **multiple of 32**
  (q_scale is block-quantized in 32-token blocks → split point must be 32-aligned),
  then commit (`may_append`). Builds `mixed_decode_batch` + `mixed_prefill_batch`
  sub-batches on the merged batch (`is_mixed=True`).
- `forward_context.py`: ForwardMode/Context carry `is_mixed`, `num_decode_tokens`,
  `num_decode_seqs`, `mixed_decode_md`, `mixed_prefill_md`. `graphable` excludes
  `is_mixed` (mixed runs eager-compiled).
- `model_runner.py::_build_mixed_metadata`: builds decode metadata first, **clones**
  its tensors (shared `forward_vars` buffers), then `h2d_owner.finish()/begin()`
  (second publication epoch, avoids duplicate-publish + host race), re-lays the
  prefill query layout (`publish_query_metadata`), builds prefill metadata,
  writes merged `[decode|prefill]` positions into `forward_vars["positions"]`.
- `model_runner.py::_stage_input_ids`: mixed branch stages prefill prompt tokens +
  runs the **deferred-output** decode fill (`fill_deferred_decode_ids`) for the
  decode tail slots [0:nd] (this config is `is_deferred_out=True`).
- `attention_mla.py::MLAAttention._forward_mixed`: two-pass. Slices q/k_nope/k_rope/
  positions at `nd`; q_scale sliced by block (`nd//32`). Swaps thread-local
  `_fc_module._forward_context_local.ctx` to a per-segment sub-context
  (decode: is_prefill=False/mixed_decode_md; prefill: is_prefill=True/mixed_prefill_md),
  recursively calls `forward_impl`, concatenates `[out_decode, out_prefill]`.
  The attention op `unified_attention_with_output_base` is `@eager_region()` and a
  `torch.ops.aiter.*` custom op → Dynamo-opaque, so the two-pass runs eagerly at runtime.
- `embed_head.py::ParallelLMHead.forward`: mixed gather = decode rows [0:nd] +
  each prefill seq's last token (`cu_seqlens_q[1:]-1+nd`).

## TOKEN-GRAPH PORT (WIP, DISABLED) — the path to CUDA-graphed mixed + uplift
Goal: make a mixed step replay the model's prefill **token graph** (MoE/GEMMs over a
padded token bucket) with attention as the eager region — so decode rows folded in
keep graph speed. This is the mechanism DeepSeek-V4.1 uses (PR #2037 targets V4.1
precisely because its backend has it; DSR-1's `AiterMLAMetadataBuilder` does not).

### DONE (scaffolding present in files)
- `aiter_mla.py`: `build_for_token_graph_capture(tokens)` — builds a synthetic
  native prefill (requests of one KV block each, page 0, `num_cached_tokens` as a
  LIST, `block_tables` as tuple-of-tuples) and returns `(attn_metadata, context)`.
  **CONFIRMED WORKING**: "Token graphs recorded for prefill buckets (256,512,1024,2048,3072)".
- `supports_token_graph` flag on `AiterMLAMetadataBuilder` (currently **False**).
- The attention `@eager_region()` is already present (base_attention.py:1008) and
  at replay re-runs with weak-ref tensors at fixed buffer addresses — so the mixed
  two-pass + merged-positions-into-`forward_vars["positions"]` design is compatible.

### BLOCKER (what wedged the enabled run)
Flipping `supports_token_graph = True` makes **all** prefill steps replay token graphs.
Result: engine wedged — GPU 100%, **prompt+gen throughput 0.0**, 256 reqs stuck
"Running", KV ~2.8%, prefills fire but emit no tokens. Capture is fine; **replay
produces no output**. Root cause not yet found — likely the model `@token_graph`
replay (`deepseek_v2.py:2677`) interaction with the LM-head/sampling/output-commit
or a scheduling loop for AiterMLA. ETA to finish: ~1–3+ hrs (4-min iteration cycles).

### TO RESUME THE PORT
1. `aiter_mla.py`: set `supports_token_graph = True`.
2. `forward_context.py::decide`: drop `or is_mixed` from the `graphable` line.
3. `scheduler.py::_augment_with_decodes`: re-add the token-bucket cap (only mix when
   `total_tokens_num_prefill + num_new_tokens <= max(TokenGraph.captured_sizes("prefill"))`;
   `from atom.utils.region_graph import TokenGraph`). Guarantees every mixed step is
   graph-replayable (no decode regression).
4. `model_runner.py::_build_mixed_metadata`: pad the prefill segment to the bucket
   (`bucket = int(forward_mode.running_tokens)`, prefill running_tokens = `bucket - nd_tokens`,
   write merged positions over `[:bucket]`, re-lay prefill query at `prefill_running_tokens`).
5. Debug the 0-output replay wedge (item BLOCKER). Start with: is the LM head / sampling
   run after a token-graph prefill replay, and are output tokens committed? Check how
   `@token_graph` on the model replays and whether `_is_pure_middle_chunk`/`produces_output`
   or the deferred-output path is mis-gated for token-graph prefills.
6. Smoke (`CONFIG=offline_mixed_smoke`), watch for "Scheduled MIXED" + requests finishing
   + nonzero generation throughput. Then accuracy gate + baseline-vs-mixed offline tokens/s.

## Note on expected uplift
Offline throughput uplift from mixing is inherently small on DSR-1 because mixing only
occurs during the admission ramp (steady state is pure decode). The feature's larger
benefit is serving/interactive **TPOT** (decodes not stalled by prefills). The
token-graph port mainly removes the eager penalty on the folded-in decode rows.

## Config yamls (recreate on a new node — they live in the container, not this repo)
The mixed yamls are derived from the baked recipe configs in
`/lab-mlperf-inference/code/deepseek-r1/` (bind-populated from `/app/mlperf-recipe-scripts/code`).
Recreate inside the container:
```bash
cd /lab-mlperf-inference/code/deepseek-r1
# perf (verbatim 16-loop + only the mixed flag)
cp offline_mi455x_atom_team_test_perf.yaml offline_mi455x_atom_team_test_perf_mixed.yaml
sed -i '/^  enable_chunked_prefill: False/a\  enable_mixed_prefill_decode: True' offline_mi455x_atom_team_test_perf_mixed.yaml
# accuracy (+ flag)
cp offline_mi455x_atom_team_test.yaml offline_mi455x_atom_team_test_mixed.yaml
sed -i '/^  enable_chunked_prefill: False/a\  enable_mixed_prefill_decode: True' offline_mi455x_atom_team_test_mixed.yaml
# fast smoke (+ flag, small): start from perf_mixed then shrink
cp offline_mi455x_atom_team_test_perf_mixed.yaml offline_mixed_smoke.yaml
sed -i -E 's/^  max_num_seqs: .*/  max_num_seqs: 256/; s/^  cudagraph_capture_sizes: .*/  cudagraph_capture_sizes: "[1,2,4,8,16,32,64,128,256]"/; s/^  max_tokens: .*/  max_tokens: 1500/; s/^  total_sample_count: .*/  total_sample_count: 1024/' offline_mixed_smoke.yaml
```
Run: `CONFIG=<name> TAG=<t> bash /app/mlperf-recipe-scripts/run_performance.sh`
(accuracy: `run_accuracy.sh`). Watch log for `Scheduled MIXED batch`.

## Transfer to another node
This repo lives on the host at `/home/karverma/dsr1_mixed/ATOM`, bind-mounted into
container `dsr1_mixed` at `/app/ATOM` (edit on host, imported live by Python).
Work is committed to branch `karverma/dsr1-mixed-prefill-decode` and pushed to the
`karverma-amd/ATOM` fork. On the new node:
```bash
git clone https://github.com/karverma-amd/ATOM.git && cd ATOM
git checkout karverma/dsr1-mixed-prefill-decode
# bind-mount this dir to /app/ATOM in the rocm/mlperf-inference:gfx1250-atom-20261117-dsr1 container
```
Then follow "TO RESUME THE PORT" above (flip `supports_token_graph`, re-enable the 3
hooks, debug the 0-output replay wedge), or just use the validated eager-mixed path as-is.
