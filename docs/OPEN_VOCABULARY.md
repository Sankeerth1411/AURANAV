# Open-vocabulary discovery (experimental)

AURANAV now has an optional secondary discovery channel backed by Ultralytics YOLO-World. It can be configured with a list of text prompts such as open drain, manhole, broken pavement, curb, construction debris, low-hanging obstacle, and staircase.

## Enable it

1. Install the repository's pinned/compatible Ultralytics dependencies in a separate environment.
2. Add or edit prompts in the `open_vocabulary` section of `config.yaml`.
3. Set `enabled: true`.
4. Start with CPU mode and a small prompt list. The first model load may download weights; ensure the file is available locally before expecting fully offline operation.
5. Benchmark latency, memory, temperature, power and false positives on the target computer.

The default configuration is disabled. When enabled, the discovery model runs every N frames (default 10) to limit compute. It suppresses boxes substantially overlapping the primary detector and draws remaining candidates with a separate purple `DISCOVERY` label.

## Strict separation from safety alerts

Discovery candidates are **not** passed into the primary tracker, hazard scorer or speech system. They are not proof that the named object is present. Text-prompt detectors can miss hazards, hallucinate labels, or confuse visually similar surfaces. A discovery candidate must be checked against a labelled validation set and human review before any integration into navigation or emergency alerts.

Do not describe this feature as recognition of every object. It is a prompt-conditioned proposal channel, not an exhaustive open-world perception system.

## Suggested evaluation

- Create a held-out dataset containing both prompted objects and hard negatives.
- Report per-prompt precision, recall, AP, false positives per minute and missed-hazard rate.
- Compare the same clips with the primary detector alone and with discovery enabled.
- Measure total end-to-end latency and sustained device thermals/power.
- Keep the primary detector and its alerts operational if discovery fails or is disabled.

The model weights are not committed to the repository. For a genuinely offline deployment, pre-download the exact compatible weights and verify inference with network access disabled.
