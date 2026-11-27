# AURANAV safety architecture and validation notes

## What the current model can recognise

The current detector is a **closed-set YOLO model** trained on the classes listed in the dataset configuration. It cannot reliably name every object in the world. Objects outside its training labels may be missed or assigned an incorrect known label. The saved evaluation report is the baseline: mAP@0.5 0.6101, mAP@0.5:0.95 0.4123, precision 0.6373, recall 0.5816, and CPU benchmark 11.13 FPS on the machine used for that run. Per-class performance varies widely; some obstacle classes are weak. Do not describe this model as universally accurate.

## Changes in this upgrade branch

- Tracking now associates detections only when their class IDs match.
- Invalid/non-finite boxes and invalid confidence values are rejected.
- Unconfirmed tracks are withheld until the configured hit count is reached.
- Stale tracks are not returned as current hazards, and the default stale lifetime is reduced.
- Hazard scoring combines configured class severity with image position, lower-frame placement, relative box area, recent box-area trend, and confidence.
- Critical alerts can bypass the global speech throttle; normal alerts retain per-track and per-class cooldowns.
- The example environment file no longer contains a credential.

## What the score means (and does not mean)

A hazard score is a **relative alert-priority heuristic**, not a probability of collision. Bounding-box area and change in area are visual cues only. They do not provide metres, calibrated distance, or time-to-collision. Camera pitch, lens field of view, object dimensions, and perspective all affect apparent box size.

For metric proximity, integrate and calibrate a suitable depth sensor (for example, a stereo/depth camera or time-of-flight sensor) and fuse its measurements with the detected obstacle region. Validate sensor range, sunlight performance, reflective/transparent surfaces, latency, and failure behaviour. Never silently substitute an image-size estimate when depth is unavailable.

## Recommended recognition roadmap

1. Improve the existing labelled dataset first: correct class names and missing labels, remove duplicates/leakage, balance rare hazards, and include Indian street scenes, varied lighting, camera angles, weather, and occlusion.
2. Prioritise potholes, broken/uneven road, open drains, steps/curbs, poles, parked/moving vehicles, barriers, and pedestrians. Report per-class precision/recall and false negatives, not only aggregate mAP.
3. Add an optional open-vocabulary model as a **secondary discovery channel**, not as the sole safety detector. Require confidence thresholds and explicit "unknown object" language; benchmark its latency and false positives on the target device before enabling it.
4. Add calibrated depth and a walking-corridor/free-space estimator. A 2-D object box alone does not prove that an object intersects the user's path.
5. Optimise for the deployment target only after accuracy is measured: export to ONNX/OpenVINO where supported, then compare FP32/FP16/INT8 outputs and latency on the exact Intel or Raspberry Pi hardware. Keep a rollback model.

## Validation gates before field use

- Run automated tests and static checks on every change.
- Evaluate on a held-out, geographically and temporally separate test set.
- Measure per-class recall, false alarms per minute, missed critical hazards, end-to-end alert latency, and sustained FPS/temperature/power.
- Test indoors first with a sighted supervisor, then controlled outdoor courses with a cane/guide dog still available.
- Test sensor disconnects, camera failure, low light, backlighting, rain, motion blur, occlusion, and audio failure.
- Conduct supervised usability and mobility testing with visually impaired participants and qualified orientation-and-mobility professionals.

AURANAV remains a research prototype. It is not certified and must not replace a white cane, guide dog, or mobility training. Never test by asking a user to navigate a road or stairs relying on this system alone.
