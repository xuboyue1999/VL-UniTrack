# VL-UniTrack: A Unified Framework with Visual-Language Prompts for UAV-Ground Visual Tracking

Accepted by IEEE Transactions on Multimedia.

This repository provides the core code of VL-UniTrack. The complete code will be released soon.

The core implementation comprises five Python files in `vl_unitrack_core/`:

- `encoder.py`: Unified visual encoder.
- `vlvp.py`: Visual-Language View Prompting (VLVP).
- `pcva.py`: Prompt-guided Cross-View Adaptation (PCVA).
- `cmd.py`: Confidence-Modulated Distillation (CMD).
- `__init__.py`: Module exports.

Raw tracking result files are provided to facilitate verification.

## Tracking results and evaluation

- [Final UGVT tracking results](VL-UniTrack_UGVT_results.zip): 90 prediction files covering the UAV and ground views of 45 test sequences. No model weights are included.
- [Tool1](Tool1.zip): evaluates 45 unique test sequences per view, counting each sequence once.

The evaluation package is standalone and contains the evaluation script, 90 test-set GT files, sequence lists, dependency information, and English/Chinese instructions. It does not include training data, images, predictions, or model weights. GT remains subject to the original dataset providers' terms.

For the released tracking results, Tool1 gives 0.886 and 0.717 in UAV PR and SR, 0.853 and 0.704 in ground-view PR and SR, and 0.869 and 0.711 in average PR and SR, respectively.

**Our paper uses Tool1.** It follows the standard equal-weight, sequence-averaged evaluation convention: each distinct test sequence contributes once. Use the same evaluation protocol for all trackers being compared.

### Quick start

Requires Python 3.9+ and NumPy; no GPU or MATLAB is needed. Extract the result archive and Tool1.zip into the repository root, then run:

```bash
python -m pip install -r Tool1/requirements.txt
python Tool1/evaluate.py --results VL-UniTrack_UGVT_results
```

Append `--verify-reference` to check the separately released VL-UniTrack predictions against the reference metrics. Omit that flag when evaluating other predictions. PR is precision at a 20-pixel center-error threshold; SR is the mean success rate over IoU thresholds from 0 to 1 in steps of 0.05, not SR at IoU 0.5. The tool's `README.txt` documents the full evaluation conventions.
