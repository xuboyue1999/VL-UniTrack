# VL-UniTrack: A Unified Framework with Visual-Language Prompts for UAV-Ground Visual Tracking (Under review)

This repository provides the core code of VL-UniTrack. The complete code will be released soon.

The core implementation comprises five Python files in `vl_unitrack_core/`:

- `encoder.py`: Unified visual encoder.
- `vlvp.py`: Visual-Language View Prompting (VLVP).
- `pcva.py`: Prompt-guided Cross-View Adaptation (PCVA).
- `cmd.py`: Confidence-Modulated Distillation (CMD).
- `__init__.py`: Module exports.

Raw tracking result files are provided to facilitate verification.

## Tracking results and evaluation tools

- [Final UGVT tracking results](VL-UniTrack_UGVT_results.zip): 90 prediction files covering the UAV and ground views of 45 test sequences. No model weights are included.
- [Tool1](Tool1.zip): evaluates 45 unique test sequences per view, counting each sequence once.
- [Tool2](Tool2.zip): evaluates the same 45 unique sequences using a 46-entry list, in which `van-3` is counted twice per view.

Both evaluation packages are standalone and contain the evaluation script, the same 90 test-set GT files, sequence lists, dependency information, and English/Chinese instructions. They do not include training data, images, predictions, or model weights. GT remains subject to the original dataset providers' terms.

For the released tracking results, Tool1 gives 0.886 and 0.717 in UAV PR and SR, 0.853 and 0.704 in ground-view PR and SR, and 0.869 and 0.711 in average PR and SR, respectively.

Tool2 gives 0.889 and 0.721 in UAV PR and SR, 0.856 and 0.709 in ground-view PR and SR, and 0.872 and 0.715 in average PR and SR, respectively.

**The only reason for the difference between the two tools' scores is that Tool2 counts the same `van-3` sequence one extra time in each view.** Both tools use exactly the same tracking predictions, GT, metric definitions, and evaluation code. The duplicated sequence-list entry is the sole difference affecting the reported metrics.

**Our paper uses Tool1.** It follows the standard equal-weight, sequence-averaged evaluation convention: each distinct test sequence contributes once, without additional weight from duplicate list entries. We therefore recommend Tool1 for comparisons. Use the same tool for all trackers being compared.

### Quick start

Requires Python 3.9+ and NumPy; no GPU or MATLAB is needed. Extract the result archive and the desired tool archive into the repository root, then run:

```bash
python -m pip install -r Tool1/requirements.txt
python Tool1/evaluate.py --results VL-UniTrack_UGVT_results
```

To evaluate with Tool2, extract its archive and run:

```bash
python Tool2/evaluate.py --results VL-UniTrack_UGVT_results
```

Both tools use NumPy. Append `--verify-reference` to check the separately released VL-UniTrack predictions against the reference metrics. Omit that flag when evaluating other predictions. PR is precision at a 20-pixel center-error threshold; SR is the mean success rate over IoU thresholds from 0 to 1 in steps of 0.05, not SR at IoU 0.5. Each tool's `README.txt` documents the full evaluation conventions.
