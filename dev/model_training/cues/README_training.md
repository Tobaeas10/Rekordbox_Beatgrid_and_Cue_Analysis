# Retrain the cue (drop / fakeout) models on your PC
Run from the project root inside the venv. Needs numpy, scipy, scikit-learn, ffmpeg, `data/.env` (REKORDBOX_DATABASE_PATH, REKORDBOX_ENCRYPTION_KEY).
Inputs live in `debug/cue_training/` (`paths.CUE_TRAINING`), intermediates and new models are written to `debug/work/` (`paths.WORK`).
Copy a new model from `debug/work/` to `data/models/cues/` only after checking its numbers.

1. Band features of the library (only when audio/library changes; ~11 min with --parallel-workers 6):
   `python dev/tools/extract_band_features.py --beat-features-slim-path <beat_features_slim.pkl.gz> --cues-path <cues.jsonl> --output-path <band_features_slim.pkl.gz> --parallel-workers 6`
2. Dataset (ALWAYS pass the band-features file, otherwise the dataset has no band features and the drop trainer finds 0 tracks):
   `python dev/model_training/cues/build_cue_dataset.py <beat_features_slim.pkl.gz> <cues.jsonl> <band_features_slim.pkl.gz>`
3. Drop models (~8 min): `python dev/model_training/cues/train_drop_candidate_models.py --fit-final-models`
   -> out-of-fold scores + `drop_start_model.pkl` / `drop_end_model.pkl` in `debug/work/`
4. Decoder check: `python dev/model_training/cues/drop_decoder_cross_validation.py`
5. Fakeout: `fakeout_cross_validation.py` (out-of-fold), `fakeout_evaluation.py`, `train_fakeout_model.py` -> `fakeout_end_model.pkl`

Production models (trained on all 1559 tracks) are `data/models/cues/drop_start_model.pkl`, `drop_end_model.pkl`, `fakeout_end_model.pkl`; feature pipeline = `lib/cues/drop_features.py`.
