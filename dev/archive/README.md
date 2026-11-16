# dev/archive

Files that are not part of a current workflow (superseded versions, one-off checks, research code). They keep their OLD flat imports and
hardcoded paths and are NOT expected to run. Decide per file whether to keep or delete; nothing here is used by main/, lib/ or the other dev folders.

| file | what it is |
|---|---|
| `compare_gt_results.txt` | Output of an old compare_against_ground_truth run. |
| `downbeat_e2e.py` | Old rule-based downbeat end-to-end check (hardcoded attachment path). |
| `downbeat_lab.py` | Old research script: rule-based downbeat lab (superseded by the learned model). |
| `downbeat_replay.py` | Old rule-based downbeat replay (superseded by the learned model). |
| `kick_sweep.py` | Old research script: kick sweep detection (not referenced by the pipeline). |
| `phrasing__experimental__adfeats.py` | Experimental after-drop features used only by cand7.py. |
| `phrasing__experimental__cand7.py` | Experimental variant 7 of the drop candidate trainer (not in production). |
| `replay_e2e.py` | Older end-to-end replay (superseded by evaluation/replay_full_pipeline.py). |
| `replay_flagged.py` | Replay of the removed phase_fix/snare_override path for flagged tracks. |
| `replay_override.py` | Replay of the removed snare_override path (multiprocessing). |
| `session13__classify.py` | Early failure classification of an eval report (hardcoded upload path). |
| `session13__foldstat.py` | Truth anchor/period per track from fold statistics (feeds modstat.py). |
| `session13__modstat.py` | Mode statistics on the foldstat output (phase ranker exploration). |
| `session13__phase_model_fold0.json` | Phase model of CV fold 0 (CV artifact). |
| `session13__phase_model_fold1.json` | Phase model of CV fold 1 (CV artifact). |
| `session13__phase_model_fold2.json` | Phase model of CV fold 2 (CV artifact). |
| `session13__phase_model_fold3.json` | Phase model of CV fold 3 (CV artifact). |
| `session13__phase_model_fold4.json` | Phase model of CV fold 4 (CV artifact). |
| `session13__phasecv.py` | First phase-ranker CV (superseded by phase_ranker_cross_validation.py). |
| `session13__replay_e2e.py` | Copy of replay_e2e.py from the same session (duplicate). |
| `session14__alldata.py` | Older build of the fine/downbeat rows (superseded by the session 15 version). |
| `session14__db_data.py` | Older downbeat row builder (superseded). |
| `session14__dbcv.py` | Older downbeat CV (superseded by downbeat_cross_validation.py). |
| `session14__fine_data2.py` | Older fine-adjust row builder (superseded). |
| `session14__gm.py` | Older genre map (differs from the kept genre_map.py). |
| `session14__metric.py` | Older track metric (differs from the kept track_metrics.py). |
| `session15__downbeat_model_old961.json` | Downbeat model trained on the first 961 tracks only (used for the holdout comparison). |
| `session15__fine_model_old961.json` | Fine-adjust model trained on the first 961 tracks only (used for the holdout comparison). |
| `snap_test.py` | Test of the removed phase snapping. |
| `snare_kick.py` | Old research script: joint snare/kick lock (not referenced by the pipeline). |
