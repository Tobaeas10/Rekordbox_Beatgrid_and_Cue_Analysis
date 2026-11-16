"""cand6.py (wip37): candidates on EVERY bar, same normalised features as cand2.py, + cand%4/%8/%16 flags.
Usage (from this folder, after build.py made ds.pkl):
    python cand6.py                 # features -> c6_feat.npy (cached), 5-fold grouped CV -> c6_os.npy c6_oe.npy c6_meta.pkl
    python cand6.py --fit-final-models         # additionally fits final start/end models on ALL tracks -> c6_model_start.pkl c6_model_end.pkl
Options: --negative-row-fraction 0.3 (fraction of negative rows used for training), --maximum-iterations 150, --maximum-tracks N (quick test)
Memory: ~220k rows x ~1250 float32 = ~1.1 GB (feature cache is a memmap). Uses all CPU cores via HGB/OpenMP.
"""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pickle, sys, os, time, argparse, numpy as np
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--negative-row-fraction', type=float, default=0.3)
argument_parser.add_argument('--maximum-iterations', type=int, default=150)
argument_parser.add_argument('--maximum-tracks', type=int, default=0)
argument_parser.add_argument('--fit-final-models', action='store_true')
argument_parser.add_argument('--fold-count', type=int, default=5)
arguments = argument_parser.parse_args()
cue_dataset = pickle.load(open(paths.WORK / 'cue_dataset.pkl', 'rb'))
from dev.model_training.cues.cue_labels import labels
from lib.cues.drop_features import drop_ends, bar_level_features, beat_resolved_features, eighth_note_band_features, normalized_bar_features, phase_alignment_features, GENRE_PLAYLIST_NAMES, CANDIDATE_BAR_OFFSETS, BAR_FEATURE_COLUMNS
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold




# ---- tracks + labels
training_tracks = []
for track in cue_dataset:
    first_drop_start, first_drop_end, second_drop_start, second_drop_end = drop_ends(track)
    if first_drop_end is None or (second_drop_start is not None and second_drop_end is None): continue
    track_labels = labels(track, first_drop_start, second_drop_start)
    if track_labels['s1'] != 'ok' or (second_drop_start is not None and track_labels['s2'] not in ('ok', 'unk2')): continue   # ambiguous / no-end tracks dropped
    if track.get('BX') is None: continue          # needs band features
    training_tracks.append((track, first_drop_start, track_labels['e1'], second_drop_start, track_labels['e2']))
if arguments.maximum_tracks: training_tracks = training_tracks[:arguments.maximum_tracks]
print(len(training_tracks), 'tracks', flush=True)

# ---- features (every bar candidate)
rows = []; row_count = 0
for track, first_drop_start, first_drop_end, second_drop_start, second_drop_end in training_tracks:
    bar_count = len(track['BE']) // 4; row_count += bar_count + 1
cached_features = None
if os.path.exists(str(paths.WORK / 'drop_candidate_features.npy')) and os.path.exists(str(paths.WORK / 'drop_candidate_meta.pkl')) and not arguments.maximum_tracks:
    candidate_meta = pickle.load(open(str(paths.WORK / 'drop_candidate_meta.pkl'), 'rb')); cached_features = np.load(str(paths.WORK / 'drop_candidate_features.npy'), mmap_mode='r'); print('feature cache', cached_features.shape)
if cached_features is None:
    candidate_meta = []; chunks = []
    column_count = None; feature_table = None; write_position = 0; start_time = time.time()
    for track_index, (track, first_drop_start, first_drop_end, second_drop_start, second_drop_end) in enumerate(training_tracks):
        bar_features, bar_count = bar_level_features(track); normalized_features = normalized_bar_features(track); bar_features = np.concatenate([bar_features[:, BAR_FEATURE_COLUMNS], normalized_features], 1)
        candidate_bars = np.arange(0, bar_count + 1)
        feature_matrix = np.concatenate([bar_features[np.clip(candidate_bars + offset, 0, bar_count - 1)] for offset in CANDIDATE_BAR_OFFSETS], 1)
        genre_flags = np.array([genre in track['pl'] for genre in GENRE_PLAYLIST_NAMES], np.float32)
        feature_matrix = np.concatenate([feature_matrix, beat_resolved_features(track), eighth_note_band_features(track), np.tile(genre_flags, (len(candidate_bars), 1)), phase_alignment_features(track), np.stack([candidate_bars % 4 == 0, candidate_bars % 8 == 0, candidate_bars % 16 == 0, bar_count - candidate_bars, candidate_bars, candidate_bars / bar_count, np.full(len(candidate_bars), track['bpm'])], 1)], 1).astype(np.float32)
        if feature_table is None:
            column_count = feature_matrix.shape[1]; feature_table = np.lib.format.open_memmap(str(paths.WORK / 'drop_candidate_features.npy'), mode='w+', dtype=np.float32, shape=(row_count, column_count))
        feature_table[write_position:write_position+len(feature_matrix)] = feature_matrix; write_position += len(feature_matrix)
        candidate_meta.append((candidate_bars, bar_count, (first_drop_start, first_drop_end, second_drop_start, second_drop_end), track['id'], track['name'], write_position - len(feature_matrix)))
        if track_index % 200 == 0: print('feat', track_index, '%.0fs' % (time.time() - start_time), flush=True)
    feature_table.flush(); cached_features = feature_table[:write_position] if write_position != row_count else feature_table
    if not arguments.maximum_tracks: pickle.dump(candidate_meta, open(str(paths.WORK / 'drop_candidate_meta.pkl'), 'wb'))
feature_table = cached_features; N = len(feature_table); print(feature_table.shape, flush=True)

# ---- targets (start: D,H ; end: e1,e2), every bar
start_targets = np.zeros(N, np.int8); end_targets = np.zeros(N, np.int8); group_ids = np.zeros(N, np.int32)
for i, meta_row in enumerate(candidate_meta):
    candidate_bars, bar_count, (first_drop_start, first_drop_end, second_drop_start, second_drop_end), _, _, row_offset = meta_row
    group_ids[row_offset:row_offset+len(candidate_bars)] = i
    for event_bar, target in ((first_drop_start, start_targets), (second_drop_start, start_targets), (first_drop_end, end_targets), (second_drop_end, end_targets)):
        if event_bar is None: continue
        bar_index = int(round(event_bar))
        if abs(event_bar - bar_index) < 0.13 and 0 <= bar_index < len(candidate_bars): target[row_offset + bar_index] = 1
random_generator = np.random.RandomState(0)
training_subset = (start_targets == 1) | (end_targets == 1) | (random_generator.rand(N) < arguments.negative_row_fraction)       # training subset
def make_model(): return HistGradientBoostingClassifier(max_iter=arguments.maximum_iterations, learning_rate=0.08, max_leaf_nodes=16, l2_regularization=2.0, min_samples_leaf=30)
drop_start_scores = np.zeros(N, np.float32); drop_end_scores = np.zeros(N, np.float32)
start_time = time.time()
for fold_index, (train_indices, test_indices) in enumerate(GroupKFold(arguments.fold_count).split(np.arange(N), groups=group_ids)):
    training_rows = train_indices[training_subset[train_indices]]; training_matrix = np.asarray(feature_table[training_rows])
    for target, scores in ((start_targets, drop_start_scores), (end_targets, drop_end_scores)):
        model = make_model().fit(training_matrix, target[training_rows])
        for chunk_start in range(0, len(test_indices), 20000): scores[test_indices[chunk_start:chunk_start+20000]] = model.predict_proba(np.asarray(feature_table[test_indices[chunk_start:chunk_start+20000]]))[:, 1]
    del training_matrix
    print('fold', fold_index, '%.0fs' % (time.time() - start_time), flush=True)
np.save(str(paths.WORK / 'drop_candidate_start_oof.npy'), drop_start_scores); np.save(str(paths.WORK / 'drop_candidate_end_oof.npy'), drop_end_scores); print('OOF saved', flush=True)
if arguments.fit_final_models:
    training_rows = np.where(training_subset)[0]; training_matrix = np.asarray(feature_table[training_rows])
    pickle.dump(make_model().fit(training_matrix, start_targets[training_rows]), open(str(paths.WORK / 'drop_start_model.pkl'), 'wb'))
    pickle.dump(make_model().fit(training_matrix, end_targets[training_rows]), open(str(paths.WORK / 'drop_end_model.pkl'), 'wb')); print('final models saved')
print('DONE')
