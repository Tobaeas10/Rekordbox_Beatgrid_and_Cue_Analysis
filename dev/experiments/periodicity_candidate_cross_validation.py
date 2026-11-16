"""Grouped CV of the candidate ranker: base features vs base + periodicity. Rule: move off 'ours' only if score margin >= m. python periodicity_candidate_cross_validation.py CANDIDATES.pkl (written by periodicity_candidate_dataset.py; the _info.pkl next to it is read too)"""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import sys, numpy as np, pandas as pd
from lib.beatgrid import rankers
from dev.experiments.periodicity_candidate_dataset import PERIODICITY_FEATURE_NAMES
from sklearn.ensemble import HistGradientBoostingClassifier
candidate_table = pd.read_pickle(sys.argv[1]); track_info = pd.read_pickle(sys.argv[1].replace('.pkl', '_info.pkl'))
base_columns = list(rankers.PHASE_FEATS)
track_ids = candidate_table.id.unique(); random_generator = np.random.RandomState(0)
def grouped_cross_validation(feature_columns, seed_count=2, margins=(0.0, 0.1, 0.2, 0.3, 0.5)):
    scores = np.zeros(len(candidate_table))
    for seed in range(seed_count):
        fold_by_id = dict(zip(track_ids, np.random.RandomState(seed).randint(0, 5, len(track_ids)))); fold_of_row = candidate_table.id.map(fold_by_id).values
        for fold_index in range(5):
            train_mask, test_mask = fold_of_row != fold_index, fold_of_row == fold_index
            model = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=.05, min_samples_leaf=10, random_state=seed).fit(candidate_table.loc[train_mask, feature_columns], candidate_table.label[train_mask])
            scores[test_mask] += model.predict_proba(candidate_table.loc[test_mask, feature_columns])[:, 1] / seed_count
    scored_table = candidate_table.assign(sc=scores); results_by_margin = {}
    for margin in margins:
        fixed = broken = wrong = 0; badnames = []
        for track_id, candidates in scored_table.groupby('id'):
            ours = candidates.iloc[0]; best = candidates.loc[candidates.sc.idxmax()]
            chosen = best if (best.sc - ours.sc) >= margin else ours
            if chosen.label == 1 and ours.label == 0: fixed += 1
            if chosen.label == 0 and ours.label == 1: broken += 1
            wrong += int(chosen.label == 0)
        results_by_margin[margin] = (fixed, broken, wrong)
    return results_by_margin, scored_table
ours_wrong_count = int((candidate_table.groupby('id').label.first() == 0).sum())
print('ours wrong at start', ours_wrong_count, 'tracks', len(track_ids))
for model_name, feature_columns in (('base', base_columns), ('base+periodicity', base_columns + PERIODICITY_FEATURE_NAMES)):
    results, scored_table = grouped_cross_validation(feature_columns); print(model_name, results)
    true_errors_ms = track_info.set_index('id').tru.abs()
    gross_track_ids = [track_id for track_id in track_ids if true_errors_ms[track_id] > 20]
    for margin in (0.0, 0.2, 0.3):
        fixed_names = []
        for track_id in gross_track_ids:
            candidates = scored_table[scored_table.id == track_id]; ours = candidates.iloc[0]; best = candidates.loc[candidates.sc.idxmax()]; chosen = best if (best.sc - ours.sc) >= margin else ours
            if chosen.label == 1: fixed_names.append(track_info.set_index('id').name[track_id][:30])
        print('   gross(>20ms)', len(gross_track_ids), 'fixed at margin', margin, len(fixed_names), fixed_names if margin == 0.2 else '')
