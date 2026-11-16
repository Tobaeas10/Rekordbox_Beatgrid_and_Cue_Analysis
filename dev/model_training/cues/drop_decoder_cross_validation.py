"""cdec6.py: decode c6_os/c6_oe (every-bar candidates). D,e1,H,e2 all share one phase (bar mod 8), best phase wins.
No d4 prior. Length prior favours 16-multiples. Reports D, D+e1 (HEADLINE), H, e2, ALL4."""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pickle, numpy as np, collections, sys
from lib.cues.cue_predictor import decode, DECODER_PENALTIES
meta = pickle.load(open(str(paths.WORK / 'drop_candidate_meta.pkl'), 'rb')); drop_start_scores = np.load(str(paths.WORK / 'drop_candidate_start_oof.npy')); drop_end_scores = np.load(str(paths.WORK / 'drop_candidate_end_oof.npy'))
def is_close(a, b): return a is not None and b is not None and abs(a - b) < 0.6
def run(penalties, verbose=True, idx=None):
    hit_counts = collections.Counter(); decoded_results = []
    for i, track_meta in enumerate(meta):
        candidates, bar_count, (true_first_drop_start, true_first_drop_end, true_second_drop_start, true_second_drop_end), _, _, offset = track_meta
        decoded = decode(drop_start_scores[offset:offset+len(candidates)], drop_end_scores[offset:offset+len(candidates)], bar_count, penalties); decoded_results.append(decoded)
        if decoded is None: hit_counts['none'] += 1; continue
        hit_counts['D'] += is_close(decoded[0], true_first_drop_start); hit_counts['D+e1'] += is_close(decoded[0], true_first_drop_start) and is_close(decoded[1], true_first_drop_end)
        hit_counts['hasH'] += ((decoded[2] is None) == (true_second_drop_start is None)); hit_counts['nH'] += true_second_drop_start is not None
        if true_second_drop_start is not None: hit_counts['H'] += is_close(decoded[2], true_second_drop_start); hit_counts['e2'] += is_close(decoded[3], true_second_drop_end)
        hit_counts['all4'] += is_close(decoded[0], true_first_drop_start) and is_close(decoded[1], true_first_drop_end) and ((true_second_drop_start is None and decoded[2] is None) or (true_second_drop_start is not None and is_close(decoded[2], true_second_drop_start) and is_close(decoded[3], true_second_drop_end)))
    track_count = len(meta)
    if verbose: print(penalties, '\nD %.3f  D+e1 %.3f | H %.3f e2 %.3f | hasH %.3f | ALL4 %.3f (n=%d)' % (hit_counts['D']/track_count, hit_counts['D+e1']/track_count, hit_counts['H']/max(hit_counts['nH'], 1), hit_counts['e2']/max(hit_counts['nH'], 1), hit_counts['hasH']/track_count, hit_counts['all4']/track_count, track_count))
    return decoded_results, hit_counts
if __name__ == '__main__':
    pr = DECODER_PENALTIES
    R, c = run(pr); pickle.dump(R, open(str(paths.WORK / 'drop_decoder_results.pkl'), 'wb'))
