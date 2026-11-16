"""Deterministic cue derivation (user's rules). Inputs are in BARS from the grid's first downbeat (bar 0).
Predicted core: D, H (None if absent), fk1/fk2 (fakeout-end bar or None), e1/e2 (drop-end bars).
Output: hot {'A'..'H': bar}, mem: sorted list of bars (max 10)."""
MAX_MEMORY_CUES = 10

def derive(first_drop_start, first_drop_end, second_drop_start=None, second_drop_end=None, first_fakeout_end=None, second_fakeout_end=None, max_memory_cues=MAX_MEMORY_CUES):
    hot_cues = {'D': first_drop_start}
    for cue_letter, bars_before_drop in (('A', 16), ('B', 8), ('C', 4)):
        if first_drop_start - bars_before_drop >= 0: hot_cues[cue_letter] = first_drop_start - bars_before_drop
    if second_drop_start is not None:
        hot_cues['H'] = second_drop_start
        for cue_letter, bars_before_drop in (('E', 16), ('F', 8), ('G', 4)):
            # must not overlap drop 1 (cue must be at/after drop-1 end)
            if second_drop_start - bars_before_drop >= first_drop_end: hot_cues[cue_letter] = second_drop_start - bars_before_drop
    core = {first_drop_start}                                   # mc always on A, D, E, H
    core.add(hot_cues.get('A', first_drop_start))
    if second_drop_start is not None:
        core.add(second_drop_start); core.add(hot_cues.get('E', second_drop_start))
    for bar in (first_fakeout_end, second_fakeout_end, first_drop_end, second_drop_end):                 # fakeout ends + drop ends
        if bar is not None: core.add(bar)
    memory_cues = set(core)
    # pre-A: every 8 bars before A, nearest first, until bar 0 or budget gone
    bar = hot_cues.get('A', first_drop_start) - 8
    while bar >= 0 and len(memory_cues) < max_memory_cues:
        memory_cues.add(bar); bar -= 8
    return hot_cues, sorted(memory_cues)
