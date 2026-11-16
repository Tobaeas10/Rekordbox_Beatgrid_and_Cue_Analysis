"""Review flag per analysed track: decides whether the user should check the grid by hand (goes into the Rekordbox review playlist).
Thresholds were calibrated on pooled cross-validation (dev/evaluation/review_queue_measurement.py, gross_failure_flag_study.py); loosen/tighten here."""
TIMING_RAW_MS = 0.7      # learned fine-offset prediction magnitude (13% of tracks; misses ~6/23 timing failures)
DOWNBEAT_MARGIN = 0.9    # p[decided] - best other phase score below this (9% of tracks; misses ~4/39 downbeat failures)
FIT_RMS_MS = 5.0         # lattice fit rms (kick and snare disagree)
PLAYLIST_NAME = "Beatgrid Review"   # legacy single playlist (kept for old state)
PLAYLISTS = {"wrong": "Beatgrid: Likely Wrong", "uncertain": "Beatgrid: Uncertain"}
DOWNBEAT_WRONG_MARGIN = 0.3   # margin below this: 89% of such tracks were wrong on the big library (in-sample caveat)
ADV_DISAGREE_WRONG_MS = 4.0   # basic and advanced fine shift differ by at least this: likely wrong; >= ADV_DISAGREE_UNSURE_MS: uncertain
ADV_DISAGREE_UNSURE_MS = 2.0
PHASE_PBEST_UNSURE = 0.9      # phase ranker confidence below this: gross-failure signal (old reports: flags 5.4%, 40 of 87 gross failures, precision ~43%; none reaches 85% so it stays Uncertain)
MIN_COVERAGE = 0.5            # tempo sections cover less than this fraction of the track (2% flagged, 47% precision on gross failures)


def review_level(sections, level, assessment_notes, duration_seconds=None):
    """-> (flag_level, reasons): flag_level is "wrong" (Likely Wrong: no grid, tempo forced from Rekordbox, downbeat margin < 0.3, basic vs advanced timing
    disagree >= 4 ms), "uncertain" (any other reason below) or None (nothing to look at). reasons lists everything found, Likely Wrong first."""
    if not sections:
        return "wrong", ["no grid found"]
    likely_wrong_reasons, uncertain_reasons = [], []
    first_section = sections[0]
    if level == "check":
        uncertain_reasons += [f"assess: {note}" for note in (assessment_notes or ["check"]) if "starts at" not in note]
    elif level == "note":
        relevant_notes = [note for note in (assessment_notes or []) if ("parity" in note or "tempo sections" in note or "few snares" in note)]
        uncertain_reasons += [f"assess: {note}" for note in relevant_notes]
    if first_section.get("forced_rb"):
        likely_wrong_reasons.append("tempo forced from Rekordbox bpm")
    if first_section.get("fit_rms_ms", 0) > FIT_RMS_MS:
        uncertain_reasons.append(f"fit rms {first_section['fit_rms_ms']:.1f} ms")
    if len(sections) > 1:
        uncertain_reasons.append(f"{len(sections)} tempo sections")
    phase_best_probability = (first_section.get("phase_select") or {}).get("p_best")
    if phase_best_probability is not None and phase_best_probability < PHASE_PBEST_UNSURE and not first_section.get("forced_rb"):
        uncertain_reasons.append(f"grid phase unsure (ranker {phase_best_probability:.2f})")
    raw = first_section.get("fine_raw_ms")
    if raw is not None and abs(raw) >= TIMING_RAW_MS:
        uncertain_reasons.append(f"timing model unsure ({raw:+.1f} ms)")
    for section in sections:
        if section.get("fine_adv") and section.get("fine_basic_ms") is not None:
            disagreement_ms = abs(float(section["fine_shift_ms"]) - float(section["fine_basic_ms"]))
            if disagreement_ms >= ADV_DISAGREE_WRONG_MS:
                likely_wrong_reasons.append(f"basic and advanced timing disagree by {disagreement_ms:.1f} ms")
            elif disagreement_ms >= ADV_DISAGREE_UNSURE_MS:
                uncertain_reasons.append(f"basic and advanced timing disagree by {disagreement_ms:.1f} ms")
            break
    downbeat_info = first_section.get("downbeat")
    if downbeat_info and downbeat_info.get("scores"):
        scores = downbeat_info["scores"]; decided_phase = downbeat_info.get("q", 0)
        margin = scores[decided_phase] - max(score for index, score in enumerate(scores) if index != decided_phase)
        if margin < DOWNBEAT_WRONG_MARGIN:
            likely_wrong_reasons.append(f"downbeat very unsure (margin {margin:.2f})")
        elif margin < DOWNBEAT_MARGIN:
            uncertain_reasons.append(f"downbeat unsure (margin {margin:.2f})")
        if downbeat_info.get("adv") and (downbeat_info.get("basic") or {}).get("q", decided_phase) != decided_phase:
            uncertain_reasons.append("downbeat changed by the advanced model")
    elif not downbeat_info:
        uncertain_reasons.append("downbeat not evaluated")
    return ("wrong" if likely_wrong_reasons else "uncertain" if uncertain_reasons else None), likely_wrong_reasons + uncertain_reasons


def review(sections, level, assessment_notes, duration_seconds=None):
    """-> (flag, reasons); flag = any review level. Kept for callers that only need yes/no."""
    flag_level, reasons = review_level(sections, level, assessment_notes, duration_seconds)
    return flag_level is not None, reasons
