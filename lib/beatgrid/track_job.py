"""Per-track analysis job for evaluate_beatgrid_on_library.py. Imports only numpy/scipy and grid_pipeline, so worker processes start fast."""
import json
import pickle
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from lib.beatgrid import advanced_fine_adjust as advanced
from lib.beatgrid import grid_pipeline
from lib.beatgrid import tempo_detection


def wrap(value, period):
    return (value + period / 2) % period - period / 2


def header_facts(path):
    try:
        probe_data = tempo_detection.probe_json(path)
        audio_stream = next(stream for stream in probe_data['streams'] if stream.get('codec_type') == 'audio')
    except Exception as e:
        return {'error': repr(e)}
    tags = {tag_name.lower(): tag_value for tag_name, tag_value in (probe_data.get('format', {}).get('tags') or {}).items()}
    tags.update({tag_name.lower(): tag_value for tag_name, tag_value in (audio_stream.get('tags') or {}).items()})
    return {'codec': audio_stream.get('codec_name'), 'sr': audio_stream.get('sample_rate'), 'start_time': audio_stream.get('start_time'),
            'encoder': tags.get('encoder'), 'itunsmpb': tags.get('itunsmpb'), 'bit_rate': audio_stream.get('bit_rate')}


def measure_timeline_offset(cache_dir, track_id, audio_path, extended_analysis_path):
    cache_file = Path(cache_dir) / f"{track_id}.wave.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    waveform_offset = None
    if extended_analysis_path and (os.environ.get('BEATGRID_ALWAYS_MEASURE_WAVEFORM') or grid_pipeline.probe_header(str(audio_path)).get('codec') != 'aac'):
        # container_offset_ms ignores the waveform for AAC (priming is used), so the measurement is skipped there
        waveform_offset = grid_pipeline.waveform_offset_ms(str(extended_analysis_path), str(audio_path))
    cache_file.write_text(json.dumps(waveform_offset))
    return waveform_offset


def detect(cache_dir, audio, track_id, rekordbox_bpm=None):
    cache_file = Path(cache_dir) / f"{track_id}.ff2{grid_pipeline.CACHE_TAG}.pkl"
    if cache_file.exists():
        events = pickle.loads(cache_file.read_bytes())
    else:
        events = grid_pipeline.get_events(str(audio), cache=False)
        cache_file.write_bytes(pickle.dumps(events))
    events_changed = grid_pipeline.add_bass(events, audio)
    events_changed = grid_pipeline.add_novelty(events, audio) or events_changed
    if events_changed:
        cache_file.write_bytes(pickle.dumps(events))
    blocks = grid_pipeline.block_scans(events)
    raw_sections = grid_pipeline.find_sections(blocks)
    fits = grid_pipeline.clean_sections(events, raw_sections, [grid_pipeline.fit_section(events, raw_section) for raw_section in raw_sections])
    fits = grid_pipeline.fix_rekordbox_ratio(events, raw_sections, fits, rekordbox_bpm)
    fits = grid_pipeline.fix_whole_ratio(events, raw_sections, fits)
    fits = grid_pipeline.fix_rekordbox_near(events, fits, rekordbox_bpm)
    fits = grid_pipeline.drop_ghost_tail(events, fits)
    if not [fit for fit in fits if fit]:
        forced_section = grid_pipeline.forced_rekordbox_section(events, rekordbox_bpm)
        fits = [forced_section] if forced_section else fits
    sections = [grid_pipeline.fine_adjust(events, grid_pipeline.refine_anchor(events, grid_pipeline.select_phase(events, grid_pipeline.fix_parity(events, fit, first=(section_index == 0))))) for section_index, fit in enumerate(fits)]
    advanced_mode = advanced.mode()
    if advanced_mode == 'advanced' or (advanced_mode == 'auto' and advanced.needs(sections)):
        sections = advanced.apply(events, sections, audio)
    return events, sections


def build_grid_advanced(events, sections, model_ms, audio):
    """P.build_grid plus the advanced downbeat second pass: the basic build decides first; when advanced.wants_downbeat (basic margin < 0.9, or
    --force-advanced) the per-beat band energies are computed from the audio and the grid is built again with the bar-folded downbeat model."""
    built = grid_pipeline.build_grid(sections, model_ms, events['dur'], events)
    if built and advanced.wants_downbeat(sections) and advanced.attach_band_energies(events, sections, audio):
        built = grid_pipeline.build_grid(sections, model_ms, events['dur'], events)
    return built


def evaluate(sections, rekordbox_beats, model_ms):
    """Compare each detected section with the Rekordbox beat times (ms) that fall inside it.
    model_ms is the container offset model (Rekordbox time minus decoded time); err_* is measured minus model."""
    rekordbox_ms = np.array([beat[2] for beat in rekordbox_beats], float)
    rows = []
    for section in sections:
        period_ms, anchor_ms = 1000 * section["period_s"], 1000 * section["anchor_s"]
        in_section = (rekordbox_ms >= 1000 * section["start_s"]) & (rekordbox_ms <= 1000 * section["end_s"])
        row = dict(start_s=section["start_s"], end_s=section["end_s"], bpm=section["bpm"], snapped=section["snapped"], anchor_ms=anchor_ms,
                   fit_rms_ms=section["rms_ms"], n_rb_beats=int(in_section.sum()), parity=section.get("parity"), refine=section.get("refine"),
                   phase_peaks=section.get("phase_peaks"), phase_shift_ms=section.get("phase_shift_ms"), snare_override_ms=section.get("snare_override_ms"), phase_select=section.get("phase_select"), fine_shift_ms=section.get("fine_shift_ms"), rb_near_fix=section.get("rb_near_fix"), rb_ratio_fix=section.get("rb_ratio_fix"), forced_rb=section.get("forced_rb"), fine_raw_ms=section.get("fine_raw_ms"), fine_adv=section.get("fine_adv"), fine_basic_ms=section.get("fine_basic_ms"), whole_ratio_fix=section.get("whole_ratio_fix"))
        if in_section.sum() >= 8:
            offsets_ms = wrap(rekordbox_ms[in_section] - anchor_ms, period_ms)
            median_offset_ms = float(np.median(offsets_ms))
            residuals_ms = offsets_ms - median_offset_ms
            row.update(offset_mod_beat_ms=median_offset_ms, offset_mod_half_beat_ms=float(np.median(wrap(rekordbox_ms[in_section] - anchor_ms, period_ms / 2))),
                       err_vs_model_ms=float(wrap(median_offset_ms - model_ms, period_ms)),
                       spread_p05=float(np.percentile(residuals_ms, 5)), spread_p95=float(np.percentile(residuals_ms, 95)),
                       spread_worst=float(np.abs(residuals_ms).max()), spread_rms=float(np.sqrt(np.mean(residuals_ms ** 2))))
        rows.append(row)
    return rows


def run_track(job):
    """job: dict(id, src, ext, beats, cache). Returns a dict of results (or error) for one track."""
    start_time = time.time()
    result = {}
    try:
        audio_path = job['src']
        result['header'] = header_facts(audio_path)
        header = grid_pipeline.probe_header(audio_path)
        try:
            waveform_offset = measure_timeline_offset(job['cache'], job['id'], audio_path, job.get('ext'))
        except Exception as e:
            waveform_offset = None
            result['waveform_error'] = repr(e)
        model_ms, status = grid_pipeline.container_offset_ms(header, waveform_offset)
        result['offset_model'] = dict(ms=model_ms, status=status, waveform=waveform_offset)
        events, sections = detect(job['cache'], audio_path, job['id'], job.get('rb_bpm'))
        result['sections'] = evaluate(sections, job['beats'], model_ms)
        built = build_grid_advanced(events, sections, model_ms, audio_path)
        result['dry_run'] = grid_pipeline.dry_compare(built, job['beats'])
        result['downbeat'] = sections[0].get('downbeat') if sections else None
        if job.get('dat') and built:
            from lib.rekordbox import anlz_files as anlz_grid
            try:
                result['dry_run']['anlz_rebuild_ok'] = bool(anlz_grid.roundtrip_ok(Path(job['dat']).read_bytes(), built))
            except Exception as e:
                result['dry_run']['anlz_rebuild_ok'] = False
                result['dry_run']['anlz_error'] = repr(e)
        result['assess'] = dict(zip(('level', 'reasons'), grid_pipeline.assess(events, sections, events['dur'])))
        result['n_kicks'], result['n_snares'], result['duration_s'] = events['nk'], events['ns'], round(events['dur'], 1)
        if job.get('beat_features') and sections:
            from lib import audio_features
            beat_features_folder = Path(job['cache']).parent / 'beat_features'
            beat_features_folder.mkdir(exist_ok=True)
            cache_file = beat_features_folder / f"{job['id']}.npz"
            if not cache_file.exists():
                np.savez_compressed(cache_file, **audio_features.compute_beat_features(events, sections, audio_path, job['beats'], model_ms))
        if job.get('band_envelopes'):
            cache_file = Path(job['cache']) / f"{job['id']}.band_envelopes.npz"
            if not cache_file.exists():
                from lib import audio_features
                np.savez_compressed(cache_file, **audio_features.compute_band_envelopes(audio_path))
    except Exception as e:
        result['error'] = repr(e)
        result.setdefault('sections', [])
    result['seconds'] = round(time.time() - start_time, 1)
    return result


def sweep_track(job):
    """Onset-threshold sweep: for each fraction re-detect events and report the residual mode of snares and kicks against the
    Rekordbox eighth-note lattice (ms; positive = detector later than the stored grid)."""
    result = dict(fracs={})
    try:
        audio_path = job['src']
        waveform_offset = measure_timeline_offset(job['cache'], job['id'], audio_path, job.get('ext'))
        model_ms, _ = grid_pipeline.container_offset_ms(grid_pipeline.probe_header(audio_path), waveform_offset)
        rekordbox_ms = np.array([beat[2] for beat in job['beats']], float)
        half = 0.5 * 6000000.0 / job['beats'][0][1] / 1000.0
        first_beat_time = (rekordbox_ms[0] - model_ms) / 1000.0
        for fraction in job['fracs']:
            events = grid_pipeline.get_events(audio_path, cache=False, onset_fraction=fraction)
            residuals_ms = (events['t'] - (first_beat_time + np.round((events['t'] - first_beat_time) / half) * half)) * 1000.0
            result['fracs'][str(fraction)] = {instrument_name: grid_pipeline.residual_mode(residuals_ms[events['c'] == class_id], events['w'][events['c'] == class_id]) for instrument_name, class_id in (('snare', 1), ('kick', 0))}
    except Exception as e:
        result['error'] = repr(e)
    return result
