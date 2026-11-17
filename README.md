# Rekordbox Beatgrid & Cue Analysis

Automatically fixes the **beat grids** of new tracks in your Rekordbox library and places **hot cues and memory cues** (intro / build-up / drop / fakeout structure of bass music) on top of the corrected grid.

**TL;DR:** import tracks into Rekordbox and let it analyse them as usual. Then run one command. For every track the script re-detects the beat grid (usually fixing Rekordbox's own grid, which is wrong on roughly half of dubstep/riddim/tearout tracks), writes it back into Rekordbox's analysis file and database, predicts cues A-H plus memory cues, and shows the progress in a live table.

```
python main/predict_beatgrids_and_cues.py
```

> **Read this first**
> - **Tuned for one library.** The detectors and cue models were trained on the author's own library (bass music: dubstep, riddim, tearout, DnB, UKG, 140) and on the author's own cue system. They work very well there. For other music or a different cue system you should expect worse results until you adapt and retrain (see [Will it work on my library?](#5-will-it-work-on-my-library)).
> - **It writes to your Rekordbox library** (`master.db` and the track analysis files). Every run makes backups and never touches cues you set yourself, but **make a Rekordbox library backup before your first run** and try `--dry-run` first.
> - Tested with **Rekordbox 7** and **Python 3.14** on Windows 10/11 (see [Known limitations](#9-known-limitations)).

---

## Contents

1. [Setup](#1-setup)
2. [Your first run](#2-your-first-run)
3. [Which script do I run?](#3-which-script-do-i-run) (scripts, song selection, options)
4. [What to expect](#4-what-to-expect) (the table, what gets written, review playlists, undoing things)
5. [Will it work on my library?](#5-will-it-work-on-my-library)
6. [How it works](#6-how-it-works) (beat grid, cues, parallelism)
7. [Project layout and data folders](#7-project-layout-and-data-folders)
8. [Retraining and development](#8-retraining-and-development)
9. [Known limitations and troubleshooting](#9-known-limitations)
10. [Project history](#10-project-history)

---

## 1. Setup

**Prerequisite:** Rekordbox installed, preferably Rekordbox 7 (other versions have not been tested)

1. On GitHub click **Code > Download ZIP** and unzip it at your desired installation location.
2. Double-click **`SETUP (Windows).bat`** or **`SETUP (MacOS).command`**.
   - Windows: if it says "Windows protected your PC", click **More info > Run anyway**.
   - macOS: if it refuses to open, right-click the file > **Open** > **Open**. It may ask once for your Mac password (Python installer).
3. When it is done, copy the three new shortcuts (`Beatgrids and Cues`, `Beatgrids only`, `Cues only`) to your Desktop and double-click one to run. If you move the program folder, run the setup again.

Advanced: `python -m lib.setup_install [--force] [--no-download] [--no-pause]` in a terminal.


## 2. Your first run

1. **Back up your library** in Rekordbox (File > Library > Backup Library). Scripts also back up `master.db` themselves, but one manual backup is cheap insurance.
2. **Import 2-3 new tracks** into Rekordbox and let Rekordbox finish its own analysis. The scripts wait for Rekordbox's analysis files of each track (they need them), so you can also start the script while Rekordbox is still analysing; it simply works on each track as soon as Rekordbox is done with it.
3. **Double-click the `Beatgrids and Cues` shortcut.** It takes the tracks **added in the last 8 hours** and shows its progress in a live table. When it is done, press Enter to close the window.
4. **Restart Rekordbox** and check the tracks (new BPM, grids and cues show after a restart / reloading the track). It is fine to run the scripts while Rekordbox is open, but Rekordbox does not see database changes until it is restarted.
5. Happy? Import more tracks and run it again. Stop any time with `Ctrl+C`; finished tracks are kept.

**Nothing found in the last 8 hours?** The window then asks what you want instead: tracks added in the last N hours, tracks added between two dates (it tells you how to type them), all tracks of a playlist, or the full list of options (`--help`) where you can type your own. So you can pick other tracks without a terminal.

Want to try without writing anything, or give options right away? Use a terminal in the program folder and add options, see [song selection](#song-selection):

```bash
python main/predict_beatgrids_and_cues.py --dry-run
python main/predict_beatgrids_and_cues.py --names "Track name 1,Track name 2"
```

---

## 3. Which script do I run?

The three scripts in `main/` are the ones to run. Double-click a shortcut made by the setup (or the `.py` file itself on Windows) to run one with its defaults, or start it from a terminal to give it options (below). Each script finds the project root itself and uses the `.venv` automatically.

| Script | What it does | When to use it |
|---|---|---|
| `predict_beatgrids_and_cues.py` | Beat grid **and** cues for the same songs, one run, one live table. A song is done when its grid is written, its cues are written and `master.db` is committed. | **The normal choice.** |
| `predict_beatgrids.py` | Beat grid only (and BPM, cue snapping, review playlists). | You do not want cues, or you want to review grids before cues. |
| `predict_cues.py` | Cues only, for tracks whose grid `predict_beatgrids.py` analysed earlier (it uses that grid as the scaffold). | After a grid-only run; redoing cues; `--recolor` repair. |

### Song selection

The three analysis scripts select songs the same way. Default: **tracks added in the last 8 hours**.

| Option | Meaning |
|---|---|
| `--since-hours N` | tracks added in the last N hours |
| `--from DATE --to DATE` | tracks added in a date range (e.g. `--from 2026-11-01 --to 2026-11-09`) |
| `--playlist NAME` | every track of a playlist (can be given several times) |
| `--names "A,B"` | tracks whose file name or title contains one of the words |
| `--force` | redo tracks that are already done (see below) |
| `--dry-run` | analyse and report, write nothing |

Tracks already done are remembered (`data/state/`) and skipped unless you use `--force`. With `--force` the grid is redone and the cues **this tool wrote earlier** are replaced; cues you made yourself are never replaced.

### Examples

```bash
python main/predict_beatgrids_and_cues.py                              # last 8 hours
python main/predict_beatgrids_and_cues.py --since-hours 48
python main/predict_beatgrids_and_cues.py --playlist "Dubstep"
python main/predict_beatgrids_and_cues.py --force --names "Strawman"   # redo one track
python main/predict_beatgrids_and_cues.py --no-cues                    # grid only, same table
python main/predict_beatgrids.py --since-hours 24 --basic-only         # fastest, grid only
python main/predict_cues.py --since-hours 24                           # cues for grids analysed earlier
python main/predict_cues.py --recolor                                  # repair cue colours, then exit
```

### More options

**Grid options** (`predict_beatgrids.py` and `predict_beatgrids_and_cues.py`):

| Option | Meaning |
|---|---|
| `--basic-only` | never run the slower "advanced" timing/downbeat models (fastest) |
| `--force-advanced` | run the advanced models on every track (slowest, most thorough). Default: only on tracks the basic pass is unsure about |
| `--no-cue-snapping` | do not move your existing cues onto the new grid |
| `--only-snap-cues` | only snap the existing cues of the selected tracks to their current grid, no analysis |
| `--cue-snap-margin-ms N` | cues within N ms of a beat are moved onto it (default 10, or `CUE_SNAP_MARGIN_MS` in `.env`) |
| `--defer-database-write` | do not write `master.db` now; BPM updates are kept for the next run (the combined script then writes no cues) |
| `--wait-minutes N` | how long to wait per track for Rekordbox's own analysis files (default 30) |
| `--plain` | plain log lines instead of the live table (also used automatically when output is not a terminal) |
| `--no-pause` | (all scripts) do not wait for Enter at the end |

**Cue options** (`predict_beatgrids_and_cues.py` and `predict_cues.py`):

| Option | Meaning |
|---|---|
| `--no-cues` | (combined script only) grid only |
| `--minimum-confidence P` | no cues for tracks whose drop probability is below P (they are still reported) |
| `--skip-likely-wrong` | no cues for tracks whose grid was flagged "Likely wrong". Default: cues for everything |
| `--allow-unanalysed` | (`predict_cues.py`) also use tracks the grid script did not analyse (grid comes from Rekordbox itself; unreliable) |
| `--recolor` | (`predict_cues.py`) give the cues this tool wrote earlier the library's colours per cue kind, then exit |

**Parallelism and connection:**

| Option | Meaning |
|---|---|
| `--force-number-of-jobs N` | use exactly N parallel workers. Default: adapt automatically to free CPU/RAM (start with 1) |
| `--database-path`, `--encryption-key`, `--share-directory` | override what `data/.env` says |

Every script has `--help` with the full list.

---

## 4. What to expect

### The live table

In a terminal the script draws a table that is updated in place: one row per song with its status (waiting for Rekordbox, queued, grid, cues, done, failed, skipped), a review flag, the detected BPM, the number of tempo sections and the number of cues written. Below it: songs done, failures, review count, elapsed and estimated time left.

Scroll with **Up/Down**, **PgUp/PgDn**, **Home/End**; **F** jumps back to following the running songs; **Ctrl+C** stops (finished songs keep their results). When the run ends, the full table is printed to your scrollback. Without a terminal (or with `--plain`) you get one log line per song instead.

Expect very roughly 10-25 seconds of work per track and worker (more on a slow CPU or with `--force-advanced`), with several tracks running in parallel (see [parallelism](#parallelism)).

### What gets written

| Where | What |
|---|---|
| Track's Rekordbox analysis file (`ANLZ0000.DAT`, the beat grid "PQTZ" tag) | the new beat grid. The original file is copied to `data/backups/beatgrid/<track id>_ANLZ0000.DAT` first |
| `master.db`, `DjmdContent.BPM` | the BPM of the new grid |
| `master.db`, `DjmdCue` | hot cues A-H and memory cues, **only on tracks that have no cues of your own** |
| Your existing cues | moved onto the nearest beat of the new grid if they are within the snap margin (grid and cue snapping can be switched off) |
| Playlists "Beatgrid: Likely Wrong" and "Beatgrid: Uncertain" | tracks the tool is not sure about (see below), created at the root of your playlists |
| `data/state/` | which tracks are done (so they are skipped next time), plus pending updates |
| `logs/cues/phrase_cues.jsonl` | the planned cues of the last run (also in a dry run) |

New cue rows are clones of an existing cue of the same kind in your library, so they get **your colours**. `master.db` is backed up before writing (`data/backups/`, the newest 5 are kept per script).

### Review playlists

The tool knows when it is unsure and says so instead of silently guessing:

- **Beatgrid: Likely Wrong**: no grid could be found, the tempo had to be taken from Rekordbox, the downbeat (which beat is "1") is a toss-up, or two internal methods disagree by 4 ms or more. In the author's library about 60% of these grids were actually correct, but this is where wrong grids hide.
- **Beatgrid: Uncertain**: everything softer (several tempo sections, a loose fit, an unsure downbeat, ...).

The "REVIEW" flag and the reason are shown in the table. Check those tracks by eye/ear, fix them if needed, and remove them from the playlist when done.

### Safety rules the scripts follow

- Tracks with **any cue you made yourself** get no cues from this tool, and your cues are never deleted or overwritten. Only cues written by this tool are replaced (`--force`).
- Tracks with several tempo sections get a grid, but no cues.
- Everything is dry-run-able, and the first thing written is a backup.

### Undoing things

Close Rekordbox, then restore the backup you need: copy a `master.db` backup from `data/backups/` over your live `master.db`, and/or copy `<track id>_ANLZ0000.DAT` back to the track's analysis folder (in Rekordbox's `share/PIONEER/USBANLZ/...`). Or restore the library backup you made in step 1 of [your first run](#2-your-first-run).

---

## 5. Will it work on my library?

**Short answer: for similar music probably yes (check with `--dry-run` and a handful of tracks); for anything else, expect to adapt and retrain.** Be honest with yourself about which of the two parts applies to you.

**The models were fitted to one person's library and one person's cue system.**

- *Beat grids.* The detectors (kick/snare/clap onsets on a bar-folded timeline) and several small learned models (phase ranker, fine timing correction, downbeat classifier) were trained on ~1,700 tracks of **bass music** (dubstep, riddim, tearout, DnB, UKG, 140) whose grids the author had corrected by hand. "Correct" here means *where the author aligns the grid by eye in Rekordbox's waveform*: on the snare/clap onsets. If you align differently (e.g. on the kick attack) the tool will be systematically off from your taste by a few milliseconds. Four-on-the-floor house/techno, live drumming, heavily swung or tempo-drifting music are outside what it was built and tested for.
- *Cues.* The cue models predict where the **drop** starts/ends and where the **fakeout** ends, and a small rule set (`lib/cues/cue_rules.py`) turns that into hot cues and memory cues according to the author's convention:
  - **D** = first downbeat of drop 1; **A, B, C** = 16, 8 and 4 bars before D (skipped if the intro is shorter).
  - **H** = first downbeat of drop 2; **E, F, G** = 16, 8 and 4 bars before H.
  - **Memory cues** on A, D, E, H, at the end of each drop, every 8 bars back to the start of the song, and one at the end of each fakeout (maximum 10 per track).
  
  If your cue system differs (different hot cue meaning, different spacing, other structure), the *prediction* part may still help, but the rules need rewriting and the models need to be retrained on **your** cues.
- *Hand-set cues and the grid.* Many assumptions come from the author's habits: tracks start on beat 1 (about 96% do), cues sit on multiples of 4 beats from beat 1, drops are multiples of 8 bars.

**How good is it where it was tuned?** (honest numbers, see [How it works](#6-how-it-works) for caveats)

| | Result | Evidence |
|---|---|---|
| Beat grid, never-seen tracks | 96 of 99 judged correct (grid within a few ms, right downbeat) | holdout of 99 "Liquid Riddim" tracks, never used for training |
| Beat grid, downbeat ("which beat is 1") | about 1.4% of tracks still wrong (23 of 1,697) | pooled cross-validation on the author's library |
| Beat grid, design target | at most 1% wrong ideally, under 5% acceptable; the rest should land in the review playlists | the goal the project was steered by |
| Cues, drop start D | 94.2% exactly right | out-of-fold on 1,559 tracks |
| Cues, hot cues A-D all correct | 93.9% | out-of-fold |
| Cues, hot cues A-H all correct | 89.8% | out-of-fold |
| Cues, memory cues | 84% recall, 91% precision | out-of-fold |

"Out-of-fold" means every number comes from models that did not see that track, but all of it is the *same* library and the *same* person, so it says little about yours. Your results will differ.

**Adapting it to your library** is a real project (a weekend, not a minute): export your corrected grids and your cues from your own Rekordbox database, retrain the models with the scripts in `dev/model_training/`, and review the cue rules. Start at [Retraining and development](#8-retraining-and-development). What you need: a few hundred to a thousand-plus tracks whose grids you have hand-corrected and whose cues follow one consistent system, in Rekordbox. Even without retraining, the beat grid part may already be an improvement over Rekordbox's own analysis for similar music, so try `predict_beatgrids.py --dry-run` on a few tracks and look at the result in Rekordbox before deciding.

---

## 6. How it works

### Pipeline in one picture

```
Rekordbox analyses the new track (BPM, waveform, its own grid)
        |
        v
[1] beat grid:  decode audio -> kick/snare events -> tempo sections -> grid position -> downbeat
        |        writes ANLZ .DAT (PQTZ) + BPM into master.db; review flags
        v
[2] cues:       features on that grid, per bar -> drop start/end + fakeout end -> cue rules
        |        writes hot cues A-H + memory cues into master.db (one commit per song)
        v
you restart Rekordbox
```

### Beat grid (`lib/beatgrid/`)

1. **Decode** the audio with ffmpeg and detect **kicks** (band-pass + matched filter) and **snares/claps** (2-12 kHz envelope jump).
2. **Tempo scan** in 30 s blocks (15 s hop) that fold events into a 4-beat bar; consistent stretches become **tempo sections**, each fitted with one constant BPM and an eighth-note lattice (precision 0.0002 BPM). Ghost sections (e.g. a beatless outro with a wobbling bass that looks like a tempo change) are dropped. BPMs within 0.1 of a whole number are snapped. Rekordbox's own BPM is trusted as a sanity anchor: if the detected tempo is a simple ratio of it (3/4, 4/3, 2, 1/2, ...) the tempo is re-fitted at Rekordbox's value.
3. **Grid position (phase)**: a learned ranker chooses among candidate positions (peaks of the folded kick/snare profile), then the anchor is refined onto the snare onset and a small learned regressor corrects the remaining millisecond-level offset (dead zone 1 ms, cap 15 ms).
4. **Downbeat** (which beat is "1"): a logistic-regression model over beat-position features, plus, for unsure tracks, a second model on bar-folded band energies.
5. **Container offset**: decoded time differs from Rekordbox time by the encoder delay (m4a: read from ffprobe, about 47.9 ms for AAC; wav: 0; mp3: found by correlating Rekordbox's own waveform file with the decoded envelope).
6. **Two passes**: the default is a "basic" pass; tracks it is unsure about (about 16% for timing, 8% for the downbeat) get the slower "advanced" models. `--basic-only` and `--force-advanced` override this.
7. **Review flags**: see above; they combine rule-based checks and the disagreement between basic and advanced.
8. **Writing**: only the `PQTZ` beat-grid tag of the `.DAT` file is replaced (rebuilt bytes are verified by a round-trip check first). Constant-grid tracks only; the tempo-change tag `PQT2` is not touched.

### Cues (`lib/cues/`)

1. **Features per bar** on the *stored* grid (the one just written): per-beat energies, per-eighth-note band features (10 bands), onset density, multi-scale novelty, genre-playlist hints (the playlists a track is in).
2. **Drop start / drop end**: every bar is a candidate; two gradient-boosting models score "a drop starts here" and "a drop ends here"; a small decoder picks the combination that fits the typical structure (one or two drops, lengths in multiples of 8 bars).
3. **Fakeout end** (the part at the start of a drop where not everything plays): a separate model.
4. **Cue rules** (`cue_rules.py`) derive hot cues A-H and the memory cues from D, e1 (end of drop 1), H, e2 and the fakeout positions, as listed under [Will it work on my library?](#5-will-it-work-on-my-library).
5. **Writing**: new `DjmdCue` rows cloned from existing ones in your library (colours!), cue positions in ms and frames, hot cue kinds A, B, C = 1-3 and D-H = 5-9, memory = 0.

### Parallelism

Songs are analysed in parallel with a process pool, and the number of workers **adapts by itself** (`lib/adaptive_workers.py`). It measures how much CPU and RAM is used by its own workers, by Rekordbox (which is usually analysing at the same time) and by everything else. From that it works out how many workers fit under 85% CPU and 95% RAM and moves toward that number gradually (at most a third of your physical cores at a time). Short spikes of other programs are ignored; when Rekordbox finishes its analysis the controller notices within seconds and ramps up. `--force-number-of-jobs N` switches this off.

The scripts also wait for Rekordbox's own analysis files per track (they contain the waveform used for the mp3 offset), so you can run them right after importing.

---

## 7. Project layout and data folders

```
SETUP (Windows).bat / SETUP (MacOS).command    one-click installation (section 1): install Python 3.14 if missing, then run lib/setup_install.py
main/        the three scripts you run (see section 3)
  other helpful tools/   small standalone helpers (see "Other helpful tools" below)
lib/         all importable code
  rekordbox/   database access, ANLZ file bytes, cue snapping, track selection, ground truth, Rekordbox program-folder detection (install_locations.py)
  beatgrid/    detection pipeline, learned rankers, advanced models, review flags, grid writer
  cues/        drop/fakeout predictors, feature extraction, cue rules, cue export
  adaptive_workers.py, progress_table.py, audio_features.py, paths.py, ffmpeg_tools.py, populate_env.py and setup_install.py (used by the SETUP files), interactive_selection.py, exit_pause.py
data/        runtime data (not for git: .env, state/, backups/, cache/, ffmpeg/); models/ holds the trained models
logs/        reports of the last runs (not for git)
debug/       generated analysis output of the dev scripts (not for git)
dev/         tools to evaluate and retrain (section 8)
docs/        project history and hand-over notes
```

`lib/paths.py` is the single place that defines every path. Settings live in `data/.env`; besides the two Rekordbox entries and `FFMPEG_PATH` / `FFPROBE_PATH` (the ffmpeg programs to use), optional keys are `CUE_SNAP_MARGIN_MS`, `BEATGRID_ANALYSIS_MODE` (`auto`/`basic`/`advanced`, normally set with the command line flags) and a few `BEATGRID_*` tuning fractions.

### Other helpful tools

Standalone helpers in `main/other helpful tools/`. They share the project's `.env` and libraries, so run SETUP first.

| script | what it does |
|---|---|
| `snap_cues_to_grid.py` | Moves every memory cue, hot cue and loop end of the whole collection onto the nearest beat of its grid when it is within `CUE_SNAP_MARGIN_MS` (default 10 ms); cues farther away stay as they are. Updates master.db and the ANLZ files. Backups go to `data/backups/snap/`, the report to `logs/`. Close Rekordbox first, use `--dry-run` to only report. |
| `kill_rekordbox_samples.py` | Renames the demo/sample files Rekordbox installs (`DemoTrack.zip`, `GROOVE_CIRCUIT_FACTORY_SAMPLES.spp`, `MERGE FX.zip`, `OSC_SAMPLER.zip`) to `<name>.old` in every installed Rekordbox version. Run it again after each Rekordbox update. `--dry-run` only reports, `--undo` restores. On Windows, Program Files is protected: when a rename is denied the script asks for administrator rights itself (UAC prompt) and re-runs in an elevated window (`--no-elevate` turns that off). Close Rekordbox first. It only renames files and never touches the Rekordbox database, so remove the demo tracks from your collection by hand in Rekordbox. On macOS the exact location of these files inside the app bundle is not confirmed, so check with `--dry-run` first. |

Both start with a double click or `python "main/other helpful tools/<script>.py"`. The Rekordbox program folder used by the second one is `REKORDBOX_INSTALL_PATH` in `data/.env`; SETUP detects it (default Program Files / Pioneer folders, `/Applications`, pyrekordbox's configuration, the Windows registry, a running Rekordbox) and you can override it by hand or with `--install-path`. It is the base folder that holds the versioned folders (for example `C:\Program Files\rekordbox`), so it keeps working after an update.

Dev scripts expect their inputs in `debug/` subfolders (all git-ignored):

| folder | what goes in |
|---|---|
| `debug/eval_out/` | output of `dev/tools/evaluate_beatgrid_on_library.py` |
| `debug/eval_reports/` | the `eval_report*.json` files you want to train/evaluate on |
| `debug/event_caches/<set>/` | `<track id>.ff2.pkl` event caches |
| `debug/beat_features/` | per-track `.npz` from `--beat-features` |
| `debug/genre_playlists/` | Rekordbox playlist exports used for genre labels |
| `debug/cue_training/` | `cues.jsonl`, `beat_features_slim.pkl.gz`, band feature dumps |
| `debug/work/` | intermediates and newly trained models (copy to `data/models/` only after checking) |

---

## 8. Retraining and development

Everything under `dev/` is for developing and retraining; none of it is needed to run the analysis.

| Folder | Contents |
|---|---|
| `dev/tools/` | export and evaluation scripts: `evaluate_beatgrid_on_library.py` (run the pipeline on your library and compare with the grids stored in Rekordbox), `export_cues_for_training.py`, `extract_band_features.py`, `export_ground_truth_examples.py`, ... |
| `dev/model_training/beatgrid/` | build the training rows, cross-validate and train the phase ranker, fine adjust, downbeat and advanced models |
| `dev/model_training/cues/` | build the cue dataset, train the drop-start/drop-end/fakeout models, check the decoder and the cue rules. Step-by-step: [`README_training.md`](dev/model_training/cues/README_training.md) |
| `dev/evaluation/` | replay the pipeline on cached features, compare against ground truth, measure review flags |
| `dev/experiments/`, `dev/diagnostics/` | research scripts and plots (some results were negative and are documented as such in `docs/`) |
| `dev/archive/` | superseded scripts, kept for reference |

The broad recipe to adapt it to your library:

1. Correct the grids of a few hundred tracks by hand in Rekordbox and set your cues the way you want the tool to set them. These stored grids and cues are the **ground truth** the models learn from.
2. Run `dev/tools/evaluate_beatgrid_on_library.py` to get reports and event caches for those tracks, and `dev/tools/export_cues_for_training.py` for the cue labels.
3. Retrain with the scripts in `dev/model_training/beatgrid/` and `dev/model_training/cues/`; always look at the **cross-validation** numbers, not the in-sample ones.
4. Copy new models from `debug/work/` to `data/models/` only after checking them. Adapt `lib/cues/cue_rules.py` to your cue convention.
5. Keep a few tracks aside that are never trained on and judge by them.

Scripts that need many inputs say what they need in their docstring or `--help`. All scripts are meant to be run from the project root.

---

## 9. Known limitations

- **Tested on Windows 10/11 with Rekordbox 7 and Python 3.14.** The terminal UI has Windows and POSIX key handling, but a run on macOS has not been tested with a real Rekordbox library. "Is Rekordbox running?" is only detected on Windows.
- **Music.** Built for bass music with a strong kick/snare grid. Acapellas, ambient, beatless and continuously tempo-changing tracks are not handled (they may get flagged or a wrong grid).
- **One constant grid per tempo section.** Gradual tempo changes are not modelled. Only the `PQTZ` grid tag is rewritten; a populated `PQT2` tag (tempo-change tracks) has never been seen and is left alone.
- **Cues** are only written for tracks without cues of their own and with a single tempo section. Tracks with a very unusual structure (several short drops, vocal/melodic tracks that follow a different cue system) will get plausible but wrong cues. Drop 1 matters most; drop 2 and fakeout placement are less accurate.
- **Not an official tool.** This project is not affiliated with or endorsed by AlphaTheta/Pioneer DJ. It modifies Rekordbox's database through [pyrekordbox](https://github.com/dylanljones/pyrekordbox); a Rekordbox update can change the database format. Use at your own risk and keep backups.

### Troubleshooting

| Problem | What to do |
|---|---|
| `REKORDBOX_DATABASE_PATH missing` | run the setup again, or pass `--database-path` / `--encryption-key` |
| ffmpeg / ffprobe not found | run the setup again (it finds or downloads ffmpeg), or set `FFMPEG_PATH` and `FFPROBE_PATH` in `data/.env` to the full paths of the programs |
| "Rekordbox analysis files never appeared" (timeout) | Rekordbox has not analysed that track (is auto-analysis on? does the file exist?). Raise `--wait-minutes` or analyse the track in Rekordbox and run again |
| "psutil is not installed ... using 2 workers" | `pip install psutil`, or give `--force-number-of-jobs N` |
| New grid/cues do not show in Rekordbox | restart Rekordbox; load the track again |
| A shortcut does nothing or says the file was not found | the program folder was moved or renamed: run the setup again |
| A track was skipped | the table's note column says why (audio file missing, has cues of its own, several tempo sections, ...) |
| Results look systematically off by a few ms | your alignment convention differs from the one the models learned; see [Will it work on my library?](#5-will-it-work-on-my-library) |

---

## License

Licensed under the [Apache License 2.0](LICENSE) **with the Commons Clause**. In plain words: you may use, modify and share this software, including for professional work (DJing, studio, library services for yourself). You may **not sell** it, modified or not, or sell a product or service whose value comes substantially from it, unless you have agreed that with the author in writing. This makes it source-available, not "open source" in the OSI sense. Contributions are accepted on the condition that the author may license them under these same terms.