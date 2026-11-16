"""Rerun detect() on all cached tracks of an eval_out and compare with the report: python dev/evaluation/rerun_detection_on_cache.py debug/eval_out"""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
import sys,json,numpy as np
from pathlib import Path
from lib.beatgrid import track_job
from lib.beatgrid import grid_pipeline
eval_out_folder=(sys.argv[1] if len(sys.argv)>1 else 'eval_out').rstrip('/')+'/'
report=json.load(open(eval_out_folder+'eval_report.json',encoding='utf-8'))
changes=[];track_count=0
for report_row in report:
    events,sections=track_job.detect(eval_out_folder+'cache',None,report_row['id'],report_row.get('rekordbox_bpm'))
    track_count+=1
    old_sections=report_row['sections']
    for old_section,section in zip(old_sections,sections):
        anchor_difference_ms=(section['anchor_s']*1000-old_section['anchor_ms'])
        if abs(anchor_difference_ms)>0.5: changes.append((report_row['artist'][:16]+' - '+report_row['title'][:22],round(anchor_difference_ms,1),section['parity']['method'],section['parity'].get('bass_off_on'),report_row['dry_run']['diff_median_ms']))
    if len(old_sections)!=len(sections): changes.append((report_row['title'],'nsec',len(old_sections),len(sections)))
print(track_count,'tracks; anchor changes:');[print(' ',change) for change in changes]
