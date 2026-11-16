import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import json,collections,os
from dev.model_training.beatgrid.track_metrics import ACAPELLA_PATTERN, TEMPO_CHANGE_PATTERN, failure_category
playlists_folder=str(paths.GENRE_PLAYLISTS)+'/'
def load_playlist_tracks(playlist_file):
    lines=open(playlists_folder+playlist_file,'rb').read().decode('utf-16').splitlines()[1:]
    return {(line.split('\t')[1]+' - '+line.split('\t')[2]).strip().lower() for line in lines if line.count('\t')>2}
genre_track_sets={}
for genre_name,playlist_files in (('dubstep',['Dubstep.txt']),('riddim',['Riddim.txt']),('tearout',['Tearout.txt']),('140',['140.txt']),('dnb',['DnB.txt','Liquid Dnb&Jungle.txt']),('jungle',['Jungle.txt']),('2step',['2-Step.txt']),('ukg',['UKG (4x4 only).txt'])):
    genre_track_sets[genre_name]=set().union(*[load_playlist_tracks(playlist_file) for playlist_file in playlist_files])
def genre_of_track(report_row):
    track_key=(report_row['artist']+' - '+report_row['title']).strip().lower()
    for genre_name in ('tearout','riddim','jungle','dnb','2step','ukg','140','dubstep'):
        if track_key in genre_track_sets[genre_name]: return genre_name
    return 'unknown'
def load_json(path): return json.load(open(path))
reports_folder=str(paths.EVAL_REPORTS)+'/'
REPORT_SETS={set_name:load_json(reports_folder+report_file) for set_name,report_file in (('old961','eval_report.json'),('new698','eval_report_-_698_dubstep_songs.json'),('extra','eval_report__extra_subgenres_.json')) if os.path.exists(reports_folder+report_file)}
PHASE_RANKER_FOLD_TRACK_IDS=set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'phase_ranker_folds.json'))))
if __name__=='__main__':
    for set_name,report in REPORT_SETS.items():
        genre_counts=collections.Counter(); phase_select_count=sum(any('phase_select' in section for section in row['sections']) for row in report); fine_shift_count=sum(any(section.get('fine_shift_ms') is not None for section in row['sections']) for row in report)
        print(set_name,len(report),'phase_select',phase_select_count,'fine_shift',fine_shift_count,'overlap with old ids',len({row['id'] for row in report}&{row['id'] for row in REPORT_SETS['old961']}) if set_name!='old961' else '')
        print(collections.Counter(genre_of_track(row) for row in report))
