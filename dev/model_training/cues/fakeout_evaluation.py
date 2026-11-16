import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pickle,numpy as np,collections,sys
fakeout_rows,is_fakeout_end,group_index=pickle.load(open(str(paths.WORK / 'fakeout_rows.pkl'),'rb'));out_of_fold_probabilities=np.load(sys.argv[1] if len(sys.argv)>1 else str(paths.WORK / 'fakeout_oof_probabilities.npy'))
group_rows=[np.where(group_index==group)[0] for group in range(len(fakeout_rows))]
max_probability=np.array([out_of_fold_probabilities[i].max() for i in group_rows]);has_fakeout=np.array([is_fakeout_end[i].sum()>0 for i in group_rows]);drop_kind=np.array([row[1] for row in fakeout_rows])
predicted_position=np.array([int(np.argmax(out_of_fold_probabilities[i])) for i in group_rows]);true_position=np.array([int(np.argmax(is_fakeout_end[i])) if is_fakeout_end[i].sum() else -9 for i in group_rows])
from sklearn.metrics import roc_auc_score
for drop_number in (0,1):
    selection=drop_kind==drop_number;print('drop',drop_number+1,'n',selection.sum(),'fk',has_fakeout[selection].sum(),'AUC hasfk(maxp)',round(roc_auc_score(has_fakeout[selection],max_probability[selection]),3))
    for threshold in (0.2,0.3,0.4,0.5,0.6):
        predicted_has_fakeout=max_probability[selection]>threshold;exact_correct=np.where(has_fakeout[selection],predicted_has_fakeout&(predicted_position[selection]==true_position[selection]),~predicted_has_fakeout);within_one_correct=np.where(has_fakeout[selection],predicted_has_fakeout&(abs(predicted_position[selection]-true_position[selection])<=1),~predicted_has_fakeout)
        print(' th',threshold,'overall exact %.3f within1 %.3f | hasfk acc %.3f | fk exact %.3f | fk exact|pred %.3f'%(exact_correct.mean(),within_one_correct.mean(),(predicted_has_fakeout==has_fakeout[selection]).mean(),(predicted_has_fakeout&(predicted_position[selection]==true_position[selection]))[has_fakeout[selection]].mean(),(predicted_position[selection]==true_position[selection])[has_fakeout[selection]].mean()))
