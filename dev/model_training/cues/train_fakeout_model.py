"""Train the production fakeout-end model on all rows (needs fk_X.npy + fk_rows.pkl from fk_cv.py). Writes models/fk_model.pkl"""
import _bootstrap  # noqa: F401  (puts the project root on sys.path)
from lib import paths
import pickle,numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
fakeout_rows,is_fakeout_end,group_index=pickle.load(open(str(paths.WORK / 'fakeout_rows.pkl'),'rb'));features=np.load(str(paths.WORK / 'fakeout_features.npy'),mmap_mode='r')
random_generator=np.random.RandomState(1);training_rows=np.where((is_fakeout_end==1)|(random_generator.rand(len(is_fakeout_end))<0.35))[0]
model=HistGradientBoostingClassifier(max_iter=200,learning_rate=0.08,max_leaf_nodes=24,l2_regularization=1.0,random_state=0).fit(features[training_rows],is_fakeout_end[training_rows])
pickle.dump(model,open(str(paths.WORK / 'fakeout_end_model.pkl'),'wb'));print('ok',features.shape)
