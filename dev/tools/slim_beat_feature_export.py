# slim_beat_feature_export.py - keep only the small keys (BE, info_*, label_*) of every beat_features npz -> beat_features_slim.pkl.gz
import sys, pickle, gzip, glob, os
import numpy as np
src = sys.argv[1] if len(sys.argv) > 1 else "eval_out/beat_features"
out = {}
for f in glob.glob(os.path.join(src, "*.npz")):
    z = np.load(f)
    out[os.path.basename(f)[:-4]] = {k: z[k] for k in z.files if k == "BE" or k.startswith(("info_", "label_"))}
with gzip.open("beat_features_slim.pkl.gz", "wb") as fh:
    pickle.dump(out, fh)
print(len(out), "tracks ->", os.path.getsize("beat_features_slim.pkl.gz") / 1e6, "MB")
