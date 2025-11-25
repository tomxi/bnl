import random

from tqdm import tqdm

import bnl

if __name__ == "__main__":
    slm_ds = bnl.Dataset("/scratch/qx244/data/salami/metadata.csv")
    tids = slm_ds.track_ids
    random.shuffle(tids)

    for tid in tqdm(tids):
        try:
            relevance = slm_ds[tid].lsd_relevance()
        except Exception as e:
            print(f"Failed to compute scores for {tid}: {e}")
