import pickle
import gzip
from tqdm import tqdm

path = 'monte_rnd_full_trajectories.pkl.gz'
dump_path = 'monte_rnd_good_trajs.pkl.gz'
skip = 9600

with gzip.open(path, 'rb') as f:
    print(f"[+] Skipping {skip} trajectories...")
    for _ in tqdm(range(skip)):
        traj = pickle.load(f)

    print(f"[+] Dumping trajectories...")
    with gzip.open(dump_path, "ab") as df:
        try:
            while True:
                traj = pickle.load(f)
                pickle.dump(traj, df)
        except EOFError:
            pass
