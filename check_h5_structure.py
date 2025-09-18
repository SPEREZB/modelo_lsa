import os
import h5py

data_dir = "data/keypoints"

for filename in os.listdir(data_dir):
    if filename.endswith(".h5"):
        filepath = os.path.join(data_dir, filename)
        print(f"\n📂 {filename}")
        with h5py.File(filepath, "r") as f:
            print("Grupos en el archivo:", list(f.keys()))
            for key in f.keys():
                try:
                    shape = f[key].shape
                    dtype = f[key].dtype
                    print(f" - {key}: shape={shape}, dtype={dtype}")
                except:
                    print(f" - {key}: no es dataset")
