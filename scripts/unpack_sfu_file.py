from pathlib import Path
import tarfile

def unpack_sfu(archive="SFU.tar.bz2", out_dir="data/sfu_raw"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as t:
        t.extractall(out)
    files = sorted(out.glob("SFU/*/*_poses.npz"))
    print(f"extracted {len(files)} motions")
    return files

if __name__ == "__main__":
    unpack_sfu()