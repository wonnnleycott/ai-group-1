import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # main project folder
sys.path.insert(0, str(ROOT))                           # so we can import Part 1 files
from joint_limits import lower, upper                   # robot joint limits 

GAIT_FILE = ROOT / "best_gait.csv"                      # made by ga_analysis.py 
OUT_FILE = Path(__file__).resolve().parent / "dataset.npz"
TEST_EVERY = 5                                       


def scale_angles(angles):
    return (angles - lower) / (upper - lower)


def unscale_angles(scaled):
    return lower + scaled * (upper - lower)


def make_inputs(n_frames):
    t = np.arange(n_frames) / n_frames
    return np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])


def build():
    gait = np.loadtxt(GAIT_FILE, delimiter=",")      
    n_frames = gait.shape[0]

    X = make_inputs(n_frames)                       
    Y = scale_angles(gait)                           

    frame_ids = np.arange(n_frames)
    test_mask = frame_ids % TEST_EVERY == 0             
    train_mask = ~test_mask                    

    np.savez(OUT_FILE,
             X_train=X[train_mask], Y_train=Y[train_mask],
             X_test=X[test_mask], Y_test=Y[test_mask],
             X_all=X, Y_all=Y)

    print("Walk loaded:", gait.shape)
    print("Inputs X:", X.shape, " Targets Y:", Y.shape)
    print("Training frames:", int(train_mask.sum()), " Test frames:", int(test_mask.sum()))
    print("Scaled targets between", round(float(Y.min()), 3), "and", round(float(Y.max()), 3))
    print("Saved to", OUT_FILE)


if __name__ == "__main__":
    build()