"""Single source of truth for the 9 challenge tasks and their keypoint counts.

Previously TASK_KEYPOINTS was copied verbatim into three model files and the task
name set into two more places. Everything now imports from here.
"""

# Number of landmarks per task (used for MRE and as the per-task head output width).
TASK_KEYPOINTS = {
    "A4C": 16,
    "AOP": 4,
    "FA": 4,
    "FUGC": 2,
    "HC": 4,
    "IVC": 2,
    "PLAX": 22,
    "PSAX": 4,
    "fetal_femur": 2,
}

# Deterministic ordering used by evaluation/visualization (alphabetical-ish, kept
# identical to the original visualization/common.py TASK_ORDER for reproducibility).
TASK_ORDER = ["A4C", "AOP", "FA", "FUGC", "HC", "IVC", "PLAX", "PSAX", "fetal_femur"]

VALID_TASKS = set(TASK_KEYPOINTS.keys())

NUM_TASKS = len(TASK_KEYPOINTS)
