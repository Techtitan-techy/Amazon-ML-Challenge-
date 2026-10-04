import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "6ab10eb3b23ba_student_resource", "student_resource", "dataset")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

TRAIN_S1 = os.path.join(DATA_DIR, "train", "train_source1.tsv").replace("\\", "/")
TRAIN_S2 = os.path.join(DATA_DIR, "train", "train_source2.tsv").replace("\\", "/")
TRAIN_S3 = os.path.join(DATA_DIR, "train", "train_source3.tsv").replace("\\", "/")
TRAIN_GT = os.path.join(DATA_DIR, "train", "train_ground_truth.tsv").replace("\\", "/")

TEST_S1 = os.path.join(DATA_DIR, "test", "test_source1.tsv").replace("\\", "/")
TEST_S2 = os.path.join(DATA_DIR, "test", "test_source2.tsv").replace("\\", "/")
TEST_S3 = os.path.join(DATA_DIR, "test", "test_source3.tsv").replace("\\", "/")

os.makedirs(REPORTS_DIR, exist_ok=True)
