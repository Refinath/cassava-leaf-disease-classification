"""Project configurations
"""

TRAIN_PATH = ""
TEST_PATH = ""
DEBUG = False
APEX = False
PRINT_FREQ = 100
NUM_WORKERS = 4
MODEL_NAME = "tf_efficientnet_b4_ns"
SIZE = 512
# ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
SCHEDULER = "CosineAnnealingWarmRestarts"
CRITERION = "BiTemperedLoss"  # ['CrossEntropyLoss', LabelSmoothing', 'FocalLoss',
#  'FocalCosineLoss', 'SymmetricCrossEntropyLoss', 'BiTemperedLoss', 'TaylorCrossEntropyLoss']
EPOCHS = 10
# factor=0.2 # ReduceLROnPlateau
# patience=4 # ReduceLROnPlateau
# eps=1e-6 # ReduceLROnPlateau
# T_max=10 # CosineAnnealingLR
T_0 = 10  # CosineAnnealingWarmRestarts
LR = 1e-4
MIN_LR = 1e-6
BATCH_SIZE = 12
WEIGHT_DECAY = 1e-6
GRADIENT_ACCUM_STEPS = 1
MAX_GRAD_NORM = 1000
SEED = 42
TARGET_SIZE = 5  # Num of Classes
LABEL_MAP = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}
TARGET_COL = "label"
N_FOLD = 5
TRN_FOLD = [0]  # Change Values according to the fold which you are training.
TRAIN = True
INFERENCE = False
SMOOTHING = 0.05
# bi-tempered-loss https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/202017
T1 = 0.3
T2 = 1.0
