import os

class Config:
    EXPERIMENT_NAME = "torchvision.mobilenet_v2 for cats_and_dogs classification experiment"
    RUN_NAME = "default"
    RANDOM_SEED = 42
    DATASET_NAME = "Oxford-IIIT Pet dataset"
    DATASET_LINK = "https://www.robots.ox.ac.uk/~vgg/data/pets/"
    DATASET_LABELS = "annotations.csv"
    MODEL_NAME = "cats_and_dogs_mobilenet_v2"
    MODEL_WEIGHTS = "IMAGENET1K_V1"
    LR = 1e-3
    MIN_DELTA = 0.05
    PATIENCE = 10
    MAX_EPOCHS = 35
    BATCH_SIZE = 256
    NUM_WORKERS = 2
    JSON_PATH = "data/labels.json"
    CSV_PATH = "data/annotations.csv"