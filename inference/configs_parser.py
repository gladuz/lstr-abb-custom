# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from yacs.config import CfgNode as CN
import os.path as osp
import argparse
import json

# ---------------------------------------------------------------------------- #
# Config Definition
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# Metadata
# ---------------------------------------------------------------------------- #
_C.SEED = 0

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.MODEL_NAME = ''
_C.MODEL.CHECKPOINT = ''

# ---------------------------------------------------------------------------- #
# Feature Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FEATURE_HEAD = CN()
_C.MODEL.FEATURE_HEAD.LINEAR_ENABLED = True
_C.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES = 1024

# ---------------------------------------------------------------------------- #
# Transformer Network
# ---------------------------------------------------------------------------- #
_C.MODEL.LSTR = CN()
# Hyperparameters
_C.MODEL.LSTR.NUM_HEADS = 8
_C.MODEL.LSTR.DIM_FEEDFORWARD = 1024
_C.MODEL.LSTR.DROPOUT = 0.2
_C.MODEL.LSTR.ACTIVATION = 'relu'
# Memory choices
_C.MODEL.LSTR.AGES_MEMORY_SECONDS = 0
_C.MODEL.LSTR.AGES_MEMORY_SAMPLE_RATE = 1
_C.MODEL.LSTR.LONG_MEMORY_SECONDS = 0
_C.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE = 1
_C.MODEL.LSTR.WORK_MEMORY_SECONDS = 8
_C.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE = 1
# Design choices
_C.MODEL.LSTR.ENC_MODULE = [
    [16, 1, True], [32, 2, True]
]
_C.MODEL.LSTR.DEC_MODULE = [-1, 2, True]
# Inference modes
_C.MODEL.LSTR.INFERENCE_MODE = 'batch'

# ---------------------------------------------------------------------------- #
# Criterion
# ---------------------------------------------------------------------------- #
_C.MODEL.CRITERIONS = [['MCE', {}]]

# ---------------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------------- #
_C.DATA = CN()
_C.DATA.DATA_INFO = 'data/data_info.json'
_C.DATA.DATA_NAME = None
_C.DATA.DATA_ROOT = None
_C.DATA.CLASS_NAMES = None
_C.DATA.NUM_CLASSES = None
_C.DATA.IGNORE_INDEX = None
_C.DATA.METRICS = None
_C.DATA.FPS = 5
_C.DATA.TRAIN_SESSION_SET = None
_C.DATA.TEST_SESSION_SET = None

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.MODALITY = 'twostream'
_C.INPUT.VISUAL_FEATURE = 'rgb_anet_resnet50'
_C.INPUT.MOTION_FEATURE = 'flow_anet_resnet50'
_C.INPUT.TARGET_PERFRAME = 'target_perframe'

# ---------------------------------------------------------------------------- #
# Data Loader
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()
_C.DATA_LOADER.BATCH_SIZE = 32
_C.DATA_LOADER.NUM_WORKERS = 4
_C.DATA_LOADER.PIN_MEMORY = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.START_EPOCH = 1
_C.SOLVER.NUM_EPOCHS = 50

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.SOLVER.OPTIMIZER = 'adam'
_C.SOLVER.BASE_LR = 0.00005
_C.SOLVER.WEIGHT_DECAY = 0.00005
_C.SOLVER.MOMENTUM = 0.9

# ---------------------------------------------------------------------------- #
# Scheduler
# ---------------------------------------------------------------------------- #
_C.SOLVER.SCHEDULER = CN()
_C.SOLVER.SCHEDULER.SCHEDULER_NAME = 'multistep'
_C.SOLVER.SCHEDULER.MILESTONES = []
_C.SOLVER.SCHEDULER.GAMMA = 0.1
_C.SOLVER.SCHEDULER.WARMUP_FACTOR = 0.3
_C.SOLVER.SCHEDULER.WARMUP_EPOCHS = 10.0
_C.SOLVER.SCHEDULER.WARMUP_METHOD = 'linear'

# ---------------------------------------------------------------------------- #
# Others
# ---------------------------------------------------------------------------- #
_C.SOLVER.PHASES = ['train', 'test']

# ---------------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = 'checkpoints'
_C.SESSION = ''

# ---------------------------------------------------------------------------- #
# Misc
# ---------------------------------------------------------------------------- #
_C.VERBOSE = False


def get_cfg():
    return _C.clone()


def assert_and_infer_cfg(cfg, gpu, config_file):
    # Setup the visible devices
    cfg.GPU = gpu

    # Infer data info
    # with open(cfg.DATA.DATA_INFO, 'r') as f:
    #     data_info = json.load(f)[cfg.DATA.DATA_NAME]

    # cfg.DATA.DATA_ROOT = data_info['data_root'] if cfg.DATA.DATA_ROOT is None else cfg.DATA.DATA_ROOT
    # cfg.DATA.CLASS_NAMES = data_info['class_names'] if cfg.DATA.CLASS_NAMES is None else cfg.DATA.CLASS_NAMES
    # cfg.DATA.NUM_CLASSES = data_info['num_classes'] if cfg.DATA.NUM_CLASSES is None else cfg.DATA.NUM_CLASSES
    # cfg.DATA.IGNORE_INDEX = data_info['ignore_index'] if cfg.DATA.IGNORE_INDEX is None else cfg.DATA.IGNORE_INDEX
    # cfg.DATA.METRICS = data_info['metrics'] if cfg.DATA.METRICS is None else cfg.DATA.METRICS
    # cfg.DATA.FPS = data_info['fps'] if cfg.DATA.FPS is None else cfg.DATA.FPS
    # cfg.DATA.TRAIN_SESSION_SET = data_info['train_session_set'] if cfg.DATA.TRAIN_SESSION_SET is None else cfg.DATA.TRAIN_SESSION_SET
    # cfg.DATA.TEST_SESSION_SET = data_info['test_session_set'] if cfg.DATA.TEST_SESSION_SET is None else cfg.DATA.TEST_SESSION_SET

    # Input assertions
    assert cfg.INPUT.MODALITY in ['visual', 'motion', 'twostream']

    # Infer memory
    if cfg.MODEL.MODEL_NAME in ['LSTR']:
        cfg.MODEL.LSTR.AGES_MEMORY_LENGTH = cfg.MODEL.LSTR.AGES_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.LONG_MEMORY_LENGTH = cfg.MODEL.LSTR.LONG_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.WORK_MEMORY_LENGTH = cfg.MODEL.LSTR.WORK_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.TOTAL_MEMORY_LENGTH = \
            cfg.MODEL.LSTR.AGES_MEMORY_LENGTH + \
            cfg.MODEL.LSTR.LONG_MEMORY_LENGTH + \
            cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        assert cfg.MODEL.LSTR.AGES_MEMORY_LENGTH % cfg.MODEL.LSTR.AGES_MEMORY_SAMPLE_RATE == 0
        assert cfg.MODEL.LSTR.LONG_MEMORY_LENGTH % cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE == 0
        assert cfg.MODEL.LSTR.WORK_MEMORY_LENGTH % cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE == 0
        cfg.MODEL.LSTR.AGES_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.AGES_MEMORY_LENGTH // cfg.MODEL.LSTR.AGES_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH // cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH // cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.TOTAL_MEMORY_NUM_SAMPLES = \
            cfg.MODEL.LSTR.AGES_MEMORY_NUM_SAMPLES + \
            cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES + \
            cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES

        assert cfg.MODEL.LSTR.INFERENCE_MODE in ['batch', 'stream']

    # Infer output dir
    config_name = osp.splitext(config_file)[0].split('/')[1:]
    cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, *config_name)
    if cfg.SESSION:
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, cfg.SESSION)


def load_cfg(config_file, opts, gpu):
    cfg = get_cfg()
    if config_file is not None:
        cfg.merge_from_file(config_file)
    if opts is not None:
        cfg.merge_from_list(opts)
    assert_and_infer_cfg(cfg, gpu, config_file)
    return cfg
