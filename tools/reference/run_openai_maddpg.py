"""Run the archived OpenAI MADDPG/DDPG program with an explicit random seed.

This wrapper deliberately does not copy or modify reference code.  It seeds Python,
NumPy, and TensorFlow, then executes ``experiments/train.py`` from a separately cloned
OpenAI repository.  Use the Python-3.7/TensorFlow-1 environment documented by the
reference project's own requirements.

Example:
    python tools/reference/run_openai_maddpg.py --source /path/to/openai/maddpg \
        --seed 11 -- --scenario simple_spread --good-policy ddpg \
        --num-episodes 10000 --exp-name ddpg_seed11
"""


import argparse
import os
import random
import runpy
import sys

import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("reference_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    reference_args = args.reference_args[1:] if args.reference_args[:1] == ["--"] else args.reference_args

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    os.environ["SUPPRESS_MA_PROMPT"] = "1"

    source = os.path.abspath(args.source)
    sys.path.insert(0, source)
    sys.argv = [os.path.join(source, "experiments", "train.py")] + reference_args
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
