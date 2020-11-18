#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import argparse

from _test_commons import run_subprocess

import logging

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s",
    level=logging.DEBUG)
log = logging.getLogger("Build")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", help="cwd")
    return parser.parse_args()

def run_checkpoint_tests(cwd, log):
    log.debug('Running: Checkpoint tests')

    command = [sys.executable, 'orttraining_test_checkpoint.py']
    
    return run_subprocess(command, cwd=cwd, log=log)

def main():
    import torch
    ngpus = torch.cuda.device_count()

    if ngpus <= 1:
        return 1

    args = parse_arguments()
    cwd = args.cwd

    log.info("Running distributed tests pipeline")

    succeed_flag = True

    succeed_flag = succeed_flag and run_checkpoint_tests(cwd, log)

    return succeed_flag


if __name__ == "__main__":
    sys.exit(main())