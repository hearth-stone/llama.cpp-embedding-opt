#!/usr/bin/env bash
set -euo pipefail

dst_dir="$(basename "$PWD")"

rsync -avzP \
    --copy-links \
    --exclude='.git' \
    --exclude='.git/' \
    --exclude='.DS_Store' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='*.so' \
    --exclude='*.dylib' \
    --exclude='*.o' \
    --exclude='build/' \
    --exclude='dist/' \
    --exclude='*.egg-info/' \
    --exclude='.eggs/' \
    --exclude='.hypothesis/' \
    --exclude='.pytest_cache/' \
    --exclude='.codebuddy/' \
    --exclude='.claude/' \
    --exclude='.gemini/' \
    --exclude='.venv/' \
    --exclude='.codegraph/' \
    ./ "Arm-codex:/home/zhangxu/codex/$dst_dir"
