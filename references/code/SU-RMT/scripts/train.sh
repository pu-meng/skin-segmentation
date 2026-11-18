#!/usr/bin/env bash

uv run python train.py --dataset_name Synapse --id SURMT --epochs 250 --gpu_id 0 --lr 3e-5 --top_k 3
uv run python train.py --dataset_name Polyp --id SURMT --epochs 200 --gpu_id 0 --lr 3e-5 --top_k 2

uv run python train.py --dataset_name ISIC2017 --id SURMT --epochs 180 --gpu_id 0 --lr 5e-5 --top_k 5
uv run python train.py --dataset_name ISIC2018 --id SURMT --epochs 180 --gpu_id 0 --lr 5e-5 --top_k 5
