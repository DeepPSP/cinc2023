#!/bin/sh
black . -v --exclude="/build|dist|official_baseline_classifier|official_scoring_metric|artifact_pipeline|helper_code\.py|run_model\.py|train_model\.py|evaluate_model\.py|remove_data\.py|remove_labels\.py|truncate_data\.py|\.ipynb_checkpoints"
flake8 . --count --ignore="E501 W503 E203 F841 E402" --show-source --statistics --exclude=./.*,build,dist,official*,artifact_pipeline,helper_code.py,run_model.py,train_model.py,evaluate_model.py,remove_data.py,remove_labels.py,truncate_data.py,*.ipynb
