#!/bin/bash

for model in resnet20
do
    echo "python -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model"
    python -u trainer.py  --arch=$model --epochs=6 --save-dir=save_$model |& tee -a log_$model
done