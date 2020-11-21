#!/bin/bash

for model in resnet20
do
    echo "python -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model"
    python -u trainer.py  --arch=$model --epochs=1 --resume=pretrained_models/resnet20-12fca82f.th --save-dir=save_$model |& tee -a log_$model
done