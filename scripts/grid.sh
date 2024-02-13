#!/bin/zsh

lrs=(-4 -6 -8)
wds=(0 1e-3 1e-4 1e-5)

for lr in $lrs; do
    for wd in $wds; do
        fname=grid/lr1e$lr\_wd$wd
        echo $fname
        cp scripts/grid/template.sh scripts/$fname.sh
        echo "--output_dir /mnt/data/sonia/ckpts/$fname \\" >> scripts/$fname.sh
        echo "    --learning_rate 1e$lr \\" >> scripts/$fname.sh
        echo "    --weight_decay $wd \\" >> scripts/$fname.sh
        chmod +x scripts/$fname.sh

        ./scripts/$fname.sh
    done
done
