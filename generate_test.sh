#!/bin/bash

fileName=test_todo.sh

epoch=50
nb_batch=1000


for dataset in mnist fashion-mnist cifar10; do
    echo python main.py --dataset $dataset --gan_type Classifier --epoch $epoch --batch_size 64 --device 0 --train_G True --nb_batch $nb_batch >> $fileName
done

echo >> $fileName

for gan_type in VAE WGAN ACGAN CVAE; do
    for dataset in mnist fashion-mnist cifar10; do

        echo python main.py --dataset $dataset --gan_type $gan_type --epoch $epoch --batch_size 64 --device 0 --train_G True --nb_batch $nb_batch >> $fileName
        for tau in `LANG=en_US seq 0.125 0.125 1`; do
            echo python main.py --dataset $dataset --gan_type $gan_type --epoch $epoch --batch_size 64 --device 0 --classify True --tau $tau  >> $fileName
        done

        echo >> $fileName

    done
done