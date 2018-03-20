#!/bin/bash

echo """
fileName=test_classifier.sh

epoch=200

##################################  generate reference classifier #######################################

for seed in `seq 8`; do
	for num_example in  50000; do # 50 100 500 1000 5000 10000
		for dataset in  mnist fashion-mnist; do #mnist
			echo python main.py --dataset $dataset --gan_type Classifier --epoch $epoch --device 0 --num_examples $num_example --seed $seed --dir /slowdata/tim_bak/Generative_Model >> $fileName
			#for tresh in `LANG=en_US seq 0.0 0.125 1`; do
		    	#	echo python main.py --dataset $dataset --gan_type Classifier --epoch $epoch --batch_size 64 --device 0 --num_examples $num_example --tresh_masking_noise $tresh >> $fileName
			#done

			#for sigma in `LANG=en_US seq 0.0 0.125 1`; do
		    	#	echo python main.py --dataset $dataset --gan_type Classifier --epoch $epoch --batch_size 64 --device 0 --num_examples $num_example --sigma $sigma >> $fileName
			#done
		done
	done
done
"""

fileName=test_todo.sh
epoch_generator=100
epoch_classif=200
##################################  generate classifier to train with generator #######################################

for seed in `seq 8`; do
for gan_type in WGAN ;do #VAE CVAE CGAN GAN BEGAN ;do #VAE WGAN CVAE CGAN GAN ACGAN CVAE BEGAN; do
    for dataset in  mnist fashion-mnist;do #mnist fashion-mnist cifar10 timagenet; do
	for num_example in 50000; do	# 50  100 500 1000 5000 10000
		#echo python main.py --dataset $dataset --gan_type $gan_type --epoch $epoch_generator --device 0 --train_G True --num_examples $num_example --seed $seed --dir /slowdata/tim_bak/Generative_Model >> $fileName
        echo python main.py --dataset $dataset --gan_type $gan_type --epoch $epoch_classif --device 0 --IS True --tau 0 --num_examples $num_example --seed $seed --dir /slowdata/tim_bak/Generative_Model >> $fileName
		for tau in `LANG=en_US seq 0.125 0.125 1`; do
            echo 'plop'
		    #echo python main.py --dataset $dataset --gan_type $gan_type --epoch $epoch_classif --device 0 --classify True --tau $tau --num_examples $num_example --seed $seed --dir /slowdata/tim_bak/Generative_Model >> $fileName
            #echo python main.py --dataset $dataset --gan_type $gan_type --epoch $epoch_classif --device 0 --knn True --tau $tau --num_examples $num_example --seed $seed --dir /slowdata/tim_bak/Generative_Model >> $fileName
		done
	done
    done
       echo >> $fileName
done
done


echo "

fileName=test_DA_classifier.sh

##################################  generate classifier to train with data augmentation #######################################

for seed in `seq 8`; do
	for num_example in 100 500 1000 5000 10000 50000; do # 50
		for dataset in mnist fashion-mnist; do
			for tresh in `LANG=en_US seq 0.0 0.125 1`; do
		    		echo python main.py --dataset $dataset --gan_type Classifier --epoch $epoch --batch_size 64 --device 0 --num_examples $num_example --tresh_masking_noise $tresh --seed $seed >> $fileName
			done

			for sigma in `LANG=en_US seq 0.0 0.125 1`; do
		    		echo python main.py --dataset $dataset --gan_type Classifier --epoch $epoch --batch_size 64 --device 0 --num_examples $num_example --sigma $sigma  --seed $seed >> $fileName
			done
		done
	done
done
"


