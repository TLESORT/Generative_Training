#!/bin/bash


fileName=test_todo.sh
epoch_generator=25
epoch_classif=200
num_example=50000


# most important results
for seed in `seq 8`; do
fileName=test_todo$seed.sh ################# FILENAME ##########################################
echo '#!/bin/bash' >> $fileName
echo 'cd ..' >> $fileName
    for dataset in  mnist fashion-mnist;do

        # train the classifier for tau=0
	    echo python main.py --dataset $dataset --gan_type Classifier --epoch $epoch_classif --device 0 --tau 0 --num_examples $num_example --seed $seed --dir /slowdata/tim_bak/Generative_Model >> $fileName

	    for gan_type in WGAN VAE CVAE CGAN GAN BEGAN ;do #VAE WGAN CVAE CGAN GAN ACGAN BEGAN; do
	        for num_example in 50000; do	# 50  100 500 1000 5000 10000
                # train generator, train classifier for tau=1, also train knn, compute also IS and FID
                echo python main.py --dataset $dataset --gan_type $gan_type --epoch $epoch_classif --device 0 --train_G True --tau 1  --classify True --knn True --IS True --FID True --TrainEval True --num_examples $num_example --seed $seed --dir /slowdata/tim_bak/Generative_Model >> $fileName

	        done
        done
       echo >> $fileName
    done
done

# side results
for seed in `seq 8`; do
fileName=test_todo$seed.sh ################# FILENAME ##########################################
    for gan_type in WGAN VAE CVAE CGAN GAN BEGAN ;do
      for dataset in  mnist fashion-mnist;do #mnist fashion-mnist cifar10 timagenet; do
	    for num_example in 50000; do
		    for tau in `LANG=en_US seq 0.125 0.125 0.875`; do
		        echo python main.py --dataset $dataset --gan_type $gan_type --epoch $epoch_classif --device 0 --classify True --tau $tau --knn True --IS True --FID True --TrainEval True --num_examples $num_example --seed $seed --dir /slowdata/tim_bak/Generative_Model >> $fileName
		    done
	    done
        done
           echo >> $fileName
    done
done
