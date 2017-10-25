#0 ok
python main.py --dataset mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 50
python main.py --dataset fashion-mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 50
python main.py --dataset mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 100
#5 ok
python main.py --dataset fashion-mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 100
#7 ok
python main.py --dataset mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 500
#1 ok
python main.py --dataset fashion-mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 500
python main.py --dataset mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 1000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 1000
python main.py --dataset mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 5000
#6 ok
python main.py --dataset fashion-mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 5000
#2 ok
python main.py --dataset mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 10000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 10000
python main.py --dataset mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 60000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 60000
#3 ok
python main.py --dataset mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 50
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 50
python main.py --dataset mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 100
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 100
#4 ok
python main.py --dataset mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 500
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 500
python main.py --dataset mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 1000
#5 ok
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 1000
python main.py --dataset mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 5000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 5000
#6 ok
python main.py --dataset mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 10000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 10000
#7 ok
python main.py --dataset mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 60000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 50 --batch_size 64 --device 0 --train_G True --num_examples 60000

#1 ok - 500 #1 - #1 restart done
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 50
python main.py --dataset mnist --gan_type VAE --epoch 50 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 50
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 50
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 50
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 50
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 50
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 50
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 50
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 50
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 100
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 100
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 100
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 100
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 100
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 100
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 100
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 100
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 100
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 500
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 500
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 500
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 500
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 500
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 500
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 500
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 500
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 500
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 1000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 1000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 1000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 1000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 1000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 1000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 1000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 1000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 1000

#7 ab. - 500 #7 - restart 6 bis
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 5000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 5000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 5000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 5000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 5000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 5000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 5000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 5000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 5000

# restart 4 bis
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 10000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 10000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 10000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 10000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 10000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 10000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 10000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 10000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 10000

# restart 1 bis done
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 60000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 60000
# restart 5 bis
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 60000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 60000
# restart 2 
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 60000
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 60000
# restart 7 bis
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 60000
# restart 1 (7bis too...)
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 60000
# restart 3 bis
python main.py --dataset mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 60000


#6 - 500 - #6 restart done
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 50
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 50
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 50
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 50
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 50
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 50
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 50
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 50
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 50
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 100
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 100
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 100
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 100
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 100
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 100
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 100
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 100
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 100
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 500
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 500
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 500
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 500
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 500
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 500
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 500
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 500
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 500
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 1000

#3 ab - 500 #3 - #5 restart
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type VAE --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 60000
#4 ok - 500 4# - #4 restart done
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 50
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 50
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 50
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 50
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 50
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 50
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 50
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 50
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 50
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 100
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 100
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 100
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 100
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 100
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 100
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 100
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 100
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 100
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 500
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 500
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 500
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 500
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 500
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 500
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 500
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 500
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 500
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 1000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 1000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 1000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 1000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 1000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 1000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 1000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 1000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 1000
#2 ab - 500 #2 - #3 restart
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 5000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 5000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 5000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 5000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 5000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 5000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 5000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 5000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 5000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 10000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 10000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 10000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 10000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 10000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 10000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 10000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 10000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 10000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 60000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 60000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 60000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 60000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 60000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 60000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 60000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 60000
python main.py --dataset mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 60000
#1 ok #5 500 - #2 restart
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 50
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 50
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 50
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 50
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 50
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 50
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 50
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 50
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 50
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 100
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 100
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 100
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 100
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 100
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 100
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 100
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 100
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 100
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 500
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 500
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 500
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 500
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 500
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 500
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 500
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 500
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 500
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 1000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 1000
#0 ab - 500 #0 - #0 restart 
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 5000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 10000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.000 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.125 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.250 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.375 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.500 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.625 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.750 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 0.875 --num_examples 60000
python main.py --dataset fashion-mnist --gan_type WGAN --epoch 500 --batch_size 64 --device 0 --classify True --tau 1.000 --num_examples 60000

