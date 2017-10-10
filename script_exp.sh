
#GAN
python main.py --dataset mnist --gan_type GAN --epoch 25 --batch_size 64 --device 0
python main.py --dataset mnist --gan_type GAN --epoch 25 --batch_size 64 --device 0 --conditional True
python main.py --dataset fashion-mnist --gan_type GAN --epoch 25 --batch_size 64 --device 0
python main.py --dataset fashion-mnist --gan_type GAN --epoch 25 --batch_size 64 --device 0 --conditional True
python main.py --dataset cifar10 --gan_type GAN --epoch 25 --batch_size 64 --device 0
python main.py --dataset cifar10 --gan_type GAN --epoch 25 --batch_size 64 --device 0 --conditional True


#VAE
python main.py --dataset mnist --gan_type VAE --epoch 25 --batch_size 64 --device 0
python main.py --dataset mnist --gan_type VAE --epoch 25 --batch_size 64 --device 0 --conditional True
python main.py --dataset fashion-mnist --gan_type VAE --epoch 25 --batch_size 64 --device 0
python main.py --dataset fashion-mnist --gan_type VAE --epoch 25 --batch_size 64 --device 0 --conditional True
python main.py --dataset cifar10 --gan_type VAE --epoch 25 --batch_size 64 --device 0
python main.py --dataset cifar10 --gan_type VAE --epoch 25 --batch_size 64 --device 0 --conditional True