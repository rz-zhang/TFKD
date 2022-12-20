CUDA_VISIBLE_DEVICES=2 python main.py --model_dir experiments/kd_experiments/mobilenet_distill/mobilenet_self_teacher/ --self_training --log_path tfkd_self.log
CUDA_VISIBLE_DEVICES=2 python main.py --model_dir experiments/kd_experiments/mobilenet_distill/mobilenet_self_teacher/ --perturb --log_path ptloss01.log

CUDA_VISIBLE_DEVICES=1 python main.py --model_dir experiments/kd_experiments/shufflenet_distill/shufflenet_self_teacher/ --self_training
CUDA_VISIBLE_DEVICES=2 python main.py --model_dir experiments/kd_experiments/resnet18_distill/resnet18_self_teacher/ --self_training
CUDA_VISIBLE_DEVICES=2 python main.py --model_dir experiments/kd_experiments/resnet18_distill/resnet18_self_teacher/ --perturb --log_path ptloss05.log
CUDA_VISIBLE_DEVICES=0 python main.py --model_dir experiments/kd_experiments/resnext29_distill/resnext29_self_teacher/ --self_training --log_path bigserver_tfkd.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --model_dir experiments/kd_experiments/resnext29_distill/resnext29_self_teacher/ --self_training

CUDA_VISIBLE_DEVICES=1 python main.py --model_dir experiments/kd_experiments/googlenet_distill/googlenet_self_teacher/ --self_training
CUDA_VISIBLE_DEVICES=6 python main.py --model_dir experiments/kd_experiments/googlenet_distill/googlenet_self_teacher/ --perturb --log_path ptloss01.log

CUDA_VISIBLE_DEVICES=0 python main.py --model_dir experiments/kd_experiments/densenet121_distill/densenet_self_teacher/ --self_training
CUDA_VISIBLE_DEVICES=7 python main.py --model_dir experiments/kd_experiments/densenet121_distill/densenet_self_teacher/ --perturb --log_path ptloss01.log


