# Examples:
# bash scripts/train_policy.sh dp3 metaworld_reach-wall 0317 0 0
# bash scripts/train_policy.sh dp metaworld_reach-wall 0317 0 0


DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


# if [ $DEBUG = True ]; then
#     wandb_mode=disabled
#     # wandb_mode=online
#     echo -e "\033[33mDebug mode!\033[0m"
#     echo -e "\033[33mDebug mode!\033[0m"
#     echo -e "\033[33mDebug mode!\033[0m"
# else
#     wandb_mode=online
#     echo -e "\033[33mTrain mode\033[0m"
# fi

if [ "$alg_name" = "dp3" ]; then
    train_script="train_dp3.py"
else
    train_script="train_dp.py"
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python ${train_script} --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} 



                                