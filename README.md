# Extended 3D Diffusion Policy with Diffusion Policy

## 📊 Implementation Scope

This implementation focuses on:
1. Implementing DP approach based on DP3 framework
2. Benchmarking both approaches on the MetaWorld reach-wall task

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/YanjieZe/3D-Diffusion-Policy.git
cd 3D-Diffusion-Policy

# Create and activate conda environment
mamba create -n dp3 python=3.8
conda activate dp3

# Install PyTorch (based on CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install DP3
cd 3D-Diffusion-Policy && pip install -e . && cd ..

# Install MuJoCo
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate
tar -xvzf mujoco210.tar.gz

# Set environment variables (add to ~/.bashrc)
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' >> ~/.bashrc
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
source ~/.bashrc

# Install MuJoCo-py (ensure you're in the dp3 environment)
conda activate dp3
cd 3D-Diffusion-Policy/third_party
cd mujoco-py-2.1.2.14 && pip install -e . && cd ../..

# Install MetaWorld environment
# Install build tools
pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

# Install Gym 0.21.0
cd third_party
cd gym-0.21.0 && pip install -e . && cd ..

# Install MetaWorld
cd Metaworld && pip install -e . && cd ..

# Install PyTorch3D (simplified version)
cd pytorch3d_simplified && pip install -e . && cd ..

# Install required packages
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 numba==0.56.4 moviepy imageio av matplotlib termcolor natsort open3d

# Fix version conflicts
pip uninstall diffusers huggingface_hub -y
pip install diffusers==0.20.0 huggingface_hub==0.14.1
```

## 📦 Dataset

Download the pre-collected expert demonstrations for MetaWorld reach-wall task:

[![Download](https://img.shields.io/badge/Download-OneDrive-blue)](https://1drv.ms/u/s!AqMe_P3KHrgna-XR5n3pkFVI5-o?e=CBH2EK)

Data Preparation:
```bash
# Create data directory
cd 3D-Diffusion-Policy
mkdir -p data

# Extract dataset
unzip /path/to/data/metaworld_reach-wall_expert.zarr.zip -d data/
rm /path/to/data/metaworld_reach-wall_expert.zarr.zip
```

## 🛠️ Usage

1. **Training a DP3 policy on MetaWorld reach-wall task**:
   ```bash
   conda activate dp3
   wandb login
   bash scripts/train_policy.sh dp3 metaworld_reach-wall 0317 0 0
   ```

2. **Training a DP policy on MetaWorld reach-wall task**:
   ```bash
   conda activate dp3
   wandb login
   bash scripts/train_policy.sh dp metaworld_reach-wall 0317 0 0
   ```

## 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## 😺 Acknowledgement
This implementation integrates and extends the original [3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) and [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) repositories. The code also builds upon [MetaWorld](https://github.com/Farama-Foundation/Metaworld) for the environment.

## 📝 Citation

If you find this work useful, please consider citing the original papers:

```
@inproceedings{Ze2024DP3,
	title={3D Diffusion Policy: Generalizable Visuomotor Policy Learning via Simple 3D Representations},
	author={Yanjie Ze and Gu Zhang and Kangning Zhang and Chenyuan Hu and Muhan Wang and Huazhe Xu},
	booktitle={Proceedings of Robotics: Science and Systems (RSS)},
	year={2024}
}

@inproceedings{chi2023diffusion,
    title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
    author={Chi, Cheng and Vyetrenko, Sashank and Ichter, Brian and Allshire, Andrew and Goodrich, Ben and Yao, Hao and Xia, Yunzhu and Kagaya, Montse and Ho, Jonathan and Chebotar, Yevgen and Black, Alexander W and Hausman, Karol and Jang, Eric and Zhao, Dorsa and Majumdar, Anirudha},
    booktitle={Proceedings of Robotics: Science and Systems (RSS)},
    year={2023}
}
```
