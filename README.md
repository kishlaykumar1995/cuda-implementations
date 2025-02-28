# CUDA Implementations
**Practicing cuda implementations of various models and algorithms**

## Setting up CUDA on local machine
### CUDA Toolkit install commands
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```
### Driver Installation (Open Kernel)
`sudo apt-get install -y nvidia-open`

Then add the following lines to your .bashrc file (depending on the CUDA version):
```
export PATH="/usr/local/cuda-12/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH"
```

**NOTE: The `nvidia-smi` command shows the latest version of CUDA supported by the driver. The actual CUDA toolkit version installed can be seen using `nvcc --version`.**

See [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux) for more details