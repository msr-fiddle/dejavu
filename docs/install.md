### Run with containers

Use the [Dockerfile](Dockerfile) to build the image

### Run without containers

We expect NVIDIA drivers to be installed. We also expect CMake >= 3.19.
The current guide was tested on Ubuntu 20.04.

1.  Install Python-3.8

```bash
    sudo apt-get update -y
    sudo apt install software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt install python3.8-dev -y

    echo "alias python=python3.8" >> $HOME/.bashrc
    echo "alias python3=python3.8" >> $HOME/.bashrc
    source $HOME/.bashrc
```

2. Install CUDA (CUDA-11.8 in this doc, but that's configurable)

```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-11.8

    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib" >> $HOME/.bashrc
    echo "export PATH=$PATH:/usr/local/cuda-11.8/bin" >> $HOME/.bashrc
    source $HOME/.bashrc
```

3. Install CUDNN (8.6.0 here)

Download [CUDNN](https://developer.nvidia.com/rdp/cudnn-archive)
```bash

    tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
    sudo cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/cudnn*.h /usr/local/cuda-11.8/include
    sudo cp -P cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64
    sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
```

4. Install OpenMPI

```bash

    wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.0.tar.bz2
    tar -jxf openmpi-5.0.0.tar.bz2
    cd openmpi-5.0.0
    ./configure --prefix=$HOME/opt/openmpi
    make all
    make install

    echo "export PATH=\$PATH:\$HOME/opt/openmpi/bin" >> $HOME/.bashrc
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$HOME/opt/openmpi/lib" >> $HOME/.bashrc
    source $HOME/.bashrc

```

5. Install NCCL (2.15.5 in this doc)

```bash

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update

    sudo apt install libnccl2=2.15.5-1+cuda11.8 libnccl-dev=2.15.5-1+cuda11.8

```

6. Install PyTorch (1.13.0 in this doc - from source)

```bash

    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    git reset --hard 7c98e70d44abc7a1aead68b6ea6c8adc8c554db5
    git submodule sync
    git submodule update --init --recursive --jobs 0
    python setup.py develop

```

7. Install necessary libraries

```bash

    sudo apt-get install -y libasio-dev libboost-dev
    sudo apt-get install -y libprotobuf-dev protobuf-compiler libgrpc-dev protobuf-compiler-grpc libgrpc++-dev
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs -y

```

8. Remove PyTorch-based protobuf installation (since it messes up with DéjàVu compilation):

```bash

rm $HOME/pytorch/build/lib/libprotobuf.a
rm -rf $HOME/pytorch/torch/include/google/

```


9. Install DéjàVu (Replace 'xx' at the cmake command with the compute capability of your GPU).
We provide a set of compile flags for different use cases.
Please look at [dejavulib.md](dejavulib.md), [dejavu.md](dejavu.md) for more details.


```bash

git clone https://github.com/msr-fiddle/dejavu.git
cd dejavu
mkdir build
cd build
protoc -I ../src/fastertransformer/models/multi_gpu_gpt/ --cpp_out=../src/fastertransformer/models/multi_gpu_gpt/ ../src/fastertransformer/models/multi_gpu_gpt/state_stream.proto
protoc -I ../src/fastertransformer/models/multi_gpu_gpt/ --grpc_out=../src/fastertransformer/models/multi_gpu_gpt/  --plugin=protoc-gen-grpc=`which grpc_cpp_plugin`  ../src/fastertransformer/models/multi_gpu_gpt/state_stream.proto

protoc -I ../src/fastertransformer/models/multi_gpu_gpt/ --grpc_out=../src/fastertransformer/models/multi_gpu_gpt/  --plugin=protoc-gen-grpc=`which grpc_cpp_plugin`  ../src/fastertransformer/models/multi_gpu_gpt/ft_state.proto
protoc -I ../src/fastertransformer/models/multi_gpu_gpt/ --cpp_out=../src/fastertransformer/models/multi_gpu_gpt/ ../src/fastertransformer/models/multi_gpu_gpt/ft_state.proto

pip install -r ../examples/pytorch/gpt/requirement.txt --user
python -m grpc_tools.protoc --proto_path=../examples/pytorch/gpt/api/ --python_out=../examples/pytorch/gpt/api/ --grpc_python_out=../examples/pytorch/gpt/api/ ../examples/pytorch/gpt/api/protos/api_server.proto

cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DBUILD_MICROBENCHMARKS=ON ..
make -j12

wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P ../models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P ../models

```


### Run a simple example

Download and convert the model as needed, following the instructions [here](https://github.com/msr-fiddle/dejavu/blob/master/docs/original_ft/gpt_guide.md#download-huggingface-gpt-model-and-convert). Note that for fp16 support you need to add the `` and `` flags when calling the *huggingface_gpt_convert.py* script. Make sure to call the *huggingface_opt_convert.py* and *huggingface_bloom_convert.py* scripts for the OPT and BLOOM models respectively.

We assume we are at the path 'dejavu/build'. The following command will run a single-GPU example:

```bash

python ../examples/pytorch/gpt/gpt_batch_maker.py  --tensor_para_size=1 --pipeline_para_size=1 --ckpt_path <path_to_model> --ubatch_size 1

```
