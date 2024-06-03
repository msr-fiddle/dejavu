FROM nvcr.io/nvidia/pytorch:22.10-py3
WORKDIR /workspace
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y libasio-dev libboost-dev
RUN apt-get install -y libprotobuf-dev protobuf-compiler libgrpc-dev protobuf-compiler-grpc libgrpc++-dev
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs -y


COPY . /workspace/dejavu
CMD mkdir -p /workspace/dejavu/build
WORKDIR /workspace/dejavu/build
RUN protoc -I ../src/fastertransformer/models/multi_gpu_gpt/ --cpp_out=../src/fastertransformer/models/multi_gpu_gpt/ ../src/fastertransformer/models/multi_gpu_gpt/state_stream.proto
RUN protoc -I ../src/fastertransformer/models/multi_gpu_gpt/ --grpc_out=../src/fastertransformer/models/multi_gpu_gpt/  --plugin=protoc-gen-grpc=`which grpc_cpp_plugin`  ../src/fastertransformer/models/multi_gpu_gpt/state_stream.proto

RUN protoc -I ../src/fastertransformer/models/multi_gpu_gpt/ --grpc_out=../src/fastertransformer/models/multi_gpu_gpt/  --plugin=protoc-gen-grpc=`which grpc_cpp_plugin`  ../src/fastertransformer/models/multi_gpu_gpt/ft_state.proto
RUN protoc -I ../src/fastertransformer/models/multi_gpu_gpt/ --cpp_out=../src/fastertransformer/models/multi_gpu_gpt/ ../src/fastertransformer/models/multi_gpu_gpt/ft_state.proto

RUN cp -r /opt/hpcx/ompi/lib/* /usr/lib/
RUN cp -r /opt/hpcx/ompi/include/* /usr/include/

RUN cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DBUILD_MICROBENCHMARKS=ON ..
RUN make -j12

RUN pip install -r ../examples/pytorch/gpt/requirement.txt
RUN python -m grpc_tools.protoc --proto_path=../examples/pytorch/gpt/api/ --python_out=../examples/pytorch/gpt/api/ --grpc_python_out=../examples/pytorch/gpt/api/ ../examples/pytorch/gpt/api/protos/api_server.proto

WORKDIR /workspace
CMD ["/bin/bash"]
