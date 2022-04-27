#!/bin/bash
set -x
# Thanks to Yaroslav Bulatov for the implementaion of this script.
# https://github.com/cybertronai/aws-network-benchmarks
# Note: This script is tested on Alinux2 runnin on GPU instance with Tesla volta arch.

# Remove older versions of dcgm
sudo yum remove datacenter-gpu-manager -y

# Upaate packages
sudo yum update -y
sudo yum groupinstall "Development Tools" -y
sudo yum install wget kernel-devel-$(uname -r) kernel-headers-$(uname -r) -y
# sudo yum install gcc10 kernel-devel kernel-headers -y

# Fix Polkit Privilege Escalation Vulnerability
chmod 0755 /usr/bin/pkexec

export INSTALL_ROOT=${HOME}

mkdir -p "$INSTALL_ROOT"/packages
cd "$INSTALL_ROOT"/packages || exit

export EFA_INSTALLER_FN=aws-efa-installer-latest.tar.gz
echo "Installing EFA " $EFA_INSTALLER_FN

wget https://s3-us-west-2.amazonaws.com/aws-efa-installer/$EFA_INSTALLER_FN
tar -xf $EFA_INSTALLER_FN
cd aws-efa-installer || exit
sudo ./efa_installer.sh -y

# echo "Installing CUDA"
cd "$INSTALL_ROOT"/packages || exit
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
chmod +x cuda_11.3.0_465.19.01_linux.run
sudo ./cuda_11.3.0_465.19.01_linux.run --silent --override --toolkit --samples --no-opengl-libs

export PATH="/usr/local/cuda/bin:/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

echo 'Building nccl'
cd "$INSTALL_ROOT"/packages || exit
git clone https://github.com/NVIDIA/nccl.git || echo ignored
cd nccl || exit
git checkout tags/v2.11.4-1 -b v2.11.4-1
# Choose compute capability 70 for Tesla V100 and 80 for Tesla A100
# Refer https://en.wikipedia.org/wiki/CUDA#Supported_GPUs for different architecture
make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80"
make pkg.txz.build
cd build/pkg/txz || exit

tar xvfJ nccl_2.11.4-1+cuda11.3_x86_64.txz
sudo cp -r nccl_2.11.4-1+cuda11.3_x86_64/include/* /usr/local/cuda/include/
sudo cp -r nccl_2.11.4-1+cuda11.3_x86_64/lib/* /usr/local/cuda/lib64/

echo 'Building aws-ofi-nccl'
cd "$INSTALL_ROOT"/packages || exit
git clone https://github.com/aws/aws-ofi-nccl.git || echo exists
cd aws-ofi-nccl || exit
git checkout aws
git pull
./autogen.sh

./configure --prefix=/usr --with-mpi=/opt/amazon/openmpi --with-libfabric=/opt/amazon/efa/ --with-cuda=/usr/local/cuda --with-nccl=$INSTALL_ROOT/packages/nccl/build

sudo yum install libudev-devel -y
PATH=/opt/amazon/efa/bin:$PATH LDFLAGS="-L/opt/amazon/efa/lib64" make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=$INSTALL_ROOT/packages/nccl/build
sudo make install

sudo sh -c 'echo "/opt/amazon/openmpi/lib64/" > mpi.conf'
sudo sh -c 'echo "$INSTALL_ROOT/packages/nccl/build/lib/" > nccl.conf'
sudo sh -c 'echo "/usr/local/cuda/lib64/" > cuda.conf'
sudo ldconfig

cd /usr/local/lib || exit
sudo rm -f ./libmpi.so
sudo ln -s /opt/amazon/openmpi/lib64/libmpi.so ./libmpi.s


echo 'installing NCCL'
cd "$INSTALL_ROOT"/packages || exit
git clone https://github.com/NVIDIA/nccl-tests.git || echo ignored
cd nccl-tests || exit
make MPI=1 MPI_HOME=/opt/amazon/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME="$INSTALL_ROOT"/packages/nccl/build

# Set Environment variables
export CUDA_HOME=/usr/local/cuda
export EFA_HOME=/opt/amazon/efa
export MPI_HOME=/opt/amazon/openmpi
export FI_PROVIDER="efa"
export NCCL_DEBUG=INFO
export FI_EFA_USE_DEVICE_RDMA=1  # Use for p4dn
export NCCL_ALGO=ring

echo "================================"
echo "===========Check EFA============"
echo "================================"
fi_info -t FI_EP_RDM -p efa

echo "================================"
echo "====Testing all_reduce_perf====="
echo "================================"
# test all_reduce_perf
bin=$INSTALL_ROOT/packages/nccl-tests/build/all_reduce_perf
LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$EFA_HOME/lib64:$MPI_HOME/lib64:$INSTALL_ROOT/packages/nccl/build/lib $bin -b 8 -e 128M -f 2 -g 8

# TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
# curl -H "X-aws-ec2-metadata-token: $TOKEN" -v http://169.254.169.254/latest/meta-data/local-ipv4 >> my-hosts

# /opt/amazon/openmpi/bin/mpirun \
#     -x FI_PROVIDER="efa" \
#     -x FI_EFA_USE_DEVICE_RDMA=1 \
#     -x LD_LIBRARY_PATH=$INSTALL_ROOT/packages/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:$INSTALL_ROOT/packages/aws-ofi-nccl/lib:$LD_LIBRARY_PATH \
#     -x NCCL_DEBUG=INFO \
#     -x NCCL_ALGO=ring \
#     -x NCCL_PROTO=simple \
#     --hostfile my-hosts -n 8 -N 8 \
#     --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
#     $INSTALL_ROOT/packages/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -c 1 -n 100

# Install Fabric Manager
nvidia_info=$(find /usr/lib/modules -name nvidia.ko)
nvidia_version=$(modinfo "$nvidia_info" | grep ^version | awk '{print $2}')
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
sudo yum clean all
# sudo wget -O /tmp/NVIDIA-Linux-driver.run https://us.download.nvidia.com/tesla/${nvidia_version}/NVIDIA-Linux-x86_64-${nvidia_version}.run
# sudo CC=gcc10-cc sh /tmp/NVIDIA-Linux-driver.run -q -a --ui=none
sudo curl -O https://developer.download.nvidia.com/compute/nvidia-driver/redist/fabricmanager/linux-x86_64/fabricmanager-linux-x86_64-${nvidia_version}-archive.tar.xz
sudo tar xf fabricmanager-linux-x86_64-"${nvidia_version}"-archive.tar.xz -C /tmp
sudo rsync -al /tmp/fabricmanager-linux-x86_64-"${nvidia_version}"-archive/ /usr/ --exclude LICENSE
sudo mv /usr/systemd/nvidia-fabricmanager.service /usr/lib/systemd/system
sudo systemctl enable nvidia-fabricmanager && sudo systemctl start nvidia-fabricmanager

# Verifying GPU Routing
sudo nvswitch-audit

# Download and Install Nvidia DCGM
cd /lustre || exit
sudo yum install -y datacenter-gpu-manager
# wget -O datacenter-gpu-manager-2.2.6-1-x86_64_debug.rpm https://mlbucket-4d8b827c.s3.amazonaws.com/datacenter-gpu-manager-2.2.6-1-x86_64_debug.rpm
# wget -O datacenter-gpu-manager-2.2.6-1-x86_64_debug.rpm https://fsdp-expeirments.s3.us-west-2.amazonaws.com/datacenter-gpu-manager-2.2.6-1-x86_64_debug.rpm
# sudo rpm -i datacenter-gpu-manager-2.2.6-1-x86_64_debug.rpm

# Start nv-hostengine
sudo -u root nv-hostengine -b 0

# Install EFA Exporter
sudo /usr/bin/python3 -m pip install --upgrade pip
sudo pip3 install boto3
sudo yum install amazon-cloudwatch-agent -y
git clone https://github.com/aws-samples/aws-efa-nccl-baseami-pipeline.git /tmp/aws-efa-nccl-baseami
sudo mv /tmp/aws-efa-nccl-baseami/nvidia-efa-ami_base/cloudwatch /opt/aws/
sudo mv /opt/aws/cloudwatch/aws-hw-monitor.service /lib/systemd/system
echo -e "#!/bin/sh\n" | sudo tee /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh
echo -e "/usr/bin/python3 /opt/aws/cloudwatch/nvidia/aws-hwaccel-error-parser.py &" | sudo tee -a /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh
echo -e "/usr/bin/python3 /opt/aws/cloudwatch/nvidia/accel-to-cw.py /opt/aws/cloudwatch/nvidia/nvidia-exporter >> /dev/null 2>&1 &\n" | sudo tee -a /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh
echo -e "/usr/bin/python3 /opt/aws/cloudwatch/efa/efa-to-cw.py /opt/aws/cloudwatch/efa/efa-exporter >> /dev/null 2>&1 &\n" | sudo tee -a /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh
sudo chmod +x /opt/aws/cloudwatch/aws-cloudwatch-wrapper.sh
sudo cp /opt/aws/cloudwatch/nvidia/cwa-config.json /opt/aws/amazon-cloudwatch-agent/bin/config.json
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/bin/config.json -s
sudo systemctl enable aws-hw-monitor.service
sudo systemctl start aws-hw-monitor.service
sudo systemctl restart amazon-cloudwatch-agent.service

#Load AWS Parallelcluster environment variables
. /etc/parallelcluster/cfnconfig

#get GitHub repo to clone and the installation script
monitoring_url=${cfn_postinstall_args[0]}
monitoring_dir_name=${cfn_postinstall_args[1]}
monitoring_tarball="${monitoring_dir_name}.tar.gz"
setup_command=${cfn_postinstall_args[2]}
monitoring_home="/home/${cfn_cluster_user}/${monitoring_dir_name}"

case ${cfn_node_type} in
    HeadNode)
        wget ${monitoring_url} -O ${monitoring_tarball}
        mkdir -p ${monitoring_home}
        tar xvf ${monitoring_tarball} -C ${monitoring_home} --strip-components 1
    ;;
    ComputeFleet)
        # export cuda paths
        export PATH="/usr/local/cuda/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    ;;
esac

#Execute the monitoring installation script
bash -x "${monitoring_home}/parallelcluster-setup/${setup_command}" >/tmp/monitoring-setup.log 2>&1

source /lustre/.conda/etc/profile.d/conda.sh
conda activate

cat >> ~/.bashrc << EOF
export PATH=/usr/local/cuda/bin:/lustre/.conda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
export CUDNN_INCLUDE_DIR="/usr/local/cuda/include"
export CUDNN_LIB_DIR="/usr/local/cuda/lib64"
export OMP_NUM_THREADS=1
export EFA_HOME=/opt/amazon/efa
export MPI_HOME=/opt/amazon/openmpi
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc
EOF
