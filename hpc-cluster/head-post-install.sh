#!/bin/bash

# configuring the conda environment
CONDA_DIRECTORY=/lustre/.conda/bin

if [ ! -d "$CONDA_DIRECTORY" ]; then
  # control will enter here if $CONDA_DIRECTORY doesn't exist.
  echo "Conda installation not found. Installing..."
  wget -O miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && bash miniconda.sh -b -p /lustre/.conda && /lustre/.conda/bin/conda init bash && eval "$(/lustre/.conda/bin/conda shell.bash hook)" && rm -rf miniconda.sh

  conda install python=3.8 -y
fi

chown -R ec2-user:ec2-user /lustre

sudo -u ec2-user /lustre/.conda/bin/conda init bash

# Override run_instance attributes for capacity reservation
# cat > /opt/slurm/etc/pcluster/run_instances_overrides.json << EOF
# {
#     "queue0-p4d24xlarge": {
#         "queue0-p4d24xlarge": {
#             "CapacityReservationSpecification": {
#                 "CapacityReservationTarget": {
#                     "CapacityReservationResourceGroupArn": "arn:aws:resource-groups:ap-northeast-2:<ACCOUNT_ID>:group/EC2CRGroup"
#                 }
#             }
#         }
#     }
# }
# EOF

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
exit $?