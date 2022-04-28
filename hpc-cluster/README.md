# AWS HPC Cluster Setup

## Prerequisites

* Start an EC2 Instance and establish and ssh an session
* Configure aws cli

  ```bash
  aws configure
  ```

* Install aws parallel cluster cli

  ```bash
  python3 -m pip install --upgrade "aws-parallelcluster"
  ```
  
* make sure node.js is installed
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
chmod ug+x ~/.nvm/nvm.sh
source ~/.nvm/nvm.sh
nvm install --lts
node --version
```

## Create s3 bucket

```bash
export BUCKET_POSTFIX=$(uuidgen --random | cut -d'-' -f1)
echo "Your bucket name will be mlbucket-${BUCKET_POSTFIX}"
aws s3 mb s3://mlbucket-${BUCKET_POSTFIX} --region us-west-2
```

Output:

```bash
make_bucket: s3://mlbucket-057bf1b1
```

## Upload post-install script

```bash
aws s3 cp head-post-install.sh s3://mlbucket-${BUCKET_POSTFIX}
upload: ./post-install.sh to s3://mlbucket-057bf1b1/head-post-install.sh

aws s3 cp compute-post-install.sh s3://mlbucket-${BUCKET_POSTFIX}
upload: ./post-install.sh to s3://mlbucket-057bf1b1/compute-post-install.sh
```

# Create VPC

```bash
aws cloudformation create-stack --stack-name VPC-Large-Scale --template-body file://VPC-Large-Scale.yml
```
# Create VPC from Console 

#### Deploy a VPC to run your workload

If using a new account, your VPC configuration will consist of one public subnet in each AZ you'll use in the target region. The p4d.24xlarge instances come with 4 network interfaces and need to be placed into a private subnet otherwise instances will not be able to communicate over the network (see [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-instance-addressing.html#public-ip-addresses)).

Unless you are comfortable deploying a private subnet, set the routes and security groups, we recommend that you deploy a custom VPC using the CloudFormation template called `VPC-Large-Scale`. This template is region agnostic and enables you to create a VPC with the required network architecture to run your workloads.

Please follow the steps below to deploy your new VPC:

1. Click on [this link](https://console.aws.amazon.com/cloudformation/home#/stacks/create/template) to access CloudFormation. Ensure you are in your target region, if not click on the upper left menu to switch to your preferred region.
2. On the Create Stack dialog, in the section *Specify template*, **tick** the box *Upload a template file*. **Click** on *Choose file* then select the file `VPC-Large-Scale.yaml`.
3. You see a list of parameters, do as follows:

  - *Stack Name* is arbitrary, pick `LargeScaleVPC` or any representative name.
  - In *Availability Zones Configuration*, select all the AZs in the region in the *Availability Zones* setting. If using `us-west-1` pick `us-west-1a`, `us-west-1b` and `us-west-1c`.
  - The *Number of Availability Zones* must be equal to the number of AZ you picked in the previous step. If you selected all AZs in `us-west-1` then set the value to `3`.
  - Leave the rest as default

4. **Click** on the *Next* orange button at the bottom of the page and do it again until landing on the page *Step 4: Review*.
5. Scroll down to the bottom of the page. **Tick** the acknowledgement box in the *Capabilities* section and create stack.

It will take a few minutes to deploy your VPC architecture. Once deployed, you need to identify the subnet IDs you'll use to place your instances using your AWS ParallelCluster configuration.

1. Go to your VPC dashboard through this [link](https://console.aws.amazon.com/vpc/home). 
2. In case running into issues with Luster permission/ security groups add it using this dochttps://docs.aws.amazon.com/fsx/latest/LustreGuide/limit-access-security-groups.html
3. **Click** on *Subnets* then filter the subnets using the availability zone ID you'd like to use. For example, filter with `eu-west-1a` to get subnets only for that Availability Zone.
4. You see a list of subnets, you should see one private subnet deployed using the CloudFormation template. Keep note of the subnet ID (similar to `subnet-abc12345defg`).
5. Clear the filter and filter using the string `Public Subnet`. Keep note of the subnet ID.
6. Modify the AWS ParallelConfiguration as indicated below

```yaml
# in the following section, set the subnet value to the Public Subnet ID
HeadNode:
  Networking:
    SubnetId: subnet-abc123456defg
# in the following section, set the subnet value to the Private Subnet ID
Scheduling:
  SlurmQueues:
      Networking:
        SubnetIds:
          - subnet-xyz123456abcd
```

6. Create your cluster with the new configuration file.
## Create key-pair for hpc cluster

```bash
aws ec2 create-key-pair --key-name hpc-key --query KeyMaterial --output text > ~/.ssh/hpc-key
chmod 600 ~/.ssh/hpc-key
```

## Build dcgm

```bash
chmod +x dcgm-build.sh
./dcgm-build.sh
```

Upload the built package from `_out` folder to a s3 bucket and update the url in `compute-post-install.sh` script.

## Edit cluster config yaml

### Modify the cluster.yaml to suit your requirement

### Refer: [Cluster configuration v3](https://docs.aws.amazon.com/parallelcluster/latest/ug/cluster-configuration-file-v3.html)

Note: Add Subnet with Public IP for headnode and Private IP for compute nodes.
## Create HPC cluster

```bash
# Create hpc cluster with No min, max count in cluster.yaml
pcluster create-cluster --cluster-name  my-hpc-cluster --cluster-configuration cluster.yaml
# Need to stop cluster
pcluster update-compute-fleet --cluster-name my-hpc-cluster  --status STOP_REQUESTED
# Now update
pcluster update-cluster --cluster-name my-hpc-cluster  --cluster-configuration cluster.yaml
# Now start 
pcluster update-compute-fleet --cluster-name my-hpc-cluster  --status START_REQUESTED
```

Output

```json
{
  "cluster": {
    "clusterName": "my-hpc-cluster",
    "cloudformationStackStatus": "CREATE_IN_PROGRESS",
    "cloudformationStackArn": "arn:aws:cloudformation:us-west-2:<ACCOUNT_ID>:stack/my-hpc-cluster/dc43a000-640b-11ec-846b-0a803e033d61",
    "region": "us-west-2",
    "version": "3.1.1",
    "clusterStatus": "CREATE_IN_PROGRESS"
  }
}
```
## SSh to headnode

```bash
pcluster ssh --cluster-name cluster -i your-key_pair
cd /lustre
chmod +x compute-post-install.sh head-post-install.sh

git clone https://github.com/lessw2020/t5_11.git

cd t5_11

chmod +x job_*

#install torch nightlies
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113
pip install -r requirements.txt
modify the bert.slurm to https://gist.github.com/HamidShojanazeri/145413925b98506b81541f6a5e86a3d0
 
```

## Create a IAM user account

Create an IAM user account with programmatic credentials and assign the AWS Managed Policy `AmazonEC2ReadOnlyAccess`, `AmazonS3ReadOnlyAccess`, `CloudWatchLogsReadOnlyAccess`, `CloudWatchReadOnlyAccess`

## Modify the prometheus.yaml

1. Update prom-config-example.yaml with region and accesskey, secretkey from above created user account.
2. Ssh into head node
3. Replace the contents of `/home/ec2-user/aws-parallelcluster-monitoring/prometheus` with updated prom-config-example.yaml

## Restart docker compose in the headnode

```bash
docker-compose --env-file /etc/parallelcluster/cfnconfig -f ~/aws-parallelcluster-monitoring/docker-compose/docker-compose.master.yml -p monitoring-master restart
```

## Run the job
(base) [ec2-user@ip-10-0-38-178 t5_11]$ pwd
/lustre/t5_11
```bash
sbatch bert.slurm
```

## For Standalone DCGM Exported

Import the below dashboard into grafana

<https://grafana.com/grafana/dashboards/12239>

## Add Slum Job Log to Grafana

### Download loki and promtail

```bash
wget https://github.com/grafana/loki/releases/download/v2.4.2/loki-linux-amd64.zip
wget https://github.com/grafana/loki/releases/download/v2.4.2/promtail-linux-amd64.zip

unzip loki-linux-amd64.zip
unzip promtail-linux-amd64.zip
```

### Download loki and promtail configs

```bash
wget https://raw.githubusercontent.com/grafana/loki/master/cmd/loki/loki-local-config.yaml
wget https://raw.githubusercontent.com/grafana/loki/main/clients/cmd/promtail/promtail-local-config.yaml
```

### Start loki

```bash
./loki-linux-amd64 --config.file=loki-local-config.yaml &
```

### Add the slum job output file path to promtail-local-config.yaml

```bash
  - targets:
      - localhost
    labels:
      job: slurmlogs
      __path__: /lustre/uber-prof/training-job/*.out
```

### Start promtail

```bash
./promtail-linux-amd64 --config.file=promtail-local-config.yaml &
```

### Add loki datasource to Grafana

![Add datasource](./images/loki_datasource.png)

### Add new dashboard

Add new dashboard with loki data source with logs as visualization panel.

![Add dashboard datasource](./images/dashboard_datasource.png)

![Add dashboard panel](./images/dashboard_panel.png)

## [EFA Supported Instance Types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types)


## Demo Videos
### [Grafana Dashboards](https://youtu.be/KhvCCPjHwCY)

### [Slurm Job Logs](https://youtu.be/RzOkHsmRM3U)

## Tests

Refer [tests](./tests) folder for NCCL and fsx tests.
