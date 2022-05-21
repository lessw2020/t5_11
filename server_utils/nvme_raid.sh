#!/bin/bash
N_DRIVES=8 # max number of drives


for DISK in $(seq 1 $N_DRIVES)
do
sed -e 's/\s*\([\+0-9a-zA-Z]*\).*/\1/' << EOF | sudo fdisk /dev/nvme${DISK}n1
  o # clear the in memory partition table
  n # new partition
  p # primary partition
  1 # partition number 1
    # default - start at beginning of disk
    # default - stop at end of disk
  p # print the in-memory partition table
  t # change partition type
  fd #
  p # print the in-memory partition table
  w # save
  q # and we're done
EOF
done

# do some checking
mdadm --examine /dev/nvme[1-$N_DRIVES]n1

# create the raid devise, echo y for the create array validation
echo y | mdadm --create /dev/md0 -l raid0 -n $N_DRIVES /dev/nvme[1-$N_DRIVES]n1

# display some information
cat /proc/mdstat
mdadm --examine /dev/nvme[1-$N_DRIVES]n1
mdadm --detail /dev/md0

# format the file system
yum install xfsprogs -y

# use 256k block size:
# see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-io-characteristics.html
mkfs.xfs -d su=256k -d sw=$N_DRIVES /dev/md0 

# and mount
mkdir /mnt/raid0
mount /dev/md0 /mnt/raid0/

echo '/dev/md0    /mnt/raid0  xfs    defaults        0   0' | sudo tee -a /etc/fstab

# saves the raid configurations
mdadm --detail --scan --verbose | sudo tee -a /etc/mdadm.conf
cat /etc/mdadm.conf

# dirty
chmod 777 /mnt/raid0

