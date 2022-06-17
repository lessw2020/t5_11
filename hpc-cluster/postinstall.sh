#!/bin/bash
set -e
. "/etc/parallelcluster/cfnconfig"

# Override run_instance attributes
cat > /opt/slurm/etc/pcluster/run_instances_overrides.json << EOF
{
    "queue0": {
        "train-p4d24xlarge": {
            "CapacityReservationSpecification": {
                "CapacityReservationTarget": {
                    "CapacityReservationResourceGroupArn": "arn:aws:resource-groups:us-east-1:320567679581:group/EC2CRGroup"
                }
            }
        }
    }
}
EOF
