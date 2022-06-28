# Use this code snippet in your app.
# If you need more information about configurations or implementing the sample code, visit the AWS docs:   
# https://aws.amazon.com/developers/getting-started/python/

import boto3
import base64
from botocore.exceptions import ClientError
import yaml
import os
import argparse
import ast

def get_secret(secret_name, region_name):

    secret_name = secret_name
    region_name = region_name

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        return client.get_secret_value(
            SecretId=secret_name
        )
        # return get_secret_value_response
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
        else:
            decoded_binary_secret = base64.b64decode(get_secret_value_response['SecretBinary'])
        
    # Your code goes here. 




if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='managing secret/access keys')
    parser.add_argument('--config_path', type=str, default="cluster.yaml",help='path to the config file')
    parser.add_argument('--region', type=str, default="us-west-2",help='AWS region')
    parser.add_argument('--secret_name', type=str,help='name of the AWS secret')

    args = parser.parse_args()

    with open(args.config_path, 'r') as config_file:
        configs = yaml.safe_load(config_file)
        response = get_secret(args.secret_name, args.region)
        keys = ast.literal_eval(response['SecretString'])
 
        for indx, item in enumerate(configs["scrape_configs"]):

            if item['job_name']=="ec2_instances":
                configs["scrape_configs"][indx]['ec2_sd_configs'][0]['secret_key'] = keys['secret_key']
                configs["scrape_configs"][indx]['ec2_sd_configs'][0]['access_key'] = keys['access_key']
        
    with open(args.config_path, 'w') as config_file:
        yaml.dump(configs, config_file)
    config_yaml == args.config_path
    os.system("cp config_yaml /home/ec2-user/aws-parallelcluster-monitoring/prometheus")
        

        
