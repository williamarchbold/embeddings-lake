import os
import tempfile
from json import dump
from boto3 import client as boto3_client
from numpy import random as np_random
from math import log as math_log
import logging

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)


s3_client = boto3_client("s3")

BUCKET_NAME = os.environ['BUCKET_NAME']


class LSH:
    def __init__(self, dim, num_hashes, bucket_size=100_000, random_seed=42):
        """Initialize LSH object.

        dim: Dimension of vectors
        num_hashes: Number of hash functions (hyperplanes) to use
        bucket_size: Size of hash buckets
        """
        self.dim = dim
        self.num_hashes = num_hashes
        self.bucket_size = bucket_size
        np_random.seed(random_seed)
        self.hyperplanes = np_random.randn(self.num_hashes, self.dim)
    
    @property
    def max_partitions(self):
        return 2**self.num_hashes


def lambda_handler(event, context):
    
    logger.info(event)
    file_name = 'lake_config.json' 

    try:
        aprox_shards = event['lake_aprox_shards']

        lsh = LSH(event['lake_dimensions'], int(math_log(aprox_shards, 2) + 0.5))
        data = {"lake_name": event['lake_name'],
                "lake_dimensions": lsh.dim,
                "lake_shards": lsh.max_partitions,
                "lake_hyperplanes": lsh.hyperplanes.tolist()
                } 
        
        with tempfile.NamedTemporaryFile(mode='w') as temporary_file:
            dump(data, temporary_file, indent=4)
            temporary_file.flush()

            s3_client.upload_file(
                Filename=temporary_file.name, 
                Bucket=BUCKET_NAME, 
                Key=f"{event['lake_name']}/{file_name}"
                ) 
            
            return {
                'statusCode': 200, 
                'body': "Successfully created lake"
            }

    except Exception as e:
        return { 
        'statusCode': 500, 
        'body': str(e)
    }