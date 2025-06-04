import logging
import os
from boto3 import client
import re

logger = logging.getLogger()
logger.setLevel(level=logging.ERROR)

BUCKET_NAME = os.environ['BUCKET_NAME']

s3_client = client("s3")


def get_segments(lake_name):
    segments_in_bucket = []
    prefix = lake_name + "/"
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix, MaxKeys=1000)
    while True:
        if 'Contents' in response:
            for obj in response['Contents']:
                obj_string = obj['Key']
                match = re.search(r"-(\d+)\.", obj_string)
                if match:
                    number = int(match.group(1))       
                    segments_in_bucket.append(number)
        if 'NextContinuationToken' in response:
            logger.info("Found continuation token")
            continuation_token = response['NextContinuationToken']
            response = s3_client.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix=prefix, 
                MaxKeys=1000,
                ContinuationToken=continuation_token
            )
        else:
            logger.info("No more tokens")
            break
    segments_in_bucket.sort()
    logger.info(segments_in_bucket)
    return segments_in_bucket
        

def get_adjacent_segments(lake_name, segment_value, num_shards, radius, segments_in_bucket):
    segment_indices = []
    try:
        hash_index = segments_in_bucket.index(segment_value)
    except ValueError:
        logger.info(f"Hashed query segment {segment_value} does not have a corresponding shard.")
        closest_segment = min(segments_in_bucket, key=lambda x: abs(x - segment_value))
        logger.info(f"Closest segment: {closest_segment}")
        hash_index = segments_in_bucket.index(closest_segment)
        logger.info(f"Closest segment hash index: {hash_index}")
    for delta in range(-radius, radius+1):
        logger.info(f"delta: {delta}")
        candidate_index = delta+hash_index
        logger.info(f"candidate_index: {candidate_index}")
        if candidate_index < 0:
            logger.info("No adjacent segment available.")
            continue
        try:
            candidate_segment = segments_in_bucket[candidate_index]
        except IndexError:
            logger.info("No adjacent segment available.")
            continue          
        candidate_key = f"{lake_name}/segment-{candidate_segment}.parquet"
        logger.info(candidate_key)
        try:
            response = s3_client.head_object(Bucket=BUCKET_NAME, Key=candidate_key)
            segment_indices.append(candidate_segment)
        except Exception:
            logger.info(f"Fragment {candidate_key} does not exist in S3.")
    return list(set(segment_indices))


def lambda_handler(event, context):

    logger.debug(event)

    #n_results: int = 4,
    radius = event['Payload']['radius']
    lake_name = event['Payload']['lake_name']
    embedding = event['Payload']['embedding']
    segment_index = event['Payload']['segment_index']
    num_shards = event["Payload"]['num_shards']
    distance_metric = event["Payload"]['distance_metric']

    segments_in_bucket = get_segments(lake_name=lake_name)

    segment_indices_to_search = get_adjacent_segments(lake_name, segment_index, num_shards, radius, segments_in_bucket)

    segments_as_strings = list(map(str, segment_indices_to_search))

    results = { 
        'segmentIndices': segments_as_strings,
        'embedding': embedding,
        'lakeName': lake_name,
        'distanceMetric': distance_metric
    }

    logger.info(results)

    return results