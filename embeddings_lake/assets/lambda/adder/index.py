import os
import uuid
import pytz
import logging
import datetime
import boto3
import numpy as np
import pandas as pd
import json
from typing import Any
from pydantic import BaseModel

logger = logging.getLogger()
logger.setLevel(level=logging.ERROR)


BUCKET_NAME = os.environ['BUCKET_NAME']


class LazyBucket(BaseModel):

    db_location: str
    segment_index: str
    bucket_name: str = "segment-{}.parquet"
    metadata_name: str = "segment-{}-metadata.json"
    loaded: bool = False
    dirty: bool = False
    frame: Any | None = None
    frame_schema: str = ["id", "vector", "metadata", "document", "timestamp"]
    vectors = []
    dirty_rows = []
    attrs: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.segment_index=} {len(self.vectors)=} {self.dirty=} {self.loaded=} )>"

    @property
    def key(self):
        return self.bucket_name.format(self.segment_index)

    @property
    def frame_location(self):
        bucket_name = self.bucket_name.format(self.segment_index)
        return f"{self.db_location}/{bucket_name}"

    def _lazy_load(self):
        if self.loaded:
            logger.debug("_lazy_load() loaded")
            return

        if os.path.exists(self.frame_location):
            self.frame = pd.read_parquet(self.frame_location)
            logger.debug("os path exists. Frame read")
        else:
            self.frame = pd.DataFrame(columns=self.frame_schema)
            self.attrs = self.frame.attrs
            logger.debug("os path doesn't exist. New frame.")
        if list(self.frame.columns) != self.frame_schema:
            raise ValueError(f"Invalid frame_schema {self.frame.columns=}")
        
        self.loaded = True

    def append(self, vector: np.ndarray, **attrs):
        if not self.loaded:
            logger.debug("Lazy bucket not loaded")
            self._lazy_load()
            logger.debug("Lazy bucket loaded")
        uid = uuid.uuid1().urn

        document = {
            "id": uid,
            "vector": vector.tolist(),
            "metadata": attrs.get("metadata", {"name": "unknown"}),
            "document": attrs.get("document", ""),
            "timestamp": attrs.get("timestamp", datetime.datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S %Z")),
        }
        self.frame = self.frame._append(document, ignore_index=True)
        self.dirty = True

        return uid

    def sync(self, **attrs):
        if not self.dirty:
            return

        if self.frame.empty:
            return
        now_dt = datetime.datetime.now(pytz.UTC)
        self.frame.attrs["last_update"] = json.dumps(now_dt, indent=4, sort_keys=True, default=str)
        for k, v in attrs.items():
            self.frame.attrs[k] = v

        os.makedirs(self.db_location, exist_ok=True)
        self.frame.to_parquet(self.frame_location, engine='pyarrow', compression="gzip")
        self.dirty = False

    # def delete(self):
    #     """This function deletes a file if it exists at a specified location.

    #     :return: If the file specified by `self.frame_location` does not exist, the function returns nothing
    #     (i.e., `None`). If the file exists and is successfully deleted, the function also returns nothing.
    #     If an exception occurs during the deletion process, the function catches the exception and does not
    #     re-raise it, so it also returns nothing.
    #     """
    #     if not os.path.exists(self.frame_location):
    #         return
    #     try:
    #         os.remove(self.frame_location)
    #     except Exception:
    #         ...

    def __len__(self):
        if not self.loaded:
            self._lazy_load()
        return len(self.vectors)

    def memory_footprint(self):
        return self.frame.memory_usage(deep=True).sum()

    def delete_local(self):
        return self.delete()

    def delete_remote(self):
        return ...


class S3Bucket(LazyBucket):
    remote_location: str = ""
    bytes_transferred: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.remote_location = BUCKET_NAME
        self.db_location = self.local_storage

    @property
    def lake_name(self):
        db_l = self.db_location.split("/")[-1]
        db_l_2 = db_l.split("_")[-1]
        return db_l_2

    @property
    def local_storage(self):
        return f"/tmp/embeddings_lake_{self.db_location}"

    @property
    def s3_client(self):
        return boto3.client("s3")

    def _lazy_load(self):
        if self.loaded:
            return
        logger.info(f"Loading fragment {self.key} from S3")
        logger.debug(f"Loading fragment {self.lake_name}/{self.bucket_name.format(self.segment_index)} from S3")
        logger.debug(f"Loading fragment {self.frame_location} from S3")
        s3_object_key = f"{self.lake_name}/{self.key}"
        logger.debug(f"s3_object_key: {s3_object_key}")
        # Check if object exists in S3
        try:
            self.s3_client.head_object(
                Bucket=self.remote_location,
                Key=s3_object_key
                )
        except Exception:
            logger.info(f"Fragment {s3_object_key} does not exist in S3")
            super()._lazy_load()
        # except Exception as e:
        #     logger.exception(f"Unexpected error while checking for fragment in S3: {e}")
        else:
            logger.info("Fragment exists in S3, downloading...")
            os.makedirs(os.path.dirname(self.frame_location), exist_ok=True)
            result = self.s3_client.download_file(
                Bucket=self.remote_location,
                Key =s3_object_key,
                Filename=self.frame_location
            )
            logger.debug(f"Download fragment result - {result}")
            super()._lazy_load()

    def sync(self):
        if not self.dirty:
            return
        super().sync()
        if self.frame.empty:
            return
        logger.debug(f"Uploading fragment {self.key} to S3")
        result = self.s3_client.upload_file(
            Filename = self.frame_location,
            Bucket = self.remote_location,
            Key = f"{self.lake_name}/{self.bucket_name.format(self.segment_index)}",
            Callback=self.upload_progress_callback(
                self.bucket_name.format(self.segment_index)
            ),
        )
        logger.debug(f"sync result - {result}")
        self.dirty = False

    def upload_progress_callback(self, key):
        def upload_progress_callback(bytes_transferred):
            self.bytes_transferred += bytes_transferred
            logger.debug(
                "\r{}: {} bytes have been transferred".format(
                    key, self.bytes_transferred
                ),
                end="",
            )

    # def delete_local(self):
    #     super().delete()

    # def delete_remote(self):
    #     try:
    #         logger.debug(f"Deleting fragment {self.key} from S3")
    #         self.s3_client.delete_object(
    #             Bucket=self.remote_location,
    #             Key=self.key,
    #         )
    #     except Exception:
    #         logger.exception("Failed to delete object from S3")

    # def delete(self):
    #     super().delete()
    #     self.delete_remote()


def lambda_handler(event, context):

    logger.debug(event['Records'][0]['body'])

    message_body = json.loads(event['Records'][0]['body'])

    logger.info(message_body)

    lake_name = message_body['lake_name']
    segment_index = message_body['segment_index']
    embedding = message_body['embedding']
    document = message_body['document']
    metadata = message_body['metadata']

    result = {
        'lakeName': lake_name,
        'document': document,
        'metadata': metadata,
        'segmentIndex': segment_index 
    }

    # Initiate Bucket
    bucket = S3Bucket(
        db_location=lake_name,
        segment_index=segment_index,
    )

    # Store Embeddings
    # Will create new segment if segment not exist already
    uid = bucket.append(
        vector= np.array(embedding),
        metadata = metadata,
        document = document
    )

    try:
        bucket.sync()
        result['success'] = True
    except:
        result['success'] = False
        logger.error("Failed to add embedding")
    logger.debug(f"Number of Vectors: {len(bucket.frame)}")

    return result