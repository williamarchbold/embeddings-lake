import os
import uuid
import pytz
import logging
import datetime
import boto3
import numpy as np
import pandas as pd
from math import log2
from json import dumps
from typing import Any
from pydantic import BaseModel
from operator import itemgetter
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest

logger = logging.getLogger()
logger.setLevel(level=logging.ERROR)


BUCKET_NAME = os.environ['BUCKET_NAME']

DISTANCE_L1 = 'l1'
DISTANCE_L2 = "l2"
DISTANCE_COSINE = "cosine"


def l2_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_distance(a, b):
    """ https://stackoverflow.com/questions/58381092/difference-between-cosine-similarity-and-cosine-distance """
    
    return (1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def manhattan_distance(a,b):
    """https://stackoverflow.com/a/72066779"""
    return np.sum(np.abs(a - b))


class HNSW:
    """Navigable small world models are defined as any network with
    (poly/)logarithmic complexity using greedy routing.

    The efficiency of greedy routing breaks down for larger networks
    (1-10K+ vertices) when a graph is not navigable [7].
    """

    def __init__(
        self, distance_type, m=5, ef=200, m0=None, heuristic=True, vectorized=False, random_seed=42
    ):
        self.data = []
        if distance_type == DISTANCE_L2:
            distance_func = l2_distance
        elif distance_type == DISTANCE_COSINE:
            distance_func = cosine_distance
        elif distance_type == DISTANCE_L1:
            logger.info(f"distance function: {distance_type} is L1")
            distance_func = manhattan_distance
        else:
            raise TypeError("Please check your distance type!")

        self.distance_func = distance_func

        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func
            self.vectorized_distance = self.vectorized_distance_

        # the number of edges per node
        self._m = m
        # The number of neighbors to consider for each node in the index construction
        self._ef = ef
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / np.log2(m)
        self._graphs = []
        np.random.seed(random_seed)
        self._enter_point = None

        self._select = self._select_heuristic if heuristic else self._select_naive

    def _distance(self, x, y):
        return self.distance_func(x, [y])[0]

    def vectorized_distance_(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def add(self, elem, ef=None):
        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        # level at which the element will be inserted
        level = int(-log2(np.random.rand()) * self._level_mult) + 1

        # elem will be at data[idx]
        idx = len(data)
        data.append(elem)

        if point is not None:  # the HNSW is not empty, we have an entry point
            dist = distance(elem, data[point])
            # for all levels in which we dont have to insert elem,
            # we search for the closest neighbor
            for layer in reversed(graphs[level:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
            # at these levels we have to insert elem; ep is a heap of entry points.
            ep = [(-dist, point)]
            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                # navigate the graph and update ep with the closest
                # nodes we find
                ep = self._search_graph(elem, ep, layer, ef)
                # insert in g[idx] the best neighbors
                layer[idx] = layer_idx = {}
                self._select(layer_idx, ep, level_m, layer, heap=True)
                # assert len(layer_idx) <= level_m
                # insert backlinks to the new node
                for j, dist in layer_idx.items():
                    self._select(layer[j], (idx, dist), level_m, layer)
                    # assert len(g[j]) <= level_m
                # assert all(e in g for _, e in ep)
        for i in range(len(graphs), level):
            # for all new levels, we create an empty graph
            graphs.append({idx: {}})
            self._enter_point = idx

    def balanced_add(self, elem, ef=None):
        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m
        m0 = self._m0

        idx = len(data)
        data.append(elem)

        if point is not None:
            dist = distance(elem, data[point])
            pd = [(point, dist)]
            for layer in reversed(graphs[1:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
                pd.append((point, dist))
            for level, layer in enumerate(graphs):
                level_m = m0 if level == 0 else m
                candidates = self._search_graph(elem, [(-dist, point)], layer, ef)
                layer[idx] = layer_idx = {}
                self._select(layer_idx, candidates, level_m, layer, heap=True)
                # add reverse edges
                for j, dist in layer_idx.items():
                    self._select(layer[j], [idx, dist], level_m, layer)
                    assert len(layer[j]) <= level_m
                if len(layer_idx) < level_m:
                    return
                if level < len(graphs) - 1:
                    if any(p in graphs[level + 1] for p in layer_idx):
                        return
                point, dist = pd.pop()
        graphs.append({idx: {}})
        self._enter_point = idx

    def search(self, q, k=None, ef=None):
        """Find the k points closest to q."""

        distance = self.distance
        graphs = self._graphs
        point = self._enter_point

        logger.info(f"HNSW search()")
        logger.info(f"point: {point}")

        if ef is None:
            ef = self._ef

        if point is None:
            raise ValueError("Empty graph")

        dist = distance(q, self.data[point])
        logger.info(f"dist: {dist}")
        # look for the closest neighbor from the top to the 2nd level
        for layer in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1(q, point, dist, layer)
        logger.info(f"second level status")
        logger.info(f"dist: {dist}")
        logger.info(f"point: {point}")
        # look for ef neighbors in the bottom level
        ep = self._search_graph(q, [(-dist, point)], graphs[0], ef)

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        return [(idx, -md) for md, idx in ep]

    def _search_graph_ef1(self, q, entry, dist, layer):
        """Equivalent to _search_graph when ef=1."""

        vectorized_distance = self.vectorized_distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = {entry}

        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))

        return best, best_dist

    def _search_graph(self, q, ep, layer, ef):
        vectorized_distance = self.vectorized_distance
        data = self.data

        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = {p for _, p in ep}

        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            dists = vectorized_distance(q, [data[e] for e in edges])
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

    def _select_naive(self, d, to_insert, m, layer, heap=False):
        if not heap:
            idx, dist = to_insert
            assert idx not in d
            if len(d) < m:
                d[idx] = dist
            else:
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:
                    del d[max_idx]
                    d[idx] = dist
            return

        assert not any(idx in d for _, idx in to_insert)
        to_insert = nlargest(m, to_insert)  # smallest m distances
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, d.items(), key=itemgetter(1))
        else:
            checked_del = []
        for md, idx in to_insert:
            d[idx] = -md
        zipped = zip(checked_ins, checked_del)
        for (md_new, idx_new), (idx_old, d_old) in zipped:
            if d_old <= -md_new:
                break
            del d[idx_old]
            d[idx_new] = -md_new
            assert len(d) == m

    def _select_heuristic(self, d, to_insert, m, g, heap=False):
        nb_dicts = [g[idx] for idx in d]

        def prioritize(idx, dist):
            return any(nd.get(idx, float("inf")) < dist for nd in nb_dicts), dist, idx

        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
        else:
            to_insert = nsmallest(
                m, (prioritize(idx, -mdist) for mdist, idx in to_insert)
            )

        assert len(to_insert) > 0
        assert not any(idx in d for _, _, idx in to_insert)

        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(
                to_check, (prioritize(idx, dist) for idx, dist in d.items())
            )
        else:
            checked_del = []
        for _, dist, idx in to_insert:
            d[idx] = dist
        zipped = zip(checked_ins, checked_del)
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break
            del d[idx_old]
            d[idx_new] = d_new
            assert len(d) == m

    def __getitem__(self, idx):
        for g in self._graphs:
            try:
                yield from g[idx].items()
            except KeyError:
                return

class LazyBucket(BaseModel):

    db_location: str
    segment_index: str
    distance_metric: str
    bucket_name: str = "segment-{}.parquet"
    metadata_name: str = "segment-{}-metadata.json"
    loaded: bool = False
    dirty: bool = False
    frame: Any | None = None
    frame_schema: str = ["id", "vector", "metadata", "document", "timestamp"]
    vectors = []
    dirty_rows = []
    hnsw: Any = None
    attrs: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hnsw = HNSW(self.distance_metric, m0=5, ef=10)

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
            logger.debug("Laready exists not gonna read os path")
            return
        if os.path.exists(self.frame_location):
            logger.debug("os path exists. Reading frame")
            self.frame = pd.read_parquet(self.frame_location)
            logger.debug("First row")
            logger.debug(self.frame.iloc[0])
        else:
            self.frame = pd.DataFrame(columns=self.frame_schema)
            self.attrs = self.frame.attrs
        if list(self.frame.columns) != self.frame_schema:
            raise ValueError(f"Invalid frame_schema {self.frame.columns=}")
        
        self.loaded = True
        for v in self.frame["vector"].tolist():
            self.hnsw.add(v)

    def append(self, vector: np.ndarray, **attrs):
        if not self.loaded:
            self._lazy_load()

        uid = uuid.uuid1().urn

        document = {
            "id": uid,
            "vector": vector.tolist(),
            "metadata": attrs.get("metadata", {"name": "unknown"}),
            "document": attrs.get("document", ""),
            "timestamp": attrs.get("timestamp", datetime.datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S %Z")),
        }
        self.frame = self.frame._append(document, ignore_index=True)
        self.hnsw.add(vector)
        self.dirty = True

        return uid
    
    def search(self, vector: np.ndarray, k: int = 4):
        self._lazy_load()
        try:
            results = self.hnsw.search(vector, k)
        except ValueError:  # Empty graph
            return []
        print("search()")
        print(f"results: {results}")
        return results

    def sync(self, **attrs):
        if not self.dirty:
            return

        if self.frame.empty:
            return
        # TODO: eval last sync time
        # self.frame.attrs["last_update"] = datetime.datetime.now(pytz.UTC)
        now_dt = datetime.datetime.now(pytz.UTC)
        self.frame.attrs["last_update"] = dumps(now_dt, indent=4, sort_keys=True, default=str)
        for k, v in attrs.items():
            self.frame.attrs[k] = v

        os.makedirs(self.db_location, exist_ok=True)
        self.frame.to_parquet(self.frame_location, engine='pyarrow', compression="gzip")
        self.dirty = False

    def delete(self):
        """This function deletes a file if it exists at a specified location.

        :return: If the file specified by `self.frame_location` does not exist, the function returns nothing
        (i.e., `None`). If the file exists and is successfully deleted, the function also returns nothing.
        If an exception occurs during the deletion process, the function catches the exception and does not
        re-raise it, so it also returns nothing.
        """
        if not os.path.exists(self.frame_location):
            return
        try:
            os.remove(self.frame_location)
        except Exception:
            ...

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
        key = f"{self.lake_name}/{self.key}"
        logger.info(f"Loading fragment {self.remote_location}/{key} from S3")
        # Check if object exists in S3
        # try:
        #     response = self.s3_client.head_object(Bucket=self.remote_location, Key=key)
        #     #response = self.s3_client.get_object(Bucket=self.remote_location, Key=key)
        #     logger.debug(response)
        # except Exception:
        #     logger.info("Fragment does not exist in S3")
        #     super()._lazy_load()
        # except Exception as e:
        #     logger.exception(f"Unexpected error while checking for fragment in S3: {e}")
        #else:
        #logger.info("Fragment exists in S3, downloading...")
        logger.debug("Downloading fragment from S3...")
        os.makedirs(os.path.dirname(self.frame_location), exist_ok=True)
        self.s3_client.download_file(
            self.remote_location, key, Filename=self.frame_location
        )
        logger.debug(f"Downloaded S3 file from {self.remote_location} bucket w/ key {key} to {self.frame_location} ")
        super()._lazy_load()

    def sync(self):
        if not self.dirty:
            return
        super().sync()
        if self.frame.empty:
            return
        logger.info(f"Uploading fragment {self.key} to S3")
        self.s3_client.upload_file(
            Filename = self.frame_location,
            Bucket = self.remote_location,
            Key = f"{self.lake_name}/{self.bucket_name.format(self.segment_index)}",
            Callback=self.upload_progress_callback(
                self.bucket_name.format(self.segment_index)
            ),
        )
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

    def delete_local(self):
        super().delete()

    def delete_remote(self):
        try:
            logger.info(f"Deleting fragment {self.key} from S3")
            self.s3_client.delete_object(
                Bucket=self.remote_location,
                Key=self.key,
            )
        except Exception:
            logger.exception("Failed to delete object from S3")

    def delete(self):
        super().delete()
        self.delete_remote()

def query(bucket, search_vector, top_k):

    results = []
    # Search the bucket
    closest_indices_d = bucket.search(search_vector, k=top_k)
    logger.debug("closest_indices_d")
    logger.debug(closest_indices_d)
    # Load the dirty rows
    bucket.frame["vector"] = bucket.frame["vector"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    dirty_rows = bucket.frame.to_dict(orient="records")
    # Save both distance value and idx together
    for idx, distance in closest_indices_d:
        results.append((
            distance,
            dirty_rows[idx]
        ))
        logger.debug("distance, dirty row")
        logger.debug(f"{distance}, {dirty_rows[idx]['metadata']['file_path']}")
    # Remove Duplicates and Sort based on the distance
    results.sort(key=lambda x: x[0])
    unique_results = list({row["id"]: {**row, "distance":float(dist)} for dist, row in results}.values())

    for r in unique_results:

        logger.debug(f"r - {r}")
        del r['vector']
    
    logger.debug("unique results sans vectors")
    logger.debug(unique_results)

    #vectors_ret = [result["vector"] for result in unique_results]
    return unique_results#, vectors_ret


def lambda_handler(event, context):

    logger.info(event['segmentIndex'])
    # Event Handler
    lake_name = event['lakeName']
    segment_index = event['segmentIndex']
    search_embedding = event['embedding']
    distance_metric = event['distanceMetric']
    top_k = 4

    # Initiate Bucket
    #if lake_name.startswith("s3://"):
    bucket = S3Bucket(
        db_location=lake_name,
        segment_index=segment_index,
        distance_metric=distance_metric
    )

    #results, _ = query(bucket, np.array(search_embedding), top_k)
    results = query(bucket, np.array(search_embedding), top_k)

    logger.info(f"Found {len(results)} results")
    logger.debug(f"First result - {results[0]}")

    return results