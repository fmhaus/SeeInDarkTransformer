import cv2
import numpy as np
import io
import boto3
import json

class JsonUserdata:
    def __init__(self, file):
        with open(file) as fr:
            self.dict = json.load(fr)

    def get(self, key):
        return self.dict[key]
    
class S3ObjectStorage:
    def __init__(self, userdata):
        self.s3 = boto3.client(
                service_name ="s3",
                endpoint_url = userdata.get("s3_endpoint"),
                aws_access_key_id = userdata.get("s3_access_key"),
                aws_secret_access_key = userdata.get("s3_access_secret"),
                region_name="auto",
        )
        self.s3_bucket = "bach100"

    def load_image(self, key):
        response = self.s3.get_object(Bucket=self.s3_bucket, Key=key)
        image_raw = np.frombuffer(response["Body"].read(), np.uint8)
        image = cv2.imdecode(image_raw, cv2.IMREAD_COLOR)
        return image

    def load_file(self, key, binary=False):
        response = self.s3.get_object(Bucket=self.s3_bucket, Key=key)
        if binary:
            buffer = io.BytesIO(response["Body"].read())
            return buffer
        else:
            return response["Body"].read().decode("utf-8")

    def store_file(self, key, data, binary=False):
        if not binary:
            data = data.encode("utf-8")
        self.s3.put_object(
            Bucket=self.s3_bucket,
            Key=key,
            Body=data
        )

    def file_exists(self, key):
        try:
            self.s3.head_object(Bucket=self.s3_bucket, Key=key)
            return True
        except Exception as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                raise

    # limit is 1000
    def list_files_limit(self, path, limit = 1000):
        if limit > 1000:
            raise
        
        response = self.s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=path, MaxKeys=limit)
    
        return list(map(lambda obj : obj["Key"], response["Contents"]))

    def list_files_all_lambda(self, path, page_lambda):
        paginator = self.s3.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self.s3_bucket, Prefix=path)

        for page in page_iterator:
            if "Contents" in page:
                keys = map(lambda obj : obj["Key"], page["Contents"])
                res = page_lambda(keys)
                if res is not None and res == False:
                    break

    def list_files_all(self, path):
        result = []
        self.list_files_all_lambda(path, lambda l : result.extend(l))
        return result