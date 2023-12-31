import os
from loguru import logger
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

r2_account = os.getenv("R2_SAFETENSORS_ACCOUNT", "https://a223539ccf6caa2d76459c9727d276e6.r2.cloudflarestorage.com")
r2_bucket = os.getenv("R2_TRANSIENT_BUCKET", "safetensors")

s3_client = boto3.client('s3', endpoint_url=r2_account,  config=Config(signature_version='s3v4'))

@logger.catch(reraise=True)
def generate_presigned_url(client, client_method, method_parameters, expires_in = 1800):
    """
    Generate a presigned Amazon S3 URL that can be used to perform an action.

    :param s3_client: A Boto3 Amazon S3 client.
    :param client_method: The name of the client method that the URL performs.
    :param method_parameters: The parameters of the specified client method.
    :param expires_in: The number of seconds the presigned URL is valid for.
    :return: The presigned URL.
    """
    try:
        url = client.generate_presigned_url(
            ClientMethod=client_method,
            Params=method_parameters,
            ExpiresIn=expires_in
        )
    except ClientError:
        logger.exception(
            f"Couldn't get a presigned URL for client method {client_method}", )
        raise
    # logger.debug(url)
    return url

def generate_safetensor_download_url(filename):
    client = s3_client
    # if not file_exists(client,  f"{procgen_id}.webp"):
    #     client = old_r2
    return generate_presigned_url(
        client = client,
        client_method = "get_object",
        method_parameters = {'Bucket': r2_bucket, 'Key': filename},
        expires_in = 1800
    )

def upload_safetensor(civitai_model):
    try:
        response = s3_client.upload_file(
            civitai_model.get_safetensor_filepath(), r2_bucket, civitai_model.get_safetensor_filename()
        )
    except ClientError as e:
        logger.error(f"Error encountered while uploading metadata {civitai_model.get_safetensor_filename()}: {e}")
        return False

def check_file(client, bucket, filename):
    try:
        return client.head_object(Bucket=bucket, Key=filename)
    except ClientError as e:
        return int(e.response['Error']['Code']) != 404

def check_safetensor(filename):
    return type(check_file(s3_client, r2_bucket, filename)) == dict

def file_exists(client, bucket, filename):
    # If the return of check_file is an int, it means it encountered an error
    return type(check_file(client, bucket, filename)) != int
