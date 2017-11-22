import io
import os

import boto3
import botocore


def create_model_dir():
    '''
    Check target model directory for existence and create it if needed.
    We use /tmp directory, which is according to AWS Limits
    (see http://docs.aws.amazon.com/lambda/latest/dg/limits.html)
    is Ephemeral disk capacity ("/tmp" space) and can be 512 MB big

    :return: Path of the model directory
    '''
    # Check target directory for existence and create it if needed.
    model_dir = '/tmp/gan_model'
    if not os.path.exists(model_dir):
        print('Going to create a model directory {}...'.format(model_dir))
        os.makedirs(model_dir)
        print('...success!')

    return model_dir


def download_image_from_S3_bucket(bucket_name, key):
    '''
    Download image from S3 bucket

    :param bucket_name: AWS S3 bucket name
    :param key: key in the bucket, efectively an image file name
    :return: image as byte array
    '''
    s3 = boto3.resource('s3')
    try:
        print('Downloading the image from S3 bucket {}/{}'.format(bucket_name, key))
        s3_bucket = s3.Bucket(bucket_name)
        data = io.BytesIO()

        s3_bucket.download_fileobj(
            key,
            data)
        data.seek(0)
        print('Successfully downloaded the image')

        return data.read()

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
