'''
Utilities for using in a project scope
'''
import io
import os
import zipfile

import boto3
import botocore

import settings


def create_model_dir():
    '''
    Check target model directory for existence and create it if needed.
    We use /tmp directory, which is according to AWS Limits
    (see http://docs.aws.amazon.com/lambda/latest/dg/limits.html)
    is Ephemeral disk capacity ("/tmp" space) and can be 512 MB big.

    :return: Path of the model directory
    '''
    # Target directory for GAN model - it must be a subdirectory of /tmp
    # Check target directory for existence and create it if needed.
    model_dir = '/tmp/gan_model'
    if not os.path.exists(model_dir):
        print 'Going to create a model directory {}...'.format(model_dir)
        os.makedirs(model_dir)
        print '...success!'
    print 'Model directory is {}'.format(model_dir)

    return model_dir


def download_model_from_bucket(model_dir):
    '''
    Downloads GAN model protobuf from S3 bucket if it was not downloaded yet
    '''
    # check the model file for existence and download if needed
    model_path = model_dir + '/' + settings.MODEL_PROTOBUF_FILE_NAME
    if not os.path.isfile(model_path):
        print 'Going to download a model file from S3 bucket {}/{}...'.format(
            settings.S3_BUCKET_NAME, settings.MODEL_ZIP_FILE_NAME
        )

        # create S3 resource
        s3_res = boto3.resource('s3')
        model_zip_file = model_dir + '/' + settings.MODEL_ZIP_FILE_NAME

        try:
            # download ZIP
            s3_bucket = s3_res.Bucket(settings.S3_BUCKET_NAME)
            s3_bucket.download_file(
                settings.MODEL_ZIP_FILE_NAME,
                model_zip_file)

            # extract everything from zip
            with zipfile.ZipFile(model_zip_file, 'r') as zip_ref:
                zip_ref.extractall(model_dir)

            # delete zip
            os.remove(model_zip_file)
        except botocore.exceptions.ClientError as exception:
            if exception.response['Error']['Code'] == "404":
                print "The object does not exist."
            if exception.response['Error']['Code'] == "403":
                print "Access denied."
            raise


def download_image_from_bucket(bucket_name, key):
    '''
    Download image from S3 bucket

    :param bucket_name: AWS S3 bucket name
    :param key: key in the bucket, efectively an image file name
    :return: image as byte array
    '''

    # create S3 resource
    s3_res = boto3.resource('s3')
    try:
        # download image into in-memory buffer
        print 'Downloading the image from S3 bucket {}/{}'.format(bucket_name, key)
        s3_bucket = s3_res.Bucket(bucket_name)
        data = io.BytesIO()

        s3_bucket.download_fileobj(
            key,
            data)
        data.seek(0)
        print 'Successfully downloaded the image'

        return data.read()

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print "The object does not exist."
        else:
            raise
