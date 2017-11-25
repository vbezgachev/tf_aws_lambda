#!/bin/bash
# run this file with 'source env_var.sh' or '. ./env_var.sh' to export variable
# into current shell environment
export TF_AWS_MODEL_ZIP_FILE_NAME='gan_model.zip'
export TF_AWS_MODEL_PROTOBUF_FILE_NAME='saved_model.pb'
export TF_AWS_S3_MODEL_BUCKET_NAME='vb-tf-aws-lambda-model'
