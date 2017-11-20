import os
import zipfile

import boto3
import botocore
import tensorflow as tf

import settings
import utils

from gan_model import GANModel


def download_model_from_S3_bucket(model_dir):
    '''
    Downloads GAN model protobuf from S3 bucket if was not downloaded yet
    '''
    # check the model file for existence amd copy if needed
    model_path = model_dir + '/' + settings.MODEL_PROTOBUF_FILE_NAME
    if not os.path.isfile(model_path):
        print('Going to download a model file from S3 bucket {}/{}...'.format(
            settings.S3_BUCKET_NAME, settings.MODEL_ZIP_FILE_NAME
        ))

        # download ZIP
        s3 = boto3.resource('s3')
        model_zip_file = model_dir + '/' + settings.MODEL_ZIP_FILE_NAME

        try:
            s3_bucket = s3.Bucket(settings.S3_BUCKET_NAME)
            s3_bucket.download_file(
                settings.MODEL_ZIP_FILE_NAME,
                model_zip_file)

            # extract everything from zip
            with zipfile.ZipFile(model_zip_file, 'r') as zip_ref:
                zip_ref.extractall(model_dir)

            # delete zip
            os.remove(model_zip_file)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

def download_model_from_local_file(model_dir):
    '''
    For tests only - copy saved model locally
    '''
    # check the model file for existence amd copy if needed
    model_path = model_dir + '/' + settings.MODEL_PROTOBUF_FILE_NAME
    if not os.path.isfile(model_path):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        model_zip_file = current_directory + '/model' +\
                        '/' + settings.MODEL_ZIP_FILE_NAME
        print('Going to copy a model file from {}...'.format(model_zip_file))
        with zipfile.ZipFile(model_zip_file, 'r') as zip_ref:
            zip_ref.extractall(model_dir)


def load_and_predict_with_saved_model(model_dir):
    '''
    Loads saved as protobuf model and make prediction on a single image

    :param model_dir: directory containd saved model
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        # restore save model
        model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)
        # print(model)
        loaded_graph = tf.get_default_graph()

        # get necessary tensors by name
        input_tensor_name = model.signature_def['predict_images'].inputs['images'].name
        input_tensor = loaded_graph.get_tensor_by_name(input_tensor_name)
        output_tensor_name = model.signature_def['predict_images'].outputs['scores'].name
        output_tensor = loaded_graph.get_tensor_by_name(output_tensor_name)

        # make prediction
        image_file_name = './svnh_test_images/image_3.jpg'
        with open(image_file_name, 'rb') as f:
            image = f.read()
            scores = sess.run(output_tensor, {input_tensor: [image]})

        # print results
        print("Scores: {}".format(scores))


def main():
    model_dir = utils.create_model_dir()
    # download_model_from_local_file(model_dir)
    download_model_from_S3_bucket(model_dir)
    # load_and_predict_with_saved_model(model_dir)

    # create model...
    with GANModel(model_dir) as gan_model:
        # ...and make prediction
        # image_file_name = './svnh_test_images/image_3.jpg'
        # with open(image_file_name, 'rb') as f:
        #     image = f.read()
        image = utils.download_image_from_S3_bucket('vb-tf-aws-lambda-images', 'image_3.jpg')
        scores = gan_model.predict(image)

        # print results
        print("Scores: {}".format(scores))


if __name__ == '__main__':
    main()
