'''
For test purpose only. Allows loading of a model and an image locally or from S3 bucket,
and making prediction of loaded image
'''
import os
import zipfile
import json

import settings
import utils

from gan_model import GANModel


def download_model_from_local_file(model_dir):
    '''
    Copy saved model locally

    :param model_dir: Target directory to copy the model
    '''
    # check the model file for existence amd copy if needed
    protobuf_file_name = utils.get_env_var_or_raise_exception(
        settings.MODEL_PROTOBUF_FILE_NAME_ENV_VAR)
    model_path = model_dir + '/' + protobuf_file_name
    if not os.path.isfile(model_path):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        model_zip_file_name = utils.get_env_var_or_raise_exception(
            settings.MODEL_ZIP_FILE_NAME_ENV_VAR)
        model_zip_file_path = current_directory + '/model' +\
                        '/' + model_zip_file_name
        print 'Going to copy a model file from {}...'.format(model_zip_file_path)
        with zipfile.ZipFile(model_zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)


def main():
    '''
    Entry point
    '''
    model_dir = utils.create_model_dir()
    # download_model_from_local_file(model_dir)
    utils.download_model_from_bucket(model_dir)

    # create model...
    with GANModel(model_dir) as gan_model:
        # ...load the image
        # image_file_name = './svnh_test_images/image_3.jpg'
        # with open(image_file_name, 'rb') as f:
        #     image = f.read()
        image = utils.download_image_from_bucket('vb-tf-aws-lambda-images', 'image_3.jpg')

        # ...make prediction
        scores = gan_model.predict(image)

        # print results
        results_json = [{'digit': str(score[0]), 'probability': str(score[1])} for score in scores]
        print "Scores: {}".format(json.dumps(results_json))


if __name__ == '__main__':
    main()
