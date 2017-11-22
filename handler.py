import os
import sys
import json

'''
This is needed so that the script running on AWS will pick up the pre-compiled dependencies
from the vendored folder
'''
current_location = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_location, 'vendored'))

from gan_model import GANModel
import utils

'''
Declare global objects living across requests
'''
model_dir = utils.create_model_dir()
utils.download_model_from_S3_bucket(model_dir)
gan_model = GANModel(model_dir)


def get_param_from_url(event, param_name):
    '''
    Retrieve query parameters from a Lambda call. Parameters are passed through the
    event object as a dictionary. We interested in 'queryStringParameters', since
    the bucket name and the key are passed in the query string

    :param event: the event as input in the Lambda function
    :param param_name: the name of the parameter in the query string
    :return: the parameter value
    '''
    params = event['queryStringParameters']
    if param_name in params:
        return params[param_name]
    return None


def lambda_gateway_response(code, body):
    '''
    This function wraps around the endpoint responses in a uniform and Lambda-friendly way.
    We have to return an HTTP response: status code, content-type in header and body

    :param code: HTTP response code (200 for OK), must be integer
    :param body: response body as JSON
    '''
    return {"statusCode": code,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body)}


def predict(event, context):
    '''
    The function is called by AWS Lambda. Sample call:

    {LambdaURL}/{stage}/predict?bucket=vb-tf-aws-lambda-images&key=image_3.jpg

    {LambdaURL} is Lambda URL as returned by serveless installation and {stage} is set in the
    serverless.yml file (dev in our case).

    :param event: AWS Lambda uses this parameter to pass in event data to the handler.
                  This parameter is usually of the Python dict type. It can also be
                  list, str, int, float, or NoneType type.
    :param context: AWS Lambda uses this parameter to provide runtime information
                    to the handler. This parameter is of the LambdaContext type.
    :return: JSON with status code and result JSON object
    '''

    try:
        bucket_name = get_param_from_url(event, 'bucket')
        key = get_param_from_url(event, 'key')

        print 'Predict function called! Bucket/key is {}/{}'.format(bucket_name, key)

        if bucket_name and key:
            image = utils.download_image_from_S3_bucket(bucket_name, key)
            results = gan_model.predict(image)
            results_json = [{'digit': str(res[0]),
                             'probability': str(res[1])} for res in results]
            print 'Results retrieved: {}'.format(results_json)
        else:
            message = 'Input parameters are invalid: bucket name and key must be specified'
            return lambda_gateway_response(400, {'message': message})
    except Exception as exception:
        error_response = {
            'error_message': "Unexpected error",
            'stack_trace': str(exception)
        }
        return lambda_gateway_response(503, error_response)

    return lambda_gateway_response(200, {'prediction_result': results_json})
