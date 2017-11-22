## What do we need
1. TensorFlow

2. AWS SDK for Python to read the model from S3 bucket
```
pip install boto3
```

3. [Serverless](https://serverless.com/framework/docs/providers/aws/guide/installation/). For deployment to AWS lambda

4. [Create access key for AWS account](http://docs.aws.amazon.com/general/latest/gr/managing-aws-access-keys.html)

5. Setup AWS credentials in Serverless
```
serverless config credentials --provider aws --key <your_aws_access_key> --secret <your_aws_key_secret>
```

6. [Install AWS console CLI](http://docs.aws.amazon.com/cli/latest/userguide/installing.html)
```
pip install awscli --upgrade --user
aws --version
```
```
aws-cli/1.11.189 Python/3.5.4 Linux/4.4.0-98-generic botocore/1.7.47
```

7. Setup AWS credentials with AWS console CLI:
```
aws configure
```
check ```~/.aws```. There you should find _config.txt_ and _credentials.txt_

8. Export the model as protobuf 

9. Two alternatives for model:
   - Deploy it directly
   - Store the model into S3 bucket, cache in /tmp space (Ephemeral disk capacity) and load it from there

10. Two alternatives for image posting:
    - POST method with image as body (works well on small images)
    - Store test images into S3 bucket + POST method that gets bucket and image name as parameters 
      (more flexible and works better for big images)

11. Preparing TensorFlow for Lambda upload. Create EC2 Ubuntu Free Tier instance
See https://github.com/ryfeus/lambda-packs.

Enable your IP to connect over SSH in Amazon instance

```
ssh ec2-54-183-117-12.us-west-1.compute.amazonaws.com
scp ubuntu@ec2-54-183-117-12.us-west-1.compute.amazonaws.com:~/tf_env.zip ~/Documents/tf_env.zip
```

```
sudo apt-get update
sudo apt-get install -y zip python-dev python-pip
export LC_ALL=C
pip install --upgrade pip
sudo pip install virtualenv
virtualenv tf_env
source tf_env/bin/activate
pip install tensorflow
```
```
touch ~/tf_env/lib/python2.7/site-packages/google/__init__.py
cd ~/tf_env/lib/python2.7/site-packages
zip -r ~/tf_env.zip . --exclude \*.pyc *.DS_Store /external/* /tensorflow/contrib/* /tensorflow/include/unsupported/* /tensorflow/examples/* /tensorboard/* /tensorflow_tensorboard-0.4.0rc3.dist-info/* /pip/* /pip-9.0.1.dist-info/*
```
12. For service deployement:
```
serverless deploy
```
For service removal:
```
serverless remove
```
DO NOT DELETE LAMBDA FUNCTIONS MANUALLY, OTHERWISE SERVERLESS WILL FAIL TO DEPLOY NEXT TIME!
Be aware - Serverless uses your AWS account to put stuff to S3 bucket!

13. Adjust policy for a lambda function role to be able to access S3 bucket