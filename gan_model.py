import os
import operator
import tensorflow as tf
import settings

class GANModel:
    def __init__(self, model_dir):
        '''
        Initialize graph and session, load saved model

        :param model_dir: directory contained exported GAN model
        '''
        # load saved model
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        self.model = tf.saved_model.loader.load(
            self.sess,
            [tf.saved_model.tag_constants.SERVING],
            model_dir)

        # read input and output tensors
        input_tensor_name = self.model.signature_def['predict_images'].inputs['images'].name
        self.input_tensor = self.graph.get_tensor_by_name(input_tensor_name)
        output_tensor_name = self.model.signature_def['predict_images'].outputs['scores'].name
        self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name)

    def __enter__(self):
        # for using with "with" block
        return self

    def __exit__(self, type_, value, traceback):
        # close session at the end of "with" block
        self.destroy()

    def predict(self, image):
        '''
        Predict the house number on the image using GAN model

        :param image: Byte array, images for prediction
        :return: List of tuples, 3 most probable digits with their probabilities
        '''
        # make prediction
        scores = self.sess.run(self.output_tensor, {self.input_tensor: [image]})

        # return 3 most probable digist with their probabilities
        sorted_scores = sorted(
            enumerate(scores[0]),
            key=lambda x: x[1],
            reverse=True)
        return sorted_scores[0:3]

    def destroy(self):
        '''
        Close TensorFlow session
        '''
        self.sess.close()
