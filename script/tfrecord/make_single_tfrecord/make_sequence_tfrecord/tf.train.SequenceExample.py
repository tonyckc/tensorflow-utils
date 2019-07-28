# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 10:57:19 2019

@author: ckc
"""

import tensorflow as tf

movie_1_actors = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'ckc', b'wmy']))
movie_2_actors = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'ckc1', b'wmy1']))
movie_actors_list = [movie_1_actors, movie_2_actors]
movie_actors = tf.train.FeatureList(feature=movie_actors_list)

# short form

movie_names = tf.train.FeatureList(feature=
        [tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[b'hh1', b'hh2']))
        ])
movie_ratings = tf.train.FeatureList(feature=[
        tf.train.Feature(float_list=tf.train.FloatList(
            value=[9.7, 9.0]))
        ])

movie_dict = {
        'movie_names': movie_names,
        'movie_actors': movie_actors,
        'movie_ratings': movie_ratings
        }

movies = tf.train.FeatureLists(feature_list=movie_dict)

example = tf.train.SequenceExample(feature_lists=movies)

print(example)

# Write TFrecord file
with tf.python_io.TFRecordWriter('customer_1.tfrecord') as writer:
    writer.write(example.SerializeToString())
# Read and print data:
sess = tf.InteractiveSession()
# Read TFRecord file
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(['customer_1.tfrecord'])
_, serialized_example = reader.read(filename_queue)
# Define features

sequence_features = {
    'Movie Names': tf.FixedLenSequenceFeature([], dtype=tf.string),
    'Movie Ratings': tf.FixedLenSequenceFeature([], dtype=tf.float32),
    'Movie Actors': tf.VarLenFeature(dtype=tf.string)
}
# Extract features from serialized data
sequence_data = tf.parse_single_sequence_example(
    serialized=serialized_example,
    sequence_features=sequence_features)
# Many tf.train functions use tf.train.QueueRunner,
# so we need to start it before we read
tf.train.start_queue_runners(sess)
# Print features
print('\nData')
for name, tensor in sequence_data.items():
    print('{}: {}'.format(name, tensor.eval()))

