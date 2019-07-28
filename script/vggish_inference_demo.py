# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:15:46 2018

@author: ckc
"""



from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', 'E:/Audio_project/ckc.wav',
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', 'E:/Audio_project/VGGish/samples/ckchh.tfrecord',
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS


def main(_):
  # In this simple example, we run the examples from a single audio file through
  # the model. If none is provided, we generate a synthetic input.
  if FLAGS.wav_file:
    wav_file = FLAGS.wav_file
  else:
    # Write a WAV of a sine wav into an in-memory file object.
    num_secs = 5
    freq = 1000
    sr = 44100
    t = np.linspace(0, num_secs, int(num_secs * sr))
    x = np.sin(2 * np.pi * freq * t)
    # Convert to signed 16-bit samples.
    samples = np.clip(x * 32768, -32768, 32767).astype(np.int16)
    wav_file = six.BytesIO()
    wavfile.write(wav_file, sr, samples)
    wav_file.seek(0)
  examples_batch = vggish_input.wavfile_to_examples(wav_file)
  # Prepare a postprocessor to munge the model embeddings.
  pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)
  # If needed, prepare a record writer to store the postprocessed embeddings.
  writer = tf.python_io.TFRecordWriter(
      FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None
  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)
    # Run inference and postprocessing.
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
    print(embedding_batch)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    print(postprocessed_batch)

    # Write the postprocessed embeddings as a SequenceExample, in a similar
    # format as the features released in AudioSet. Each row of the batch of
    # embeddings corresponds to roughly a second of audio (96 10ms frames), and
    # the rows are written as a sequence of bytes-valued features, where each
    # feature value contains the 128 bytes of the whitened quantized embedding.

    seq_example = tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(
            feature_list={
                vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
                    tf.train.FeatureList(
                        feature=[
                            tf.train.Feature(
                                        bytes_list=tf.train.BytesList(
                                    value=[embedding.tobytes()]))
                            for embedding in postprocessed_batch
                        ]
                    )
            }
        )
    )
    print(seq_example)
    if writer:
      writer.write(seq_example.SerializeToString())

  if writer:
    writer.close()

if __name__ == '__main__':
  tf.app.run()
'''
    feature_list=tf.train.FeatureList(feature=
                                      [tf.train.Feature(bytes_list=
                                                        tf.train.BytesList(value=
                                                        [embedding.tobytes()]))
                                        for embedding in postprocessed_batch])

    feature_lists = {
            vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:feature_list
            }

    frames = tf.train.FeatureLists(feature_list=feature_lists)

    seq_example = tf.train.SequenceExample(feature_lists=frames)
    '''


