import io
import logging
import string
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
import webrtcvad
from tensorflow import keras

import wav_split

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

# An integer scalar Tensor. The window length in samples.
frame_length = 256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384

lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80

# Path to the H5 file
model_path = "C:\LearnIT\Speech_to_text\mymodel (1).h5"


def CTCLoss(y_true, y_pred):
    """
    Define CTC loss function
    """
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model(model_path, custom_objects={'CTCLoss': CTCLoss}, compile=False)
# Show the model architecture
model.summary()


def encode_single_buffer(audio_numpy_array):
    """
    Describes the transformation that we apply to each element of our dataset
    """
    ###########################################
    # #  Process the Audio
    ##########################################
    # 1. Read wav file
    # file = tf.io.read_file(wav_file)
    # 2. Decode the wav file

    audio_tensor = tf.convert_to_tensor(audio_numpy_array, dtype=tf.string)
    audio, sampling_rate = tf.audio.decode_wav(audio_tensor)

    # Check for stereo audio (2 channels) and convert to mono if necessary
    if audio.shape[-1] == 2:
        # Convert stereo to mono by averaging both channels
        audio = tf.reduce_mean(audio, axis=-1, keepdims=True)

    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    stfts = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(stfts)  # get absolute value of complex number
    spectrogram = tf.math.pow(spectrogram, 2)  # get power

    # 6. mel spectrogram
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins,
                                                                        sampling_rate, lower_edge_hertz,
                                                                        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # 8. normalisation
    means = tf.math.reduce_mean(log_mel_spectrograms, 1, keepdims=True)
    stddevs = tf.math.reduce_std(log_mel_spectrograms, 1, keepdims=True)
    log_mel_spectrograms = (log_mel_spectrograms - means) / (stddevs + 1e-10)

    return log_mel_spectrograms


def encode_single_file(wav_file):
    """
    Describes the transformation that we apply to each element of our dataset
    """
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file
    file = tf.io.read_file(wav_file)
    # 2. Decode the wav file
    audio, sampling_rate = tf.audio.decode_wav(file)
    # Check for stereo audio (2 channels) and convert to mono if necessary
    if audio.shape[-1] == 2:
        # Convert stereo to mono by averaging both channels
        audio = tf.reduce_mean(audio, axis=-1, keepdims=True)

    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    stfts = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(stfts)  # get absolute value of complex number
    spectrogram = tf.math.pow(spectrogram, 2)  # get power

    # 6. mel spectrogram
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins,
                                                                        sampling_rate, lower_edge_hertz,
                                                                        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # 8. normalisation
    means = tf.math.reduce_mean(log_mel_spectrograms, 1, keepdims=True)
    stddevs = tf.math.reduce_std(log_mel_spectrograms, 1, keepdims=True)
    log_mel_spectrograms = (log_mel_spectrograms - means) / (stddevs + 1e-10)

    return log_mel_spectrograms


def pad_tensor(tensor, desired_size):
    # Calculate the padding size
    padding_size = max(desired_size - tf.shape(tensor)[-1], 0)
    # Pad the tensor to the desired size
    padded_tensor = tf.pad(tensor, [[0, 0], [0, padding_size]])
    return padded_tensor


def convert_to_tensor_from_frame(audio):
    encoded = encode_single_buffer(audio)
    padded_features = pad_tensor(encoded, 16)
    padded_features = np.expand_dims(padded_features, axis=0)
    tensor_input = tf.convert_to_tensor(padded_features, dtype=tf.float32)
    return tensor_input


def convert_to_tensor_from_file(audio_file):
    encoded = encode_single_file(audio_file)
    padded_features = pad_tensor(encoded, 16)
    padded_features = np.expand_dims(padded_features, axis=0)
    tensor_input = tf.convert_to_tensor(padded_features, dtype=tf.float32)
    return tensor_input


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


def stt_from_frames(audio):
    input_tensor = convert_to_tensor_from_frame(audio)
    inference_time = 0.0

    # Run Deepspeech
    logging.debug('Running inference...')
    inference_start = timer()

    output = model.predict(input_tensor)
    output = decode_batch_predictions(output)

    inference_end = timer() - inference_start
    inference_time += inference_end
    logging.debug('Inference took %0.3fs.' % inference_end)

    return [output, inference_time]


def stt_from_file(audio):
    input_tensor = convert_to_tensor_from_file(audio)
    inference_time = 0.0

    # Run Deepspeech
    logging.debug('Running inference...')
    inference_start = timer()

    output = model.predict(input_tensor)
    output = decode_batch_predictions(output)

    inference_end = timer() - inference_start
    inference_time += inference_end
    logging.debug('Inference took %0.3fs.' % inference_end)

    return [output, inference_time]


lowercase_chars = string.ascii_lowercase
accented_chars = "àáãạảăắằẳẵặâấầẩẫậèéẹẻẽêềếểễệđìíĩỉịòóõọỏôốồổỗộơớờởỡợùúũụủưứừửữựỳỵỷỹý"
punctuation_chars = string.punctuation
final_chars = lowercase_chars + accented_chars + punctuation_chars + " "
characters = [x for x in final_chars]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
list = char_to_num.get_vocabulary()
char2num =  " ".join(list)
char_to_num_ = char2num.encode("utf-8")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

print(f"The vocabulary is: {char_to_num_} "
      f"(size ={char_to_num.vocabulary_size()})")


def vad_segment_generator(wavFile, aggressiveness):
    logging.debug("Caught the wav file @: %s" % (wavFile))
    audio, sample_rate, audio_length = wav_split.read_wave(wavFile)
    assert sample_rate == 16000, "Only 16000Hz input WAV files are supported for now!"
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = wav_split.frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = wav_split.vad_collector(sample_rate, 30, 300, vad, frames)

    return segments, sample_rate, audio_length
