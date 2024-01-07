# Import the argparse library
import argparse
import logging
import os
import textwrap
import wave

import audio_converter
import model
import split

sample_rate = 16000
buffer_size = 2048 * 10
num_channels = 1
sample_width = 2  # Sample width in bytes


def predict_small_file(file_name):
    path = './audio_sample/' + file_name

    print(file_name)
    output_file_path = file_name.rstrip(".wav") + ".txt"
    output_file_path = 'output/' + output_file_path

    convertedFile = audio_converter.check_and_convert_to_wav(path)
    output, inference_time = model.stt_from_file(convertedFile)

    print("output:" + output[0] + "\ninference_time: " + str(inference_time))

    combined_transcription = textwrap.fill(output[0], width=70)

    combined_transcription += '\nCompleted in ' + str(inference_time)
    with open(output_file_path, 'w+', encoding="utf-8") as file:
        file.write(combined_transcription)

    print("Prediction Ended!")


def predict_big_file(path):
    path = './audio_sample/' + path

    output_file_path = path.rstrip(".wav") + ".txt"
    output_file_path = 'output/' + output_file_path

    convertedFile = audio_converter.check_and_convert_to_wav(path)

    # segments = split.split_audio_on_silence(convertedFile)
    segments = split.segment_audio_on_silence(convertedFile)
    transcriptions = []
    predict_times = 0
    chunks = 0
    for i, segment in enumerate(segments):
        print("Processing chunk ", i)
        chunks += 1
        # Run deepspeech on the chunk that just completed VAD
        logging.debug("Processing chunk %002d" % (i,))

        # Convert the NumPy array to bytes
        audio_bytes = segment.tobytes()

        # Create a wave file and set the parameters
        with wave.open('temp1.wav', 'wb') as audio_file:
            audio_file.setnchannels(num_channels)
            audio_file.setsampwidth(sample_width)
            audio_file.setframerate(sample_rate)

            # Write audio data
            audio_file.writeframes(audio_bytes)

        # Perform speech-to-text
        output, inference_time = model.stt_from_file('temp.wav')

        print("output:" + output[0] + "\ninference_time: " + str(inference_time))
        transcriptions.extend(output[0])
        predict_times += inference_time

    combined_transcription = ''.join(transcriptions)
    combined_transcription = textwrap.fill(combined_transcription, width=70)

    combined_transcription += '\nCompleted in ' + str(predict_times) + "\nChunk: " + str(chunks)
    with open(output_file_path, 'w', encoding="utf-8") as file:
        file.write(combined_transcription)

    print("Prediction Ended!")

def main_predict_demo(file_path, smalll=False):
    if (smalll):
        predict_small_file(file_path)
    else:
        predict_big_file(file_path)


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Sample Command Line Program")

    # argument file
    parser.add_argument('-f', '--file', default='VIVOSDEV01_R117.wav', help='Audio input file')

    # # argument model path
    # parser.add_argument('--model', default='stt v1.h5', help='Model path')

    # argument model path
    parser.add_argument('--small', action='store_true', help='Big file')

    # Parse the arguments
    args = parser.parse_args()

    main_predict_demo(args.file, args.small)

if __name__ == "__main__":
    main()