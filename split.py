import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent


def split_audio_on_silence(audio_path, target_sample_rate=16000, silence_thresh=-50, min_silence_len=1000,
                           keep_silence=500):
    """
    Splits an audio file into segments based on silence.

    :param audio_path: Path to the audio file.
    :param silence_thresh: Silence threshold in dB. Lower values mean more silence will be detected.
    :param min_silence_len: Minimum length of silence in milliseconds to consider as a split.
    :param keep_silence: Amount of silence to leave at the beginning and end of each segment.
    :return: List of audio segments.
    """
    sound = AudioSegment.from_file(audio_path)
    sound = sound.set_frame_rate(target_sample_rate)

    # Split on silence
    segments = split_on_silence(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh,
                                keep_silence=keep_silence)
    return segments


def segment_audio_on_silence(file_path, min_silence_len=1000, silence_thresh=-40, max_segment_duration=3000):
    """
    Segment an audio file based on silence and maximum segment duration.

    Parameters:
    file_path (str): Path to the audio file.
    min_silence_len (int): Minimum length of silence to be used for splitting (in ms).
    silence_thresh (int): Silence threshold (in dB).
    max_segment_duration (int): Maximum duration of a segment (in ms).

    Returns:
    List of np.ndarray: Each element is a numpy array representing a segment.
    """
    # Load the audio file
    audio = AudioSegment.from_file(file_path)

    # Detect non-silent chunks
    nonsilent_chunks = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    # Initialize list to hold final segments
    final_segments = []

    for start_i, end_i in nonsilent_chunks:
        segment = audio[start_i:end_i]
        # Further split segment if it's longer than the max duration
        while len(segment) > max_segment_duration:
            # Split at the max duration point
            split_segment, segment = segment[:max_segment_duration], segment[max_segment_duration:]
            final_segments.append(np.array(split_segment.get_array_of_samples()))
        final_segments.append(np.array(segment.get_array_of_samples()))

    return final_segments
