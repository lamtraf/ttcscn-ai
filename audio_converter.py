import os

from pydub import AudioSegment


def convert_to_wav(source_path, target_path):
    # Load the source file
    audio = AudioSegment.from_file(source_path)

    # Export as WAV
    audio.export(target_path, format="wav")
    print(f"Converted {source_path} to {target_path}")


def check_and_convert_to_wav(file_path):
    # Check if the file is already a WAV file
    if not file_path.lower().endswith('.wav'):
        # Define a new file path with the WAV extension
        base = os.path.splitext(file_path)[0]
        new_file_path = base + ".wav"
        print("new file path: ", new_file_path)

        # Convert to WAV
        convert_to_wav(file_path, new_file_path)

        return new_file_path
    else:
        print(f"{file_path} is already in WAV format")
        return file_path
