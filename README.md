# KMA STUDENT's ASR
# DEMO
### HOW TO USE:
- Step 1: put model into **_model_** folder, replace model_path in model.py
- Step 2: install requirements with **Conda**
- Step 3: put the input audio file into **_audio_sample_** folder
- Step 4: Run app.py
  - Default, it's big file (> 15s), app will divide input audio into segments by silence. After that, run **prediction** on each segment.
  ``` python
    python app.py -f audio_file_name # big file mode
    ## Example
    # python app.py -f truyenngan.mp3
  ```
  - If you want to run in small mode (audio file < 15s):
  ``` python
    python app.py -f file_name --small # small file mode
    ## Example
    #  python app.py -f VIVOSDEV01_R117.wav --small
  ```
- -- 