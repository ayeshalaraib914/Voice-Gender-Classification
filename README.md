# Voice Gender Classification

A deep learning model that classifies a speaker's gender from audio input. Built and trained in Google Colab using a CNN trained on Mel Spectrograms. Supports both dataset-based training and live microphone prediction.

## How it works

Audio files are loaded and converted into Mel Spectrograms, which represent frequency content over time as a 2D image. These spectrograms are then fed into a CNN that learns to distinguish between male and female voice patterns. For live prediction, audio is recorded directly in the browser via JavaScript, converted to WAV, and passed through the same preprocessing pipeline before prediction.

## Requirements

Run in Google Colab. Dataset must be stored in Google Drive under the following structure:

```
MyDrive/
  data/
    train/
      female/
      male/
    valid/
      female/
      male/
    test/
```

All audio files should be in `.wav` format.

## Usage

**Training**

Mount Google Drive, run the preprocessing and training cells. The model trains for 15 epochs with a batch size of 32. Weights are saved both locally in Colab and to Google Drive at `data/voice_gender_model.weights.h5`, and automatically downloaded to your computer.

**Live Prediction**

Load the saved weights, then run `record_live_voice()`. It records 3.5 seconds from your microphone, processes the audio, and prints the predicted gender with confidence percentage.

## Model Architecture

- Input: (128, 174, 1) — Mel Spectrogram as grayscale image
- Conv2D (32 filters) → MaxPooling → Dropout 0.2
- Conv2D (64 filters) → MaxPooling → Dropout 0.2
- Flatten → Dense (64) → Dropout 0.3
- Dense (1, sigmoid) — binary output: 0 = female, 1 = male

## Training Data

| Split | Female | Male |
|-------|--------|------|
| Train | 5538 | 10179 |
| Valid | 149 | 119 |

## Notes

- Designed to run entirely in Google Colab with GPU acceleration (T4)
- Live recording uses the browser's MediaRecorder API via injected JavaScript
- Audio is resampled to 22050 Hz and clipped to 3 seconds during preprocessing
