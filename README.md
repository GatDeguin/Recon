# Recon

A simple demo for Argentinian Sign Language (LSA) action recognition. The repository includes a trained TensorFlow model and utilities for running real-time detection using MediaPipe.

## HTML demo

Open `index.html` in a modern web browser. The page loads TensorFlow.js and MediaPipe Holistic from CDNs, requests webcam access and predicts actions using `model.json`.

## Python script

Install the Python dependencies (for example `tensorflow`, `mediapipe` and `opencv-python`) and run:

```bash
python action_detection3.py --model action.h5 --video 0
```

Replace `0` with a path to a video file or another webcam index. Add `--output out.mp4` to save the annotated video.
