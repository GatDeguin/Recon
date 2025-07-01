# Recon

A simple demo for Argentinian Sign Language (LSA) action recognition. The repository includes a trained TensorFlow model and utilities for running real-time detection using MediaPipe.

## HTML demo

Before opening `index.html` you need a TensorFlow.js version of the model.  
Convert `action.h5` with:

```bash
pip install tensorflowjs
tensorflowjs_converter --input_format=keras action.h5 model
```

Copy the generated `model.json` and weight files from the `model` directory next to
`index.html`. Then open the page in a modern browser. It will load TensorFlow.js
and MediaPipe Holistic from CDNs, request webcam access and predict actions using
`model.json`.

## Python script

Install the Python dependencies (for example `tensorflow`, `mediapipe` and `opencv-python`) and run:

```bash
python action_detection3.py --model action.h5 --video 0
```

Replace `0` with a path to a video file or another webcam index. Add `--output out.mp4` to save the annotated video.

## Dataset

The repository also contains a copy of the [LSA-T](LSA-T) continuous Argentinian Sign Language dataset. See `LSA-T/README.md` for download links and details about the data.

