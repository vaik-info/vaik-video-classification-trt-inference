# vaik-video-classification-trt-inference
Inference by video classification Tensor RT model


## Install

```shell
pip install git+https://github.com/vaik-info/vaik-video-classification-trt-inference.git
```

## Usage
### Example

```python
import os
import numpy as np
import imageio

from vaik_video_classification_trt_inference.trt_model import TrtModel

input_saved_model_path = os.path.expanduser('/tmp/model.trt')
classes = ("ApplyEyeMakeup", "Biking", "CleanAndJerk", "FrisbeeCatch", "HorseRace", "LongJump", "PlayingDhol", "Punch", "Skiing", "TaiChi")
video_path = os.path.expanduser('/tmp/avi/ApplyEyeMakeup_142.avi')

model = TrtModel(input_saved_model_path, classes)
video = imageio.get_reader(video_path,  'ffmpeg')
frames = [np.array(frame, dtype=np.uint8) for frame in video]
output, raw_pred = model.inference(frames)
```

### Output

- output

```
[
  {
    'score': [
      6.333147048950195,
      5.063453674316406,
      3.904092311859131,
      ・・・
      ],
    'label': [
      'CliffDiving',
      'Basketball',
      'Biking',
      ・・・
    ],
    'start_frame': 0,
    'end_frame': 16
  },
  ・・・
```

- raw_pred

```
[[-2.7884161e-01 -3.4559977e+00  2.3612885e+00  1.8890795e+00
   8.8122386e-01  3.9267153e-01 -3.6665955e+00  5.0631857e+00
  -4.7183666e+00 -1.6739460e+00  3.9042349e+00 -4.5220504e+00
   2.6965277e+00 -6.0049925e+00  2.1489031e+00 -3.6779819e+00
   ・・・
```

----------

## Dockerfile for dev

```dockerfile
FROM nvcr.io/nvidia/tensorrt:22.10-py3
RUN apt update && apt install ffmpeg -y
RUN pip install -U pip
RUN pip install imageio imageio-ffmpeg
```