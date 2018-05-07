## Contents

<!-- MarkdownTOC -->

- [Pre-Requirements](#pre-requirements)
- [Installation](#installation)
  - [Manually](#manually)
  - [Docker](#docker)
  - [Windows](#windows)
- [Usage](#usage)
- [Data](#data)
- [Solution](#solution)
- [Results](#results)
- [Conclusion](#conclusion)

<!-- /MarkdownTOC -->


## Pre-Requirements

- Docker (optional). You may get it [here](https://docs.docker.com/install/)
- Git, Python>=3.6
- Python packages (see requirements.txt)

## Installation

### Manually

- Clone the repository 

    `git clone https://github.com/titardrew/hands-detector.git && cd hands-detector`

- Create virtualenv (optional)

    `python3.6 -m venv .venv && source .venv/bin/activate`

- Install corresponding requirements 

    `pip install -r requirements.txt`

**Note**: For gpu support you should have tensorflow with gpu + cuda + cudnn (see tf docs)

### Docker (Recommended)

#### Please note, that you may want to assign more memory to your docker containter!!!

#### Option 1.

pull the docker container:
    `docker pull titardrew/hand-detection`

#### Option 2.

Build your own docker container:

- Clone the repository `git clone https://github.com/titardrew/hands-detector.git && cd hands-detector`
- Run build command:
    - `docker build -t hand-detection .`

#### Windows

For you, my poor friend, I recommend to use docker, because I'm still not able to solve the problem with 
*.pb graph for inference.

### Usage

#### Docker
  You have to specify volume with data, volume with `optimized_inference_graph.pb` and 
  mount the output directory:
  
  `docker run -it -v path/to/raw/images:/hand-detection/path/you/want \
    --mount type=bind,source=path/to/predicted_images,target=/hand-detection/predicted_images \
    -v `pwd`/fine_tuned_model/frcnn_inc_v2_aug4:/hand-detection/fine_tuned_model/frcnn_inc_v2_aug4/ \
    hand-detection bash`
    
  For example,
  
  `docker run -it -v `pwd`/raw_data/test:/hand-detection/raw_data/test \
    --mount type=bind,source=`pwd`/predicted_images,target=/hand-detection/predicted_images \
    -v `pwd`/fine_tuned_model/frcnn_inc_v2_aug4:/hand-detection/fine_tuned_model/frcnn_inc_v2_aug4/ \
    hand-detection bash`
  
#### Manually
   Just run detect.py:
   
   `python3 detect.py`
   
   You can specify some flags within it (see `python3 detect.py --help`).
  
### Data

For this task I had 50x50 images of hands and noise, and 189x110 test images.
  
![test1](/img/test1.jpg)
![test2](/img/test2.jpg)
![test3](/img/test3.jpg)

To form training dataset, I wrote a simple script that composes 50x50 images.
It pastes them into a randomly chosen background (I'd got a couple using an image editor),
with random rotation in random position so they don't intersect too much.

![train1](/img/train1.jpg)
![train2](/img/train2.jpg)
![train3](/img/train3.jpg)

Than I labled that dataset (train ~ 1080, val ~ 200) and started to work.
However, after several tries of training I to use flip augmentation, because
it is crucual in such problem, but the methods used in Object Detection API were
just flipping the images without label swap (In our case hands are not invariant to
rotation). So I just flipped all the images with label swapping and added them to existing data.
FYI, I also used random black patches augmentation and random brightness adjustment (I wiedly thought
that test images are slightly brighter)

### Solution

For this problem I wanted to build a fast even realtime model, but something went wrong
and my business with SSD with inception_v2, pretrained on COCO was failed. It converged too
slow and the result after 10-15 epochs was very bad. I also wanted to try YOLO, but
thought about this algorithm too late.

So, all in all, I used Faster-RCNN with Inception_v2 and trained it on Tesla P?? GPU (That was
not necessery). I also tried ResNet version, but it was very slow and had not gave me any additional
performance. You can find hyperparameters for this model in .config file

In the end I used Graph Transform library in tensorflow with bazel to
optimize my final graph (Folded batch norms, unused nodes and so on), although I did not
used quantization because it crashed my inference (?) and I was not able to fix that :(

### Results
 
The performance was pretty good, espesially in detection. Here are some examples:

![good1](/img/good1.jpg)
![good2](/img/good2.jpg)
![good3](/img/good3.jpg)
![good4](/img/good4.jpg)
![good5](/img/good5.jpg)

#### Main mistake sources.

Model mistakes were almost all while considering hands that overlay or when one is above another:

![bad1](/img/bad1.jpg)
![bad2](/img/bad2.jpg)
![bad3](/img/bad3.jpg)

I also runned into the problem when network outputs two overlaying bboxes over one hand, but it was
fixed by accepting only the box that has higher score.

### Conclusion

It was quite interesting to solve this problem, especially because of abcense of D4-group invariance
(Flips in particular). I generated a little dataset, built the model that labeled test data:
https://drive.google.com/file/d/14WS6kAPPOQE3zdADdZKoDQpRvxX-JFIr/view?usp=sharing
https://drive.google.com/file/d/1uD_a9GMPTaxunVW849nx0QpVshlZYz4u/view?usp=sharing

However, I had some unimplemented ideas. For example, I'd like to make the model faster using something
less heavy than Faster-RCNN, or adding some extra logic to perform detection less often. I also
had problems with Windows "deployment", and although I solved them using docker, I'd like to add native
support for the programm.
