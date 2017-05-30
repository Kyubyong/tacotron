# A (Heavily Documented) TensorFlow Implementation of Tacotron: A Fully End-to-End Text-To-Speech Synthesis Model

## **Warning**

As of May 17, 2017, this is still a first draft. You can run it following the steps below, but probably you should get poor results. I'll be working on debugging this weekend. (**Code reviews and/or contributions are more than welcome!**)

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow >= 1.1
  * librosa

## Data
Since the [original paper](https://arxiv.org/abs/1703.10135) was based on their internal data, I use a freely available one, instead.

[The World English Bible](https://en.wikipedia.org/wiki/World_English_Bible) is a public domain update of the American Standard Version of 1901 into modern English.
Its text and audio recordings are freely avaiable [here](http://www.audiotreasure.com/webindex.htm). 
Unfortunately, however, each of the audio files matches a chapter, not a verse, so is too long in most cases. I sliced them by verse manually. 
You can get them on [my dropbox](https://dl.dropboxusercontent.com/u/42868014/WEB.zip)


## Work Flow
  * STEP 1. Adjust hyper parameters in `hyperparams.py` if necessary.
  * STEP 2. Download [the data](https://dl.dropboxusercontent.com/u/42868014/WEB.zip) and extract it. 
  * STEP 3. Run `train.py`.
  * STEP 4. Run `eval.py` to get samples.
