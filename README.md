# Sign Language Segmentation Training

This repository holds the code to performing `BIO` tagging for sign language segmentation.
This is a pre-processing step for sign language transcription.

As predicted segments are indicative of signing, this model can be used for Sign Language Detection, 
however, detection is a simpler task (`IO` vs `BIO`), and you better use [sign-language-processing/detection-train](https://github.com/sign-language-processing/detection-train).

## Models

This repository includes pre-trained models for both [python](models/py/) and [javascript](models/js/), 
and for both a realtime (unidirectional) model and an offline (bidirectional) model.

## Usage

You can use the included models to perform inference or fine-tuning.
 
To load a model in python, use
`tensorflow.python.keras.models.load_model('models/py/model.h5')`.

To load a model in the browser, use `tf.loadLayersModel('models/js/model.json')`
from [tfjs](https://github.com/tensorflow/tfjs).

You can use the [train.py](train.py) script to train the model from scratch
using a `tfrecord` dataset file.

```bash
python -m train --device="/GPU:0"
```

## Dataset

The provided models were trained on the
[Public DGS Corpus](https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start-name_en.html) 
via the [sign-language-datasets](https://github.com/sign-language-processing/datasets) library.

This is a German Sign Language corpus, and there is no evaluation of how it works for other signed languages.

## Citations

```bibtex
# Datasets library
@misc{moryossef2021datasets, 
    title={Sign Language Datasets},
    author={Moryossef, Amit},
    howpublished={\url{https://github.com/sign-language-processing/datasets}},
    year={2021}
}

# The Public DGS Corpus
@inproceedings{hanke2020extending,
  title={{E}xtending the {P}ublic {DGS} {C}orpus in Size and Depth},
  author={Hanke, Thomas and Schulder, Marc and Konrad, Reiner and Jahn, Elena},
  booktitle={Proceedings of the LREC2020 9th Workshop on the Representation and Processing of Sign Languages: Sign Language Resources in the Service of the Language Community, Technological Challenges and Application Perspectives},
  pages={75--82},
  year={2020}
}
```
