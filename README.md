# Sign Language Segmentation Training

This repository holds the code to performing `BIO` tagging for sign language segmentation.
This is a pre-processing step for sign language transcription.

## Setup

Install with pip, preferably inside a Python 3 virtual environment:

```bash
pip install git+https://github.com/bricksdont/sign-segmentation
```

This will install two entry points to the code, `sign-language-segmentation-train` and
`sign-language-segmentation-create-tfrecord`.


## Creating a tfrecord file

```bash
sign-language-segmentation-create-tfrecord -h
```

## Training a model

```bash
sign-language-segmentation-train -h
```

## Running tests

To run tests with coverage, execute

```bash
pytest
```

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
