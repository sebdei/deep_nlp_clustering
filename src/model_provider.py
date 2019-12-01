import zipfile
import numpy as np
import os
from urllib.request import urlretrieve
import gensim
import gzip
import shutil
from gensim.models.wrappers import FastText

GLOVE_MODEL_BASE_DIR = 'models'
GLOVE_MODEL_TXT_NAME = 'glove.6B.200d.txt'
GLOVE_MODEL_ZIP_NAME = 'glove.6B.zip'
GLOVE_MODEL_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'

FASTTEXT_MODEL_BASE_DIR = 'models'
FASTTEXT_MODEL_BIN_NAME = 'cc.en.300.bin'
FASTTEXT_MODEL_ZIP_NAME = 'cc.en.300.bin.gz'
FASTTEXT_MODEL_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'


def assure_glove_model_exists():
    if not os.path.exists(GLOVE_MODEL_BASE_DIR):
        os.makedirs(GLOVE_MODEL_BASE_DIR)

    if not (os.path.exists(os.path.join(GLOVE_MODEL_BASE_DIR, GLOVE_MODEL_TXT_NAME))):
        if (not os.path.exists(os.path.join(GLOVE_MODEL_BASE_DIR, GLOVE_MODEL_ZIP_NAME))):
            print('downloading GLOVE model (>800MB) ...')
            urlretrieve(GLOVE_MODEL_URL, os.path.join(GLOVE_MODEL_BASE_DIR, GLOVE_MODEL_ZIP_NAME))

        zip = zipfile.ZipFile(os.path.join(GLOVE_MODEL_BASE_DIR, GLOVE_MODEL_ZIP_NAME), 'r')
        zip.extract(GLOVE_MODEL_TXT_NAME, GLOVE_MODEL_BASE_DIR)


def provide_glove_model():
    assure_glove_model_exists()

    print('providing word embedding model ...')

    file = open(os.path.join(GLOVE_MODEL_BASE_DIR, GLOVE_MODEL_TXT_NAME), 'r')
    model = {}

    for line in file:
        split_line = line.split()
        word = split_line[0]
        model[word] = np.array([float(val) for val in split_line[1:]])

    print('model successfully loaded')

    return model


def assure_fasttext_model_exists():
    zipfile = os.path.join(FASTTEXT_MODEL_BASE_DIR, FASTTEXT_MODEL_ZIP_NAME)
    binfile = os.path.join(FASTTEXT_MODEL_BASE_DIR, FASTTEXT_MODEL_BIN_NAME)

    if not os.path.exists(FASTTEXT_MODEL_BASE_DIR):
        os.makedirs(FASTTEXT_MODEL_BASE_DIR)

    if not (os.path.exists(os.path.join(FASTTEXT_MODEL_BASE_DIR, FASTTEXT_MODEL_BIN_NAME))):
        if (not os.path.exists(zipfile)):
            print('downloading FastText model (>4GB) ...')
            urlretrieve(FASTTEXT_MODEL_URL,  zipfile)
        with gzip.open(zipfile, 'rb') as f_in:
            with open(binfile, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def provide_fasttext_model():
    assure_fasttext_model_exists()
    model = FastText.load_fasttext_format(os.path.join(FASTTEXT_MODEL_BASE_DIR, FASTTEXT_MODEL_BIN_NAME))

    return model.wv
