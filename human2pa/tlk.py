"""
Download and unpack Toloka chat dataset
"""
import csv
import logging
import shutil
import urllib.request
import zipfile

from pathlib import Path
from tempfile import mkdtemp

from bs4 import BeautifulSoup


_tlk_url = 'https://tlk.s3.yandex.net/dataset/TlkPersonaChatRus.zip'
_LOGGER = logging.getLogger(__name__)


def tlk_dl(path=Path('.')):
    result = path / 'tlk.zip'
    _LOGGER.info("Downloading %s to %s", _tlk_url, result)
    urllib.request.urlretrieve(_tlk_url, result)
    return result


def tlk_unzip(tlk_path, path=Path('.')):
    result = path / 'tlk.tsv'
    dialogues_file = 'TlkPersonaChatRus/dialogues.tsv'
    with zipfile.ZipFile(tlk_path, 'r') as zip_ref, open(result, "wb") as out:
        _LOGGER.info("Extracting %s from %s to %s", dialogues_file, tlk_path, result)
        out.write(zip_ref.read(dialogues_file))
    return result


def _parse_dialogue(dialogue):
    d = BeautifulSoup(dialogue, 'html.parser')
    for c in d.find_all('span'):
        yield c['class'][0], c.text.split(':', 2)[1].strip()


def tlk_process(tlk_path, path=Path('.')):
    result = path / 'tlk.txt'
    _LOGGER.info("Processing %s to %s", tlk_path, result)
    with open(tlk_path) as tsv, open(result, 'w') as out:
        tsv.readline()
        tsvin = csv.reader(tsv, delimiter='\t')
        for profile1, profile2, dialogue in tsvin:
            for participant, phrase in _parse_dialogue(dialogue):
                print(phrase, file=out)
    return result


def get_data(path=Path('.'), keep_intermediates=False):
    if keep_intermediates:
        intermediate_path = path
    else:
        intermediate_path = Path(mkdtemp())
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    corpus = path / 'tlk.txt'
    if not corpus.exists():
        raw_data = path / 'tlk.tsv'
        if not raw_data.exists():
            zip_file = path / 'tlk.zip'
            if not zip_file.exists():
                zip_file = tlk_dl(intermediate_path)
            raw_data = tlk_unzip(zip_file, intermediate_path)
        tlk_process(raw_data, path)
    if not keep_intermediates:
        shutil.rmtree(intermediate_path)
    return corpus
