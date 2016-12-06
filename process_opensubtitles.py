from os import path, makedirs, walk
from collections import defaultdict
from operator import itemgetter
from random import shuffle
from sys import argv
import re
from codecs import getwriter
import logging

from xml.sax.handler import ContentHandler
from xml.sax import SAXException, make_parser

from pandas import DataFrame

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

WORD_RE = re.compile('\w+')
MAX_SENTENCE_LENGTH = 60
DIGIT_RE = re.compile('^\-?\d+((\.|\,)\d+)?$')
TESTSET_RATIO = 0.1


class OpenSubtitlesHandler(ContentHandler):
    def initialize(self):
        self.sentences = []
        self.within_word = False

    def startDocument(self):
        self.initialize()

    def startElement(self, tag, attrs):
        if tag == 's':
            self.sentences.append([])
        if tag == 'w':
            self.within_word = True

    def endElement(self, tag):
        if tag == 'w':
            self.within_word = False

    def characters(self, content):
        if self.within_word:
            self.sentences[-1].append(content)


def process_sentences(in_sentences):
    sentences_filtered = []
    for sentence in in_sentences:
        sentence_filtered = [
            token.lower()
            for token in sentence
            if WORD_RE.match(token)
        ]
        sentence_filtered = [
            re.sub(DIGIT_RE, '<DIGIT>', token)
            for token in sentence_filtered
        ]
        if len(sentence_filtered):
            sentences_filtered.append(sentence_filtered)
    return sentences_filtered


def parse_corpus(text_root):
    documents_number = sum([
        len(files)
        for root, dirs, files in walk(text_root)
    ])
    handler = OpenSubtitlesHandler()
    parser = make_parser()
    parser.setContentHandler(handler)

    parsed_corpus = {}
    index = 0
    for root, dirs, files in walk(text_root):
        for filename in files:
            if not filename.endswith('xml'):
                continue
            index += 1
            logger.info('Processing file {} of {}'.format(index, documents_number))
            full_filename = path.join(root, filename)
            parser.parse(full_filename)
            parsed_corpus[full_filename] = process_sentences(handler.sentences)
    return parsed_corpus


def group_texts_into_qa_pairs(in_documents):
    qa_data = []
    for doc in in_documents:
        for question, answer in zip(doc, doc[1:]):
            if len(question) < MAX_SENTENCE_LENGTH and len(answer) < MAX_SENTENCE_LENGTH:
                qa_data.append((question, answer))
    return qa_data


def save_csv(in_qa_pairs, in_result_filename):
    dataframe = DataFrame(in_qa_pairs)
    dataframe.to_csv(
        in_result_filename,
        sep=';',
        header=False,
        index=False,
        encoding='utf-8'
    )


def save_encoder_decoder_files(in_qa_pairs, in_output_folder, in_dataset_name):
    enc_file = path.join(in_output_folder, in_dataset_name + '.enc')
    dec_file = path.join(in_output_folder, in_dataset_name + '.dec')

    with getwriter('utf-8')(open(enc_file, 'w')) as enc_out:
        with getwriter('utf-8')(open(dec_file, 'w')) as dec_out:
            for question, answer in in_qa_pairs:
                print >>enc_out, ' '.join(question)
                print >>dec_out, ' '.join(answer)


def save_easy_seq2seq(in_qa_pairs, in_result_folder):
    shuffle(in_qa_pairs)
    testset_size = int(TESTSET_RATIO * len(in_qa_pairs))
    train_set, test_set = (
        in_qa_pairs[:-testset_size],
        in_qa_pairs[-testset_size:]
    )
    save_encoder_decoder_files(train_set, in_result_folder, 'train')
    save_encoder_decoder_files(test_set, in_result_folder, 'test')


def main(in_opensubs_root, in_result_folder):
    if not path.exists(in_result_folder):
        makedirs(in_result_folder)
    parsed_texts = parse_corpus(in_opensubs_root)
    qa_pairs = group_texts_into_qa_pairs(parsed_texts.values())
    # save_csv(qa_pairs_joined, in_result_file)
    save_easy_seq2seq(qa_pairs, in_result_folder)


if __name__ == '__main__':
    if len(argv) < 3:
        print 'Usage: {} <OpenSubtitles corpus root> <result folder>'.format(
            argv[0]
        )
        exit()
    opensubs_root, result_folder = argv[1:3]
    main(opensubs_root, result_folder)

