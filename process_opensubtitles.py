from os import path, makedirs, walk
from collections import defaultdict
from operator import itemgetter
from random import shuffle, choice
from sys import argv
import re
from codecs import getwriter
import logging
from argparse import ArgumentParser
import string

from xml.sax.handler import ContentHandler
from xml.sax import SAXException, make_parser

from pandas import DataFrame

from task_list import tasks, execute_tasks, add_task

logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel('INFO')

WORD_RE = re.compile('\w+')
MAX_QUESTION_LENGTH = 60
MAX_ANSWER_LENGTH = 60
JOBS_NUMBER = 32
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


def process_document_callback(in_params):
    input_file, output_file = in_params
    logger.info('Processing file {}'.format(input_file))
    sentences = parse_document(input_file)
    sentences_processed = process_sentences(sentences)
    qa_pairs = sentences_to_qa_pairs(sentences_processed)
    save_csv(qa_pairs, output_file)


def parse_document(in_filename):
    handler = OpenSubtitlesHandler()
    parser = make_parser()
    parser.setContentHandler(handler)

    parser.parse(in_filename)
    return handler.sentences


def sentences_to_qa_pairs(in_sentences):
    qa_data = []
    for question, answer in zip(in_sentences, in_sentences[1:]):
        if (
            len(question) < MAX_QUESTION_LENGTH and
            len(answer) < MAX_ANSWER_LENGTH
        ):
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


def collect_tasks(in_src_root, in_dst_root):
    for root, dirs, files in walk(in_src_root):
        for filename in files:
            full_filename = path.join(root, filename)
            result_filename = \
                path.join(in_dst_root, filename) + \
                ''.join([choice(string.ascii_letters) for _ in xrange(16)])
            add_task((full_filename, result_filename))


def build_argument_parser():
    parser = ArgumentParser()
    parser.add_argument('corpus_root', help='OpenSubtitles corpus root')
    parser.add_argument('output_folder', help='Output folder')

    parser.add_argument(
        '--max_question_length',
        type=int,
        default=MAX_QUESTION_LENGTH
    )
    parser.add_argument(
        '--max_answer_length',
        type=int,
        default=MAX_ANSWER_LENGTH
    )
    parser.add_argument('--jobs', type=int, default=JOBS_NUMBER)
    return parser


def main(
    in_opensubs_root,
    in_result_folder,
    in_callback=process_document_callback
):
    if not path.exists(in_result_folder):
        makedirs(in_result_folder)
    collect_tasks(in_opensubs_root, in_result_folder)
    logger.info('got {} tasks'.format(len(tasks)))
    if 1 < JOBS_NUMBER:
        retcodes = execute_tasks(in_callback, JOBS_NUMBER)
    else:
        retcodes = [in_callback(task) for task in tasks]


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    MAX_QUESTION_LENGTH = args.max_question_length
    MAX_ANSWER_LENGTH = args.max_answer_length
    JOBS_NUMBER = args.jobs
    main(args.corpus_root, args.output_folder)

