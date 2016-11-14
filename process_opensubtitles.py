from os import path, makedirs, walk
from collections import defaultdict
from operator import itemgetter
from random import shuffle
from sys import argv
from re import compile

from xml.sax.handler import ContentHandler
from xml.sax import SAXException, make_parser

from pandas import DataFrame

WORD_RE = compile('\w+')


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
        if len(sentence_filtered):
            sentences_filtered.append(sentence_filtered)
    return sentences_filtered


def parse_corpus(text_root):
    handler = OpenSubtitlesHandler()
    parser = make_parser()
    parser.setContentHandler(handler)

    parsed_corpus = {}
    for root, dirs, files in walk(text_root):
        for filename in files:
            if not filename.endswith('xml'):
                continue
            full_filename = path.join(root, filename)
            parser.parse(full_filename)
            parsed_corpus[full_filename] = process_sentences(handler.sentences)
    return parsed_corpus


def make_vocabulary(in_parsed_docs, limit):
    wordcount = defaultdict(lambda: 0)
    for doc in in_parsed_docs:
        for sentence in doc:
            for word in sentence:
                wordcount[word.lower() if word != 'I' else word] += 1
    wordcount_sorted = sorted(
        wordcount.items(),
        key=itemgetter(1),
        reverse=True
    )
    result = set(map(itemgetter(0), wordcount_sorted[:limit]))
    return result


def preprocess_text(in_parsed_docs):
    docs = in_parsed_docs.values()
    vocabulary = make_vocabulary(docs)
    filtered_get = lambda word: word if word in vocabulary else UNK
    result = []
    for content in docs:
        processed_content = []
        for sentence in content:
            processed_sentence = [word.lower() if word != 'I' else word for word in sentence]
            filtered_sentence = [filtered_get(word) for word in processed_sentence]
            processed_content.append(filtered_sentence)
        result.append(processed_content)
    return result


def group_texts_into_qa_pairs(in_documents):
    qa_data = []
    for doc in in_documents:
        for question, answer in zip(doc[::2], doc[1::2]):
            qa_data.append((question, answer))

    shuffle(qa_data)
    return qa_data


def prepare_seq2seq_files(in_processed_docs, in_result_path):
    if not path.exists(in_result_path):
        makedirs(in_result_path)

    qa_data = group_texts_into_qa_pairs(in_processed_docs)
    
    trainset_size = int((1 - TESTSET_SIZE_RATIO) * len(qa_data))
    qa_train, qa_test = qa_data[:trainset_size], qa_data[trainset_size:]

    # open files
    with \
        open(path.join(in_result_path, 'train.enc'), 'w') as train_enc, \
        open(path.join(in_result_path, 'train.dec'), 'w') as train_dec, \
        open(path.join(in_result_path, 'test.enc'), 'w') as test_enc, \
        open(path.join(in_result_path, 'test.dec'), 'w') as test_dec:

        for question_train, answer_train in qa_train:
            print >>train_enc, ' '.join(question_train).encode('utf-8')
            print >>train_dec, ' '.join(answer_train).encode('utf-8')
        for question_test, answer_test in qa_test:
            print >>test_enc, ' '.join(question_test).encode('utf-8')
            print >>test_dec, ' '.join(answer_test).encode('utf-8')


def save_csv(in_qa_pairs, in_result_filename):
    dataframe = DataFrame(in_qa_pairs)
    dataframe.to_csv(
        in_result_filename,
        sep=';',
        header=False,
        index=False,
        encoding='utf-8'
    )


def main(in_opensubs_root, in_result_file):
    parsed_texts = parse_corpus(in_opensubs_root)
    qa_pairs = group_texts_into_qa_pairs(parsed_texts.values())
    qa_pairs_joined = [
        (' '.join(question), ' '.join(answer))
        for question, answer in qa_pairs
    ]
    save_csv(qa_pairs_joined, in_result_file)

 
if __name__ == '__main__':
    if len(argv) < 3:
        print 'Usage: opensubtitles_processing.py <OpenSubtitles corpus root> <result file>' 
        exit()
    opensubs_root, result_file = argv[1:3]
    main(opensubs_root, result_file)

