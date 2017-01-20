import pandas as pd


def csv_to_easy_seq2seq(in_csv_file, out_enc, out_dec):
    csv_dataframe = pd.read_csv(in_csv_file, sep=';', names=['q', 'a'])
    for index, row in csv_dataframe.iterrows():
        print >>out_enc, row['q']
        print >> out_dec, row['a']


def main(in_src_folder, in_result_folder):
    if not os.path.exists(in_result_folder):
        os.makedirs(in_result_folder)
    tmp_enc_filename = os.path.join(in_result_folder, 'tmp.enc')
    tmp_dec_filename = os.path.join(in_result_folder, 'tmp.dec')
    with open(tmp_enc_filename) as tmp_enc:
        with open(tmp_dec_filename) as tmp_dec:
            for root, dirs, files in os.walk(in_src_folder):
                for filename in files:
                    full_filename = os.path.join(root, filename)
                    csv_to_easy_seq2seq(full_filename, tmp_enc, tmp_dec)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Usage: {} <source folder> <result folder>'.format(
            os.path.basename(__file__)
        )
        exit()

    src_folder, result_folder = sys.argv[1:3]
    main(src_folder, result_folder)