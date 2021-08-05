"""Test the fruitfly algorithm + classification on test sets
Usage:
  test_models.py --test_path=<filename> --config_path=<filename>
  test_models.py (-h | --help)
  test_models.py --version
Options:
  -h --help                      Show this screen.
  --version                      Show version.
  --test_path=<filename>         Name of file to test (processed by sentencepeice)
  --config_path=<filename>       Name of the json config file containing the best hyper-params we want to test
"""


import json
import pathlib
import joblib
import multiprocessing
import sentencepiece as spm
import numpy as np
from docopt import docopt
from sklearn.feature_extraction.text import CountVectorizer

from hash import read_vocab, hash_dataset
from classify import train_model
from hyperparam_search import read_n_encode_dataset, generate_projs


def evaluate(top_word, KC_size, proj_size, percent_hash, C, num_iter, num_trial):
    def _hash_n_train(model_file):
        # print('hashing dataset')
        hash_train = hash_dataset(dataset_mat=train_set, projection_path=model_file,
                                  percent_hash=percent_hash, top_words=top_word)
        hash_test = hash_dataset(dataset_mat=test_set, projection_path=model_file,
                                 percent_hash=percent_hash, top_words=top_word)
        # print('training and evaluating')
        test_score, model = train_model(m_train=hash_train, classes_train=train_label,
                                        m_val=hash_test, classes_val=test_label,
                                        C=C, num_iter=num_iter)
        return test_score

    print('creating projections')
    model_files = [generate_projs(PN_size, KC_size, proj_size, dataset_name) for _ in range(num_trial)]
    print('training')
    job_list = [max_thread] * (num_trial // max_thread) + [num_trial % max_thread]
    job_list = [i for i in job_list if i != 0]
    score_list = []
    pointer = 0
    for num_job in job_list:
        score_model_list = joblib.Parallel(n_jobs=num_job, prefer="threads")(
            joblib.delayed(_hash_n_train)(model_file) for model_file in model_files[pointer:pointer+num_job])
        score_list += [i[0] for i in score_model_list]
        pointer += num_job

    avg_score = np.mean(score_list)
    print('average score:', avg_score)
    return avg_score


if __name__ == '__main__':
    args = docopt(__doc__, version='Test fruitfly models, ver 0.1')
    config_path = args["--config_path"]
    test_path = args["--test_path"]
    dataset_name = test_path.split('/')[2].split('-')[0]
    print('Dataset name:', dataset_name)
    pathlib.Path(f'./models/projection/{dataset_name}').mkdir(parents=True, exist_ok=True)

    # global variables
    sp = spm.SentencePieceProcessor()
    sp.load('../spmcc.model')
    vocab, reverse_vocab, logprobs = read_vocab()
    PN_size = len(vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=False, token_pattern='[^ ]+')
    print('reading dataset')
    test_set, test_label = read_n_encode_dataset(test_path, vectorizer, logprobs)
    train_set, train_label = read_n_encode_dataset(test_path.replace('test', 'train'), vectorizer, logprobs)
    max_thread = multiprocessing.cpu_count() - 1
    num_trial = 5
    num_iter = 50
    if dataset_name == '20news':
        num_iter = 2000

    # read the config file
    config_list = []
    with open(config_path) as f:
        for line in f:
            config_list.append(json.loads(line))

    # evaluate
    test_score_list = []
    for config in config_list:
        score = evaluate(top_word=config['topword'], KC_size=config['KC_size'],
                         proj_size=config['proj_size'], percent_hash=config['percent_hash'],
                         C=config['C'], num_iter=num_iter, num_trial=num_trial)
        test_score_list.append(score)

    # print average scores
    print('Mean test score:', np.mean(test_score_list))
    print('Std test score:', np.std(test_score_list))
