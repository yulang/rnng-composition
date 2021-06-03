from config import config
import pickle
import logging
from scipy.stats.stats import pearsonr
import torch.nn as nn
import torch
from os import path
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import sys

sys.path.append("/home-nfs/langyu/workspace/phrasal-composition-in-transformers/src")
from workload_generator import trivial_score_to_label

# from utilities import analyze_correlation_by_layer
import pdb
MAX_ITER = 200


def generate_embedding_file_list(workload, emb_dir):
    # ordered as stack, term, stack + term
    file_list = []
    stack_emb = path.join(emb_dir, workload + "_stack_vec.txt")
    file_list += stack_emb,
    term_emb = path.join(emb_dir, workload + "_term_vec.txt")
    file_list += term_emb,
    stack_term_emb = path.join(emb_dir, workload + "_stack+term_vec.txt")
    file_list += stack_term_emb,

    return file_list


def load_embedding_file(file):
    embeddings = []
    handler = open(file, "r")
    for line in handler:
        emb = line.strip().split()
        emb = [float(x) for x in emb]
        embeddings.append(emb)

    handler.close()
    return embeddings


def parse_embeddings(file_list):
    assert len(file_list) == 3
    stack_emb = load_embedding_file(file_list[0])
    term_emb = load_embedding_file(file_list[1])
    stack_term_emb = load_embedding_file(file_list[2])

    return [stack_emb, term_emb, stack_term_emb]


def compute_cos_sim(src_emb, tgt_emb_list, normalized=True):
    cos_sim_list = []
    cos_sim = nn.CosineSimilarity(dim=0)
    src_emb = torch.tensor(src_emb)
    for tgt_emb in tgt_emb_list:
        tgt_emb = torch.tensor(tgt_emb)
        sim = cos_sim(src_emb, tgt_emb)
        if normalized:
            sim = (sim + 1) / 2.0
        cos_sim_list.append(sim.item())

    return cos_sim_list


def analyze_correlation(phrase_dic, embeddings):
    target_scores = []
    cosine_sims = []
    for source_phrase_index in phrase_dic:
        target_phrase_list = phrase_dic[source_phrase_index]
        source_emb = embeddings[source_phrase_index]
        target_emb_list = []
        for target_phrase_index, score in target_phrase_list:
            target_emb = embeddings[target_phrase_index]
            target_scores += score,
            target_emb_list.append(target_emb)

        current_pair_cos_value = compute_cos_sim(source_emb, target_emb_list)
        cosine_sims.extend(current_pair_cos_value)

    assert len(target_scores) == len(cosine_sims)
    cor, _ = pearsonr(cosine_sims, target_scores)
    return cor


def ppdb_classification(label_dic, embeddings):
    labels = []
    inputs = []

    for source_phrase_index in label_dic:
        target_phrase_list = label_dic[source_phrase_index]
        source_emb = embeddings[source_phrase_index]

        for target_phrase_index, label in target_phrase_list:
            target_emb = embeddings[target_phrase_index]
            input_emb = source_emb + target_emb
            
            labels.append(label)
            inputs.append(input_emb)
        # embedding = torch.cat()

    logger.info("Loaded {} samples.".format(len(labels)))
    logger.info("Input embedding size {}".format(len(inputs[0])))

    x = np.asarray(inputs)
    y = np.asarray(labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    clf = MLPClassifier(hidden_layer_sizes=256, activation='relu', max_iter=MAX_ITER)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)

    return acc

    

def main():
    workload = config["workload"]
    logger.info("current workload: {}".format(workload))
    logger.info("loading embeddings and dump")
    
    # loading embeddings
    embedding_file_list = generate_embedding_file_list(workload, config["emb_dir"])
    stack_emb, term_emb, stack_term_emb = parse_embeddings(embedding_file_list)
    # loading phrase dic
    dump_file = path.join(config["dump_dir"], workload + "_phrase_dic.dump")
    phrase_dic = pickle.load(open(dump_file, "rb"))


    logger.info("evaluation")
    if workload in ["bird", "bird_abba"]:
        stack_cor = analyze_correlation(phrase_dic, stack_emb)
        term_cor = analyze_correlation(phrase_dic, term_emb)
        stack_term_cor = analyze_correlation(phrase_dic, stack_term_emb)

        logger.info("stack vector correlation: {}".format(stack_cor))
        logger.info("term vector correlation: {}".format(term_cor))
        logger.info("stack + term vector correlation: {}".format(stack_term_cor))
    elif workload in ["ppdb", "ppdb_exact"]:
        if workload == "ppdb":
            label_dic = trivial_score_to_label(phrase_dic)
        else:
            label_dic = phrase_dic
        stack_acc = ppdb_classification(label_dic, stack_emb)
        term_acc = ppdb_classification(label_dic, term_emb)
        stack_term_acc = ppdb_classification(label_dic, stack_term_emb)

        logger.info("stack vector accuracy: {}".format(stack_acc))
        logger.info("term vector accuracy: {}".format(term_acc))
        logger.info("stack + term vector accuracy: {}".format(stack_term_acc))
    else:
        logger.error("unsupported workload: {}".format(workload))


def test():
    load_embedding_file("/home-nfs/langyu/workspace/rnng-composition/emb/bird_abba_stack_vec.txt")



if __name__ == "__main__":
    main()
    # test()