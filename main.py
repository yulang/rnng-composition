from random import randint
from workload_generation import gen_parsed_sentence
from config import config
from os import path
import pickle

import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import sys

sys.path.append("/home-nfs/langyu/workspace/phrasal-composition-in-transformers/src")

from workload_generator import bird_preprocess, ppdb_exact_preprocess, ppdb_preprocess, embed_phrase_transformer, embed_phrase_and_truncate



def main():
    logger.info("parsing evaluation data")
    if config["workload"] == "bird":
        input_filename, score_dic, score_range, phrase_pos, phrase_text = bird_preprocess(config["BIRD_PATH"], config["random_seed"], -1, normalize=False)
        phrase_dic = score_dic
    elif config["workload"] == "ppdb":
        input_filename, score_dic, score_range, phrase_pos, phrase_text, samples_dic = \
            ppdb_preprocess(config["PPDB_PATH"], config["random_seed"], config["sample_size"],
                            negative_sampling_mode="half_neg",
                            overlap_threshold=0.5)
        phrase_dic = score_dic
    elif config["workload"] == "ppdb_exact":
        input_filename, exact_label_dic, phrase_pos, phrase_text = ppdb_exact_preprocess(config["PPDB_PATH"],
                                                                                         config["random_seed"],
                                                                                         config["sample_size"])
        phrase_dic = exact_label_dic
    else:
        logger.error("unknown workload: {}".format(config["workload"]))
    
    logger.info("embedding phrases")
    if config["workload"] == "ppdb_exact":
        sentence_texts, phrase_text, exact_label_dic = embed_phrase_and_truncate(phrase_dic, phrase_text, config["WIKI_PATH"])
    else:
        sentence_texts = embed_phrase_transformer(phrase_dic, phrase_text, config["WIKI_PATH"])

    logger.info("dumping data structures")
    dump_file = open("./out/phrase_dic_" + config["random_seed"] + ".dump", "wb")
    pickle.dump(phrase_dic, dump_file)

    logger.info("parsing sentences")
    parsed_out_name = path.join(config["dump_dir"], "parsed_sentences_" + config["random_seed"] + ".txt")
    gen_parsed_sentence(sentence_texts, out_file=parsed_out_name)


if __name__ == "__main__":
    main()