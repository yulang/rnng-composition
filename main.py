from random import randint
from workload_generation import gen_parsed_sentence
import pickle

import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import sys

sys.path.append("/home-nfs/langyu/workspace/phrasal-composition-in-transformers/src")

from workload_generator import bird_preprocess, ppdb_exact_preprocess, ppdb_preprocess, embed_phrase_transformer, embed_phrase_and_truncate

BIRD_PATH = "/home-nfs/langyu/data_folder/BiRD/BiRD.txt"
PPDB_PATH = "/home-nfs/langyu/data_folder/ppdb-2.0-tldr"
WIKI_PATH = "/home-nfs/langyu/data_folder/enwiki/enwiki-unidecoded.txt"
sample_size = 10000
workload = "bird"
random_seed = str(randint(10000, 99999))


def main():
    logger.info("parsing evaluation data")
    if workload == "bird":
        input_filename, score_dic, score_range, phrase_pos, phrase_text = bird_preprocess(BIRD_PATH, random_seed, -1, normalize=False)
        phrase_dic = score_dic
    elif workload == "ppdb":
        input_filename, score_dic, score_range, phrase_pos, phrase_text, samples_dic = \
            ppdb_preprocess(PPDB_PATH, random_seed, sample_size,
                            negative_sampling_mode="half_neg",
                            overlap_threshold=0.5)
        phrase_dic = score_dic
    elif workload == "ppdb_exact":
        input_filename, exact_label_dic, phrase_pos, phrase_text = ppdb_exact_preprocess(PPDB_PATH,
                                                                                         random_seed,
                                                                                         sample_size)
        phrase_dic = exact_label_dic
    else:
        logger.error("unknown workload: {}".format(workload))
    
    logger.info("embedding phrases")
    if workload == "ppdb_exact":
        sentence_texts, phrase_text, exact_label_dic = embed_phrase_and_truncate(phrase_dic, phrase_text, WIKI_PATH)
    else:
        sentence_texts = embed_phrase_transformer(phrase_dic, phrase_text, WIKI_PATH)

    logger.info("dumping data structures")
    dump_file = open("./out/phrase_dic_" + random_seed + ".dump", "wb")
    pickle.dump(phrase_dic, dump_file)

    logger.info("parsing sentences")
    parsed_out_name = "./out/parsed_sentences_" + random_seed + ".txt"
    gen_parsed_sentence(sentence_texts, out_file=parsed_out_name)


if __name__ == "__main__":
    main()