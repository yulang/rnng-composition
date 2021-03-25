from config import config
import pickle
import logging
from os import path

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import sys

sys.path.append("/home-nfs/langyu/workspace/phrasal-composition-in-transformers/src")

from utilities import analyze_correlation_by_layer, generate_classifier_workloads


def generate_embedding_file_list():
    pass


def main():
    logger.info("loading embeddings and dump")
    workload = config["workload"]
    stack_emb = path.join(config["dump_dir"], workload +)



if __name__ == "__main__":
    main()