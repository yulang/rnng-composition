from random import randint
from os import path

DATA_FOLDER = "/home-nfs/langyu/data_folder/"
WORK_FOLDER = "/home-nfs/langyu/workspace/rnng-composition/"

config = {
    "BIRD_PATH" : path.join(DATA_FOLDER, "BiRD/BiRD.txt"),
    # "BIRD_PATH" : path.join(DATA_FOLDER, "BiRD/bird_transposed.txt"),
    "PPDB_PATH" : path.join(DATA_FOLDER, "ppdb-2.0-tldr"),
    "WIKI_PATH" : path.join(DATA_FOLDER, "enwiki/enwiki-unidecoded.txt"),
    "sample_size" : 10000,
    "workload" : "bird",
    "random_seed" : str(randint(10000, 99999)),
    "out_dir": path.join(WORK_FOLDER, "out/"),
    "vocab_file": "/home-nfs/langyu/rnng/clusters-train-berk.txt",
    "dump_dir": path.join(WORK_FOLDER, "out/"),
}