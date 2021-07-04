from random import randint
from os import path

DATA_FOLDER = "/home-nfs/langyu/data_folder/"
WORK_FOLDER = "/home-nfs/langyu/workspace/rnng-composition/"

config = {
    "BIRD_PATH" : path.join(DATA_FOLDER, "BiRD/BiRD.txt"),
    # "BIRD_PATH" : path.join(DATA_FOLDER, "BiRD/bird_transposed.txt"),
    "PPDB_PATH" : path.join(DATA_FOLDER, "ppdb-2.0-tldr"),
    "WIKI_PATH" : path.join(DATA_FOLDER, "enwiki/enwiki-unidecoded.txt"),
    "SST_PATH" : path.join(DATA_FOLDER, "/home-nfs/langyu/data_folder/stanfordSentimentTreebank"),
    "TEXT_CORPUS": path.join(DATA_FOLDER, "enwiki/enwiki-unidecoded.txt"),
    "sample_size" : 15000,
    "workload" : "bird",
    "random_seed" : str(randint(10000, 99999)),
    "out_dir": path.join(WORK_FOLDER, "out/"),
    "vocab_file": "/home-nfs/langyu/rnng/clusters-train-berk.txt",
    "dump_dir": path.join(WORK_FOLDER, "out/"),
    "emb_dir": path.join(WORK_FOLDER, "emb/"),
    "tmp_dir": path.join(DATA_FOLDER, "tmp/"),
    "parsed_sst_path": path.join(WORK_FOLDER, "data/sst_full_parsed.txt"),
    "model_name": "xlnet",
    "n_epochs": 3,
    "print_every": 250,
    "evaluate_every": 500,
    "tuned_model_dir": "/home-nfs/langyu/workspace/rnng-composition/out",
    "load_tuned_model": True,
    "normalize": False,
    "embed_in_sent": False,
    "negative_sample_mode": "half_neg",
    "batch_size": 10,
    "dump_every": 5,
}