import logging
import os
import sys
from transformers import AdamW

from config import config
from preprocessing import load_parsed_sst

sys.path.append("/home-nfs/langyu/workspace/sentence-composition")

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

from model_io import load_model, get_pretrained_name
from train import train_iter


def main():
    logger.info("init model...")
    pretrained_name = get_pretrained_name(config["model_name"])
    dump_name = "tuned-" + pretrained_name + ".pt"
    model_dump_loc = os.path.join(config["tuned_model_dir"], dump_name)
    raw_sst_loc = os.path.join(config["SST_PATH"], "dictionary.txt")
    num_labels = 5

    train_loader, test_loader = load_parsed_sst(config["parsed_sst_path"], raw_sst_loc)
    model, tokenizer = load_model(pretrained_name=pretrained_name, load_tuned=False, num_labels=num_labels)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    logger.info("training...")
    model.train()
    init_prec, init_loss, best_prec, best_loss, best_model = train_iter(model, tokenizer, optimizer, train_loader, test_loader, task="standsent", early_stopping=True, max_epochs=config["n_epochs"], print_every=config["print_every"], evaluate_every=config["evaluate_every"], model_out_loc=model_dump_loc)
    logger.info("done training.")

    training_info_str = \
    """ training summary:
    training loss {} -> {}
    test precision {} -> {}
    """.format(init_loss, best_loss, init_prec, best_prec)
    logger.info(training_info_str)


if __name__ == "__main__":
    main()