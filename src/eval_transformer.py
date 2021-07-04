import os
import sys
import logging
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import EvalDataset
from transformers import *
from numpy import save, load

sys.path.append("/home-nfs/langyu/workspace/sentence-composition")
from model_io import load_model, get_pretrained_name

sys.path.append("/home-nfs/langyu/workspace/phrasal-composition-in-transformers/src")

from workload_generator import *
from classifier import *
from analyzer import TransformerAnalyzer
from utilities import adjust_transformer_range, analyze_correlation_by_layer, print_stats_by_layer, \
    concact_hidden_states

from config import config

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def remove_clf_head(model):
    cur_rand_seed = config["random_seed"]
    tmp_dir = os.path.join(config["tmp_dir"], config["model_name"], cur_rand_seed)
    model.save_pretrained(tmp_dir)
    base_model = AutoModel.from_pretrained(tmp_dir, output_hidden_states=True, output_attentions=True)
    return base_model


# replace encode_padded_input in phrasal repo
def encode_input(tokenizer, dataloader, model_name):
    max_len = 250
    input_id_list, attention_mask_list, sequence_length = [], [], []
    pad_id = tokenizer.pad_token_id
    for sequence_batch in dataloader:
        encoded_info = tokenizer(sequence_batch, \
                                return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
        input_ids, attention_masks = encoded_info['input_ids'], encoded_info['attention_mask']
        input_id_list.append(input_ids)
        attention_mask_list.append(attention_masks)
        batch_max_len = input_ids.shape[1]
        length_list = [batch_max_len - list(x).count(pad_id) for x in input_ids]
        sequence_length.extend(length_list)

    input_id_list = torch.cat(input_id_list, 0)
    attention_mask_list = torch.cat(attention_mask_list, 0)
    return input_id_list, attention_mask_list, sequence_length


def forward_input(model, model_name, input_ids, input_mask):
    if model_name in ["roberta", "bert", "xlmroberta"]:
        last_hidden_state, pooler_output, hidden_states, attentions = model(input_ids, attention_mask=input_mask)
    elif model_name in ["distillbert"]:
        last_hidden_state, hidden_states, attentions = model(input_ids, attention_mask=input_mask)
    elif model_name in ['xlnet']:
        last_hidden_state, mems, hidden_states, attentions = model(input_ids, attention_mask=input_mask)
    else:
        logger.error("unsupported model: {}".format(model_name))
        exit(1)
    return hidden_states


def eval_and_dump_embeddings(model, model_name, data_loader, dump_path, dump_every):
    assert os.path.exists(dump_path) is False
    dump_write_handler = open(dump_path, "ab")
    accumulated_hidden_states = None
    cached_count = 0

    for input_ids, input_mask in data_loader:
        hidden_states = forward_input(model, model_name, input_ids, input_mask)

        if accumulated_hidden_states is None:
            accumulated_hidden_states = list(hidden_states)
        else:
            accumulated_hidden_states = concact_hidden_states(accumulated_hidden_states, hidden_states)
        cached_count += 1

        if cached_count == dump_every:
            save(dump_write_handler, accumulated_hidden_states)  # note: dump accumulated hidden states
            cached_count = 0
            accumulated_hidden_states = None

    if cached_count != 0:
        # dump remaining segments
        save(dump_write_handler, accumulated_hidden_states)
        accumulated_hidden_states = None

    dump_write_handler.close()


def main():
    random_seed = config["random_seed"]
    logger.info("current random seed: {}".format(random_seed))
    print(config)
    model_name = config['model_name']
    workload = config["workload"]
    logger.info("preprocessing input...")

    if  workload == "bird":
        input_filename, score_dic, score_range, phrase_pos, phrase_text = bird_preprocess(config["BIRD_PATH"],
                random_seed,
                config["sample_size"],
                normalize=config["normalize"], out_folder=config["out_dir"])
        phrase_dic = score_dic
    elif workload == "ppdb":
        input_filename, score_dic, score_range, phrase_pos, phrase_text, samples_dic = \
            ppdb_preprocess(config["PPDB_PATH"], random_seed, config["sample_size"],
                            negative_sampling_mode=config["negative_sample_mode"],
                            overlap_threshold=config["overlap_threshold"], out_folder=config["out_dir"])
        phrase_dic = score_dic
    elif workload == "ppdb_exact":
        input_filename, exact_label_dic, phrase_pos, phrase_text = ppdb_exact_preprocess(config["PPDB_PATH"],
                                                                                         random_seed,
                                                                                         config["sample_size"], out_folder=config["out_dir"])
        phrase_dic = exact_label_dic
    else:
        print("unsupport workload " + workload)
        exit(1)

    logger.info("current eval_configuration: {}".format(config))
    logger.info("adjusting evaluation data....")

    if config["embed_in_sent"]:
        logger.info("Embedding phrase in wiki text")
        if workload == "ppdb_exact":
            logger.info("Before truncating: {}".format(len(phrase_text)))
            sentence_texts, phrase_text, exact_label_dic = embed_phrase_and_truncate(phrase_dic, phrase_text, config["TEXT_CORPUS"])
            logger.info("After truncating: {}".format(len(sentence_texts)))
        else:
            sentence_texts = embed_phrase_transformer(phrase_dic, phrase_text, config["TEXT_CORPUS"])

        sents_loc = os.path.join(config["out_dir"], "embedded_sents_" + random_seed + ".txt")
        sent_out = open(sents_loc, "w")
        for sentence in sentence_texts:
            sent_out.write(sentence)
        sent_out.close()

    logger.info("loading model...")
    pretrained_name = get_pretrained_name(model_name)
    trained_model_name = "tuned-" + pretrained_name + ".pt"
    trained_model_loc = os.path.join(config["tuned_model_dir"], trained_model_name)
    model, tokenizer = load_model(model_loc=trained_model_loc, load_tuned=True, pretrained_name=pretrained_name)
    model = remove_clf_head(model)

    logger.info("model being evaluated: {}".format(model.config))
    model_config = model.config
    n_layers, n_heads = model_config.num_hidden_layers, model_config.num_attention_heads

    logger.info("encoding input...")
    if config["embed_in_sent"]:
        eval_text_dataset = EvalDataset(sentence_texts)
    else:
        eval_text_dataset = EvalDataset(phrase_text)

    eval_text_loader = DataLoader(dataset=eval_text_dataset, shuffle=False, batch_size=config["batch_size"])

    input_id_list, attention_mask_list, input_sequence_length_list = encode_input(tokenizer, eval_text_loader, config["model_name"])

    logger.info("adjusting phrase position & genreating label dic")
    if (model_name in ['roberta']) and (config["embed_in_sent"] is True):
        # tokenizer is space sensitive. 'access' has different id than ' access'
        add_space_before_phrase = True
    else:
        add_space_before_phrase = False

    phrase_pos = adjust_transformer_range(phrase_text, input_id_list, tokenizer, model_name, space_before_phrase=add_space_before_phrase)

    if workload == "ppdb":
        # generate label dic for classification task
        if config["negative_sample_mode"] is None:
            label_dic = nontrivial_score_to_label(score_dic, score_range)
        else:
            label_dic = trivial_score_to_label(score_dic)

    #----------------------------- evaluation -------------------------------#
    logger.info("evaluating model")
    model.eval()
    dump_filename = "{}-dump-{}.npy".format(model_name, random_seed)
    dump_path = os.path.join(config["tmp_dir"], dump_filename)
    batch_size = config["batch_size"]

    eval_data = TensorDataset(input_id_list, attention_mask_list)
    data_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)

    eval_and_dump_embeddings(model, model_name, data_loader, dump_path, config["dump_every"])

    logger.info("dumping segment size: {} samples per segment".format(batch_size * config["dump_every"]))

    logger.info("working on downstream task")
    analyzer = TransformerAnalyzer(dump_path, n_layers, phrase_text, phrase_text, input_sequence_length_list, model_name, True)

    if workload == "bird":
        logger.info("analyzing correlation...")
        coe_by_layer, cos_sim_by_layer, target_score = analyze_correlation_by_layer(analyzer, score_dic, phrase_pos, True)
        print_stats_by_layer(coe_by_layer, is_list=False, stat_type="cor", out_folder=config["out_dir"])
        analyzer.reset_handler()
    elif workload == "ppdb":
        generate_classifier_workloads(analyzer, config, random_seed, phrase_text, label_dic, phrase_pos, True)
    elif workload == "ppdb_exact":
        generate_classifier_workloads(analyzer, config, random_seed, phrase_text, exact_label_dic, phrase_pos, True)
    else:
        logger.error("unsupport task {}".format(workload))


if __name__ == "__main__":
    main()