""" Usage:
    <file-name> --in=INPUT_FILE --batch-size=BATCH-SIZE --out=OUTPUT_FILE [--cuda-device=CUDA_DEVICE] [--debug]
"""
# External imports
import logging
from pprint import pprint
from pprint import pformat
from docopt import docopt
import json
import pdb
from tqdm import tqdm
from allennlp.pretrained import open_information_extraction_stanovsky_2018
from collections import defaultdict
from operator import itemgetter
import functools
import operator
import spacy

# Local imports
from . import format_oie
#=-----

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def create_instances(tokenized_sent):
    """
    Convert a sentence into a list of instances.
    """
    # Find all verbs in the input sentence
    pred_ids = [i for (i, t) in enumerate(tokenized_sent)
                if t.pos_ == "VERB" or t.pos_ == "AUX"]

    # Create instances
    instances = [{"sentence": tokenized_sent,
                  "predicate_index": pred_id}
                 for pred_id in pred_ids]

    return instances

def get_confidence(model, tag_per_token, class_probs):
    """
    Get the confidence of a given model in a token list, using the class probabilities
    associated with this prediction.
    """
    token_indexes = [model._model.vocab.get_token_index(tag, namespace = "labels") for tag in tag_per_token]

    # Get probability per tag
    probs = [class_prob[token_index] for token_index, class_prob in zip(token_indexes, class_probs)]

    # Combine (product)
    prod_prob = functools.reduce(operator.mul, probs)

    return prod_prob

def run_oie(tokenized_sentences, batch_size=1, cuda_device=-1, debug=False):
    """
    Run the OIE model and process the output.
    nlp is a spacy model.
    """

    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    # Init OIE
    model = open_information_extraction_stanovsky_2018()

    # Move model to gpu, if requested
    if cuda_device >= 0:
        model._model.cuda(cuda_device)

    # process sentences
    logging.info("Processing sentences")
    oie_lines = []
    for chunk in tqdm(chunks(tokenized_sentences, batch_size)):
        oie_inputs = []
        for sent in chunk:
            oie_inputs.extend(create_instances(sent))
        if not oie_inputs:
            # No predicates in this sentence
            continue

        # Run oie on sents
        sent_preds = model.predict_batch_json(oie_inputs)
        # Collect outputs in batches
        predictions_by_sent = defaultdict(list)
        for outputs in sent_preds:
            sent_tokens = outputs["words"]
            tags = outputs["tags"]
            sent_str = " ".join(sent_tokens)
            assert(len(sent_tokens) == len(tags))
            predictions_by_sent[sent_str].append((outputs["tags"], outputs["class_probabilities"]))

        # Create extractions by sentence
        for sent_idx, (sent, predictions_for_sent) in enumerate(predictions_by_sent.items()):
            raw_tags = list(map(itemgetter(0), predictions_for_sent))
            class_probs = list(map(itemgetter(1), predictions_for_sent))

            # Compute confidence per extraction
            confs = [get_confidence(model, tag_per_token, class_prob)
                     for tag_per_token, class_prob in zip(raw_tags, class_probs)]

            frames = format_oie.format_extractions(chunk[sent_idx], raw_tags)
            # for conf, frame in zip(confs, frames):
            #     frame['confidence'] = conf
            oie_lines.append(frames)
    logging.info("DONE")
    return oie_lines





if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = args["--in"]
    batch_size = int(args["--batch-size"])
    out_fn = args["--out"]
    cuda_device = int(args["--cuda-device"]) if (args["--cuda-device"] is not None) \
                  else -1
    debug = args["--debug"]

    lines = [line.strip()
            for line in open(inp_fn, encoding = "utf8")]

    oie_lines = run_oie(lines, batch_size, cuda_device, debug)

    # Write to file
    logging.info(f"Writing output to {out_fn}")
    with open(out_fn, "w", encoding = "utf8") as fout:
        fout.write("\n".join(oie_lines))
