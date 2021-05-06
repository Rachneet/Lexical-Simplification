import os
import fileinput
import re
import time

from fairseq import options
from fairseq_cli import generate

from collections import defaultdict
import shutil
from pathlib import Path

from access.preprocessors import get_preprocessors
from access.resources.prepare import prepare_models
from access.simplifiers import ComposedPreprocessor
from access.text import word_tokenize
from access.utils.helpers import yield_lines, write_lines, log_stdout


def _fairseq_generate(intermediate_filepath,
                      checkpoint_paths,
                      beam=5,
                      hypothesis_num=1,
                      lenpen=1.,
                      diverse_beam_groups=None,
                      diverse_beam_strength=0.5,
                      sampling=False,
                      batch_size=1):
    # exp_dir must contain checkpoints/checkpoint_best.pt, and dict.{complex,simple}.txt
    # First copy input complex file to exp_dir and create dummy simple file
    tmp_dir = Path(intermediate_filepath).parent
    new_complex_filepath = tmp_dir / 'test.complex-simple.complex'
    dummy_simple_filepath = tmp_dir / 'test.complex-simple.simple'
    shutil.copy(intermediate_filepath, new_complex_filepath)
    shutil.copy(intermediate_filepath, dummy_simple_filepath)
    generate_parser = options.get_generation_parser()

    args = [
        tmp_dir,
        '--path',
        ':'.join([str(path) for path in checkpoint_paths]),
        '--beam',
        beam,
        '--nbest',
        hypothesis_num,
        '--lenpen',
        lenpen,
        '--diverse-beam-groups',
        diverse_beam_groups if diverse_beam_groups is not None else -1,
        '--diverse-beam-strength',
        diverse_beam_strength,
        '--batch-size',
        batch_size,
        '--raw-text',
        '--print-alignment',
        # '--gen-subset',
        # 'tmp',
        '--cpu',
        # We don't want to reload pretrained embeddings
        '--model-overrides',
        {
            'encoder_embed_path': None,
            'decoder_embed_path': None
        },
    ]
    if sampling:
        args.extend([
            '--sampling',
            '--sampling-topk',
            10,
        ])
    args = [str(arg) for arg in args]
    generate_args = options.parse_args_and_arch(generate_parser, args)
    out_filepath = tmp_dir / 'generation.out'
    with log_stdout(out_filepath, mute_stdout=True):
        # evaluate model in batch mode
        generate.main(generate_args)

    # Retrieve translations
    def parse_all_hypotheses(out_filepath):
        hypotheses_dict = defaultdict(list)
        for line in yield_lines(out_filepath):
            match = re.match(r'^H-(\d+)\t-?\d+\.\d+\t(.*)$', line)
            if match:
                sample_id, hypothesis = match.groups()
                hypotheses_dict[int(sample_id)].append(hypothesis)
        # Sort in original order
        return [hypotheses_dict[i] for i in range(len(hypotheses_dict))]

    all_hypotheses = parse_all_hypotheses(out_filepath)
    predictions = [hypotheses[hypothesis_num-1] for hypotheses in all_hypotheses]
    # print(predictions)
    write_lines(predictions, out_filepath)
    os.remove(dummy_simple_filepath)
    os.remove(new_complex_filepath)

    output_pred_filepath = tmp_dir / "final.txt"
    composed_preprocessor = ComposedPreprocessor(preprocessors)
    composed_preprocessor.decode_file(out_filepath,
                                      output_pred_filepath,
                                      encoder_filepath=intermediate_filepath)
    for line in yield_lines(output_pred_filepath):
        # print(line)
        return line


def simplify_text(preprocessors, input):
    # setup paths
    base_path = "/home/ubuntu/access/resources/models/best_model/"
    src_path = base_path + "src_text.txt"
    intermediate_path = base_path + "processed_text.txt"
    write_lines([word_tokenize(input)], src_path)

    # encode
    composed_preprocessor = ComposedPreprocessor(preprocessors)
    composed_preprocessor.encode_file(src_path, intermediate_path)

    # simplify and decode
    model_dir = base_path + "checkpoints/checkpoint_best.pt"
    complex_dictionary_path = base_path + "dict.complex.txt"
    simple_dictionary_path = base_path + "dict.simple.txt"
    output = _fairseq_generate(intermediate_path, [model_dir], beam=8)
    output = re.sub(r'\s([?.!"](?:\s|$))', r'\1', output)
    print(output)


if __name__ == '__main__':

    # Load best model
    best_model_dir = "/home/ubuntu/access/resources/models/best_model"   # prepare_models()
    recommended_preprocessors_kwargs = {
        'LengthRatioPreprocessor': {'target_ratio': 0.90},
        'LevenshteinPreprocessor': {'target_ratio': 0.80},
        'WordRankRatioPreprocessor': {'target_ratio': 0.75},
        'SentencePiecePreprocessor': {'vocab_size': 10000},
    }
    preprocessors = get_preprocessors(recommended_preprocessors_kwargs)

    text = "How to bake a delicious white chocolate cream tart?"
    start = time.time()
    simplify_text(preprocessors, text)
    print(time.time()-start)