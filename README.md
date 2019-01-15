# Supervised OIE Wrapper

This is thin wrapper over [AllenNLP](allennlp.org)'s [pretrained Open IE model](https://demo.allennlp.org/open-information-extraction).

Outputs predictions identical to those in the demo, with batched gpu options.

## Install prerequisites

    pip install requirements.txt

## Run on raw sentences

    cd src
    python run_oie.py --in=path/to/input/file  --batch-size=<batch-size> --out=path/to/output/file [--cuda-device=<cude-device-identifier]

## Input format

Raw sentences, each in a new line.

## Output Format
Each line pertains to a single OIE extraction:

    tokenized sentence <tab> ARG0:.. <tab> V:... <tab> ARG1:...  ...

## Example

See example of input and output files in src/example.txt and src/example.oie
