# script from Hewitt et al., 2019
# Downloads example corpora and vectors for structural probing.
# Includes conllx files, raw text files, and ELMo contextual word representations

# By default, downloads a (very) small subset of the EN-EWT
# universal dependencies corpus. 

# For demo purposes, also downloads pre-trained probes on BERT-large.

wget https://nlp.stanford.edu/~johnhew/public/en_ewt-ud-sample.tgz
tar xzvf en_ewt-ud-sample.tgz
mkdir -p example/data
mv en_ewt-ud-sample example/data
rm en_ewt-ud-sample.tgz
