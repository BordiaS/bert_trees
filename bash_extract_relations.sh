source /misc/vlgscratch4/BowmanGroup/pmh330/fairseq/venv_fairseq/bin/activate

ud_input_path=/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/ud_eng_pud.json


bert_input_path=/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/bert_ud/rand_bert_large_uncased.h5
tokens_path=/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/tokens/bert-large-uncased
output_base_path=/misc/vlgscratch4/BowmanGroup/pmh330/LINGA_outputs/ud_outputs/tree_rand_bert_large_uncased_directed
mkdir -p $output_base_path
python bert_trees/extract_trees.py --ud_input_path $ud_input_path  --bert_input_path $bert_input_path  --tokens_path $tokens_path  --output_base_path $output_base_path

bert_input_path=/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/bert_ud/cola_bert_large_uncased.h5
tokens_path=/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/tokens/bert-large-uncased
output_base_path=/misc/vlgscratch4/BowmanGroup/pmh330/LINGA_outputs/ud_outputs/tree_cola_bert_large_cased_directed
mkdir -p $output_base_path
python bert_trees/extract_trees.py --ud_input_path $ud_input_path  --bert_input_path $bert_input_path  --tokens_path $tokens_path  --output_base_path $output_base_path


bert_input_path=/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/bert_ud/bert_large_uncased.h5
tokens_path=/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/tokens/bert-large-uncased
output_base_path=/misc/vlgscratch4/BowmanGroup/pmh330/LINGA_outputs/ud_outputs/tree_bert_large_uncased_directed
mkdir -p $output_base_path
python bert_trees/extract_trees.py --ud_input_path $ud_input_path  --bert_input_path $bert_input_path  --tokens_path $tokens_path  --output_base_path $output_base_path


bert_input_path=/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/bert_ud/bert_large_cased.h5
tokens_path=/misc/vlgscratch4/BowmanGroup/datasets/bert_trees/tokens/bert-large-cased
output_base_path=/misc/vlgscratch4/BowmanGroup/pmh330/LINGA_outputs/ud_outputs/tree_bert_large_cased_directed
mkdir -p $output_base_path
python bert_trees/extract_trees.py --ud_input_path $ud_input_path  --bert_input_path $bert_input_path  --tokens_path $tokens_path  --output_base_path $output_base_path



