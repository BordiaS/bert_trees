file_path=../extracted_trees_random/
#file_path=/scratch/sb6416/Ling3340/extract_tree/extracted_trees/v1/
for f in $(ls /scratch/sb6416/Ling3340/extract_tree/extracted_trees_random/)
do
    python eval.py $f >> parse_eval_random_newformat.txt 2>&1 
done
