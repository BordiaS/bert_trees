for f in $(ls /scratch/sb6416/Ling3340/extract_tree/extracted_trees/v1/)
do
    python eval.py $f >> parse_eval.txt 2>&1 
done
