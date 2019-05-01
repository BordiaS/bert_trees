file_path=../../extract_tree/extracted_trees/v1/
#file_path=../../extract_tree/extracted_trees_random/
echo "Layers,F1,acl,acl:relcl,advcl,advmod,amod,appos,aux,aux:pass,case,cc,ccomp,compound,conj,cop,csubj,csubj:pass,dep,det,det:predet,expl,fixed,flat,iobj,mark,nmod,nmod:npmod,nmod:poss,nmod:tmod,nsubj,nsubj:pass,nummod,obj,obl,obl:tmod,parataxis,punct,root,vocative,xcomp" >> parse_eval.csv 2>&1 
for f in $(ls $file_path)
do
    python eval.py $file_path $f >> parse_eval.csv 2>&1 
done
