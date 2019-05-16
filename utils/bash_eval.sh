#file_path=../../extract_tree/extracted_trees/v1/
#file_path=/misc/vlgscratch4/BowmanGroup/pmh330/LINGA_outputs/bert_large_ud_undirected
#file_path=/misc/vlgscratch4/BowmanGroup/pmh330/LINGA_outputs/random_bert_large_ud/
#file_path=/misc/vlgscratch4/BowmanGroup/pmh330/LINGA_outputs/bert_large_ud/
#file_path=../../extract_tree/extracted_trees_random/

#new paths
mode="undirected"
file_path=/misc/vlgscratch4/BowmanGroup/pmh330/LINGA_outputs/ud_outputs/tree_bert_large_uncased_directed
outpath=outputs/outputs_ud/tree_bert_large_uncased_$mode.csv
echo "Layers,F1,self, prev, next,acl,acl:relcl,advcl,advmod,amod,appos,aux,aux:pass,case,cc,ccomp,compound,conj,cop,csubj,csubj:pass,dep,det,det:predet,expl,fixed,flat,iobj,mark,nmod,nmod:npmod,nmod:poss,nmod:tmod,nsubj,nsubj:pass,nummod,obj,obl,obl:tmod,parataxis,punct,root,vocative,xcomp" >> $outpath 2>&1
for f in $(ls $file_path)
do
    python eval.py $file_path $f $mode >> $outpath 2>&1
done
echo "DONE 1"


file_path=/misc/vlgscratch4/BowmanGroup/pmh330/LINGA_outputs/ud_outputs/tree_cola_bert_large_cased_directed
outpath=outputs/outputs_ud/tree_cola_bert_large_cased_$mode.csv
echo "Layers,F1,self, prev, next,acl,acl:relcl,advcl,advmod,amod,appos,aux,aux:pass,case,cc,ccomp,compound,conj,cop,csubj,csubj:pass,dep,det,det:predet,expl,fixed,flat,iobj,mark,nmod,nmod:npmod,nmod:poss,nmod:tmod,nsubj,nsubj:pass,nummod,obj,obl,obl:tmod,parataxis,punct,root,vocative,xcomp" >> $outpath 2>&1
for f in $(ls $file_path)
do
    python eval.py $file_path $f $mode >> $outpath 2>&1
done
echo "DONE 2"

file_path=/misc/vlgscratch4/BowmanGroup/pmh330/LINGA_outputs/ud_outputs/tree_bert_large_cased_directed
outpath=outputs/outputs_ud/tree_bert_large_cased_$mode.csv
echo "Layers,F1,self, prev, next,acl,acl:relcl,advcl,advmod,amod,appos,aux,aux:pass,case,cc,ccomp,compound,conj,cop,csubj,csubj:pass,dep,det,det:predet,expl,fixed,flat,iobj,mark,nmod,nmod:npmod,nmod:poss,nmod:tmod,nsubj,nsubj:pass,nummod,obj,obl,obl:tmod,parataxis,punct,root,vocative,xcomp" >> $outpath 2>&1
for f in $(ls $file_path)
do
    python eval.py $file_path $f $mode >> $outpath 2>&1
done
echo "Done 3"


file_path=/misc/vlgscratch4/BowmanGroup/pmh330/LINGA_outputs/ud_outputs/tree_rand_bert_large_uncased_directed
outpath=outputs/outputs_ud/tree_rand_bert_large_uncased_$mode.csv
echo "Layers,F1,self, prev, next,acl,acl:relcl,advcl,advmod,amod,appos,aux,aux:pass,case,cc,ccomp,compound,conj,cop,csubj,csubj:pass,dep,det,det:predet,expl,fixed,flat,iobj,mark,nmod,nmod:npmod,nmod:poss,nmod:tmod,nsubj,nsubj:pass,nummod,obj,obl,obl:tmod,parataxis,punct,root,vocative,xcomp" >> $outpath 2>&1
for f in $(ls $file_path)
do
    python eval.py $file_path $f $mode >> $outpath 2>&1
done
echo "DONE 4"
