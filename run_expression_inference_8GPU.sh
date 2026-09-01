VARIANTS_FOLDER="data/expression_inference_liver/"
GENES_PATH="data/expression_inference_liver/genes/gene_tss.v2.csv"

B37_path="data/genomes/Homo_sapiens_assembly19.fasta"

CONFIG_PATH="data/expression_inference_liver/inference.yaml"
GENA_LM_HOME="."
DESCRIPTION_PATH="data/expression_inference_liver/description/ENCFF142RGB.json"

CHECKPOINT_PATH="$GENA_LM_HOME/models/full8192/full8192_best.bin"

for i in {1..8}; do
    python downstream_tasks/eQTL_benchmark/expression_inference_complete_tss.py \
      --variants-csv "$VARIANTS_FOLDER/Liver.train.v2.part${i}.csv" \
      --gene-tss-tsv "$GENES_PATH" \
      --hg38-fasta "$B37_path" \
      --b37-fasta "$B37_path" \
      --out "Liver.train.8192.complete_tss.part${i}.csv" \
      --config "$CONFIG_PATH" \
      --gena-lm-home "$GENA_LM_HOME" \
      --checkpoint "$CHECKPOINT_PATH" \
      --description-path "$DESCRIPTION_PATH" \
      --batch-size 32 \
      --device "cuda:$((i-1))" \
      > "part${i}.log" 2>&1 &
done

wait

echo "Train jobs finished."