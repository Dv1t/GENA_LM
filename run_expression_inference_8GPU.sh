VARIANTS_FOLDER="data/expression_inference_liver/"
GENES_PATH="data/expression_inference_liver/genes/hg38_genes.csv"

HG38_path="data/genomes/hg38.fa"
B37_path="data/genomes/Homo_sapiens_assembly19.fasta"

CONFIG_PATH="data/expression_inference_liver/inference.yaml"
GENA_LM_HOME="."
DESCRIPTION_PATH="data/expression_inference_liver/description/ENCFF142RGB.json"

model_dir=models/
timestamp="20260701"
CHECKPOINT_PATH="$GENA_LM_HOME/$model_dir/${timestamp}_full_model/pytorch_model.bin"

for i in {1..8}; do
    python downstream_tasks/eQTL_benchmark/expression_inference_varinats_in_center.py \
      --variants-csv "$VARIANTS_FOLDER/Liver.train.v2.part${i}.csv" \
      --gene-tss-tsv "$GENES_PATH" \
      --hg38-fasta "$HG38_path" \
      --b37-fasta "$B37_path" \
      --out "Liver.train.30tok_variant_in_center.part${i}.csv" \
      --config "$CONFIG_PATH" \
      --gena-lm-home "$GENA_LM_HOME" \
      --checkpoint "$CHECKPOINT_PATH" \
      --description-path "$DESCRIPTION_PATH" \
      --batch-size 8 \
      --device "cuda:$((i-1))" \
      > "part${i}.log" 2>&1 &
done

wait

echo "Train jobs finished."

for i in {1..8}; do
    python downstream_tasks/eQTL_benchmark/expression_inference_varinats_in_center.py \
      --variants-csv "$VARIANTS_FOLDER/Liver.validation.v2.part${i}.csv" \
      --gene-tss-tsv "$GENES_PATH" \
      --hg38-fasta "$HG38_path" \
      --b37-fasta "$B37_path" \
      --out "Liver.validation.30tok_variant_in_center.part${i}.csv" \
      --config "$CONFIG_PATH" \
      --gena-lm-home "$GENA_LM_HOME" \
      --checkpoint "$CHECKPOINT_PATH" \
      --description-path "$DESCRIPTION_PATH" \
      --batch-size 8 \
      --device "cuda:$((i-1))" \
      > "part${i}.log" 2>&1 &
done

wait

echo "Validation jobs finished."