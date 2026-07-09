VARIANTS_FOLDER="data/expression_inference_liver/"
GENES_PATH="data/expression_inference_liver/genes/hg38_genes.csv"
HG38_path="data/hg38.fa"
B37_path="data/Homo_sapiens_assembly19.fasta"

CONFIG_PATH="data/expression_inference_liver/inference.yaml"
GENA_LM_HOME="."
DESCRIPTION_PATH="data/expression_inference_liver/description/ENCFF142RGB.json"

model_dir=models/
timestamp="20260701"
#Optional to load checkpoint
aws s3 cp s3://genalm/expr/runs/aspeedok/final/20260510-034804/ $model_dir/${timestamp}_full_model/ --recursive --profile airi --endpoint-url https://s3.cloud.ru
CHECKPOINT_PATH="$GENA_LM_HOME/$model_dir/${timestamp}_full_model/pytorch_model.bin"
mkdir -p $model_dir/decoder/
aws s3 cp s3://genalm/runs/moderngena-expression/decoders/moderngena-expression-decoder-L3H1024I1024h8dp0.1/ $model_dir/decoder/ --recursive --profile airi --endpoint-url https://s3.cloud.ru
mkdir -p $model_dir/modernbert_large/
aws s3 cp s3://genalm/runs/moderngena-large-pretrain-promoters_multi_v2_all_checkpoints/ep36-ba108400-hf $model_dir/modernbert_large/ --recursive --profile airi --endpoint-url https://s3.cloud.ru

python downstream_tasks/eQTL_benchmark/expression_inference_varinats_in_center.py \
  --variants-csv "$VARIANTS_FOLDER/Liver.train.v2.part1.csv" \
  --gene-tss-tsv "$GENES_PATH" \
  --hg38-fasta "$HG38_path" \
  --b37-fasta "$B37_path" \
  --out Liver.train.100tok_variant_in_center.part1.csv \
  --config "$CONFIG_PATH" \
  --gena-lm-home "$GENA_LM_HOME" \
  --checkpoint "$CHECKPOINT_PATH" \
  --description-path "$DESCRIPTION_PATH" \
  --batch-size 8 \
  --device 'cuda:0'

python downstream_tasks/eQTL_benchmark/expression_inference_varinats_in_center.py \
  --variants-csv "$VARIANTS_FOLDER/Liver.train.v2.part2.csv" \
  --gene-tss-tsv "$GENES_PATH" \
  --hg38-fasta "$HG38_path" \
  --b37-fasta "$B37_path" \
  --out Liver.train.100tok_variant_in_center.part2.csv \
  --config "$CONFIG_PATH" \
  --gena-lm-home "$GENA_LM_HOME" \
  --checkpoint "$CHECKPOINT_PATH" \
  --description-path "$DESCRIPTION_PATH" \
  --batch-size 8 \
  --device 'cuda:1'

python downstream_tasks/eQTL_benchmark/expression_inference_varinats_in_center.py \
  --variants-csv "$VARIANTS_FOLDER/Liver.train.v2.part3.csv" \
  --gene-tss-tsv "$GENES_PATH" \
  --hg38-fasta "$HG38_path" \
  --b37-fasta "$B37_path" \
  --out Liver.train.100tok_variant_in_center.part3.csv \
  --config "$CONFIG_PATH" \
  --gena-lm-home "$GENA_LM_HOME" \
  --checkpoint "$CHECKPOINT_PATH" \
  --description-path "$DESCRIPTION_PATH" \
  --batch-size 8 \
  --device 'cuda:2'

python downstream_tasks/eQTL_benchmark/expression_inference_varinats_in_center.py \
  --variants-csv "$VARIANTS_FOLDER/Liver.train.v2.part4.csv" \
  --gene-tss-tsv "$GENES_PATH" \
  --hg38-fasta "$HG38_path" \
  --b37-fasta "$B37_path" \
  --out Liver.train.100tok_variant_in_center.part4.csv \
  --config "$CONFIG_PATH" \
  --gena-lm-home "$GENA_LM_HOME" \
  --checkpoint "$CHECKPOINT_PATH" \
  --description-path "$DESCRIPTION_PATH" \
  --batch-size 8 \
  --device 'cuda:3'

python downstream_tasks/eQTL_benchmark/expression_inference_varinats_in_center.py \
  --variants-csv "$VARIANTS_FOLDER/Liver.train.v2.part5.csv" \
  --gene-tss-tsv "$GENES_PATH" \
  --hg38-fasta "$HG38_path" \
  --b37-fasta "$B37_path" \
  --out Liver.train.100tok_variant_in_center.part5.csv \
  --config "$CONFIG_PATH" \
  --gena-lm-home "$GENA_LM_HOME" \
  --checkpoint "$CHECKPOINT_PATH" \
  --description-path "$DESCRIPTION_PATH" \
  --batch-size 8 \
  --device 'cuda:4'


python downstream_tasks/eQTL_benchmark/expression_inference_varinats_in_center.py \
  --variants-csv "$VARIANTS_FOLDER/Liver.train.v2.part6.csv" \
  --gene-tss-tsv "$GENES_PATH" \
  --hg38-fasta "$HG38_path" \
  --b37-fasta "$B37_path" \
  --out Liver.train.100tok_variant_in_center.part6.csv \
  --config "$CONFIG_PATH" \
  --gena-lm-home "$GENA_LM_HOME" \
  --checkpoint "$CHECKPOINT_PATH" \
  --description-path "$DESCRIPTION_PATH" \
  --batch-size 8 \
  --device 'cuda:5'

python downstream_tasks/eQTL_benchmark/expression_inference_varinats_in_center.py \
  --variants-csv "$VARIANTS_FOLDER/Liver.train.v2.part7.csv" \
  --gene-tss-tsv "$GENES_PATH" \
  --hg38-fasta "$HG38_path" \
  --b37-fasta "$B37_path" \
  --out Liver.train.100tok_variant_in_center.part7.csv \
  --config "$CONFIG_PATH" \
  --gena-lm-home "$GENA_LM_HOME" \
  --checkpoint "$CHECKPOINT_PATH" \
  --description-path "$DESCRIPTION_PATH" \
  --batch-size 8 \
  --device 'cuda:6'

  python downstream_tasks/eQTL_benchmark/expression_inference_varinats_in_center.py \
  --variants-csv "$VARIANTS_FOLDER/Liver.train.v2.part8.csv" \
  --gene-tss-tsv "$GENES_PATH" \
  --hg38-fasta "$HG38_path" \
  --b37-fasta "$B37_path" \
  --out Liver.train.100tok_variant_in_center.part8.csv \
  --config "$CONFIG_PATH" \
  --gena-lm-home "$GENA_LM_HOME" \
  --checkpoint "$CHECKPOINT_PATH" \
  --description-path "$DESCRIPTION_PATH" \
  --batch-size 8 \
  --device 'cuda:7'