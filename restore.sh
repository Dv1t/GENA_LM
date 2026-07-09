archive="gena_lm_bundle.zip"

hg38_path="data/genomes/hg38.fa"
B37_path="data/genomes/Homo_sapiens_assembly19.fasta"

mkdir -p models
mkdir -p models/decoder
mkdir -p models/modernbert_large
mkdir -p models/20260701_full_model

mkdir -p data/genomes

unzip "$archive"

# Files
mv -f "hg38.fa" "$hg38_path"
mv -f "hg38.fa.fai" "${hg38_path}.fai"

mv -f "Homo_sapiens_assembly19.fasta" "$B37_path"
mv -f "Homo_sapiens_assembly19.fasta.fai" "${B37_path}.fai"

# Directories
mv -f "20260701_full_model" "models/20260701_full_model"
mv -f "decoder" "models/decoder"
mv -f "modernbert_large" "models/modernbert_large"

echo "Done."