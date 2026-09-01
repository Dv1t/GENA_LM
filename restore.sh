archive="gena_lm_bundle.tar.gz"

B37_path="data/genomes/Homo_sapiens_assembly19.fasta"

mkdir -p models
mkdir -p models/decoder
mkdir -p models/modernbert_large_8192
mkdir -p models/full8192
mkdir -p data
mkdir -p data/genomes

tar -xzf "$archive"

# Files

mv -f "Homo_sapiens_assembly19.fasta" "$B37_path"
mv -f "Homo_sapiens_assembly19.fasta.fai" "${B37_path}.fai"

# Directories
mv -f "full8192" "models/full8192"
mv -f "decoder" "models/decoder"
mv -f "modernbert_large_8192" "models/modernbert_large_8192"

echo "Done."