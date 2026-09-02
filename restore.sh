#!/usr/bin/env bash

set -euo pipefail

archive="gena_lm_bundle.tar.gz"

B37_path="data/genomes/Homo_sapiens_assembly19.fasta"

genome_dir="data/genomes"
models_dir="models"

# Temporary extraction directory
stage="$(mktemp -d)"

cleanup() {
    rm -rf "$stage"
}
trap cleanup EXIT

echo "Extracting archive..."

mkdir -p "$genome_dir"
mkdir -p "$models_dir"

tar -xzf "$archive" -C "$stage"

# Archive wrapper directory
bundle="$stage/gena_lm_bundle"

if [[ ! -d "$bundle" ]]; then
    echo "ERROR: Expected archive directory 'gena_lm_bundle/' was not found."
    exit 1
fi

# Expected archive contents
genome="$bundle/Homo_sapiens_assembly19.fasta"
genome_fai="$bundle/Homo_sapiens_assembly19.fasta.fai"
full8192="$bundle/full8192"
decoder="$bundle/decoder"
modernbert="$bundle/modernbert_large_8192"

echo "Validating archive contents..."

for path in \
    "$genome" \
    "$genome_fai" \
    "$full8192" \
    "$decoder" \
    "$modernbert"
do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: Required archive entry is missing:"
        echo "  $path"
        exit 1
    fi
done

if [[ ! -f "$full8192/full8192_best.bin" ]]; then
    echo "ERROR: Missing $full8192/full8192_best.bin"
    exit 1
fi

if [[ ! -f "$decoder/config.json" ]]; then
    echo "ERROR: Missing $decoder/config.json"
    exit 1
fi

if [[ ! -f "$modernbert/config.json" ]]; then
    echo "ERROR: Missing $modernbert/config.json"
    exit 1
fi

echo "Archive validation successful."

echo "Installing genome..."

# Files can simply be overwritten.
mv -f "$genome" "$B37_path"
mv -f "$genome_fai" "${B37_path}.fai"

echo "Installing models..."

# IMPORTANT:
# Remove existing destination directories first.
# Otherwise:
#   mv full8192 models/full8192
# would create:
#   models/full8192/full8192/
#
# Removing the destination allows the extracted directory itself
# to become models/full8192.

rm -rf "$models_dir/full8192"
rm -rf "$models_dir/decoder"
rm -rf "$models_dir/modernbert_large_8192"

mv "$full8192" "$models_dir/full8192"
mv "$decoder" "$models_dir/decoder"
mv "$modernbert" "$models_dir/modernbert_large_8192"

echo "Validating final installation..."

for path in \
    "$B37_path" \
    "${B37_path}.fai" \
    "$models_dir/full8192/full8192_best.bin" \
    "$models_dir/decoder/config.json" \
    "$models_dir/modernbert_large_8192/config.json"
do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: Final installation is incomplete."
        echo "Missing:"
        echo "  $path"
        exit 1
    fi
done

echo
echo "Done."
echo
echo "Installed:"
echo "  $B37_path"
echo "  $models_dir/full8192/full8192_best.bin"
echo "  $models_dir/decoder/config.json"
echo "  $models_dir/modernbert_large_8192/config.json"