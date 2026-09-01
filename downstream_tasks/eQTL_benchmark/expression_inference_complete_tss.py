#!/usr/bin/env python3
"""Run ExpressionCounts inference using dual sequences:
- 3kbp centered on a credible-set variant position
- 6kbp centered on the gene TSS
Sequences are proportionally truncated if total token count exceeds 1022 (1024 - CLS/SEP).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
from pysam import FastaFile
import json
import re
import pickle
import hashlib
import tempfile
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from downstream_tasks.expression_prediction.expression_model_final import ExpressionCounts
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import math
from torch.utils.data import Sampler
import torch.distributed as dist
from tqdm import tqdm

# Target bp window sizes for each region
VARIANT_REGION_BP  = 3_000   # 3 kbp centered on variant
TSS_REGION_BP      = 12_000   # 12 kbp centered on TSS
MAX_TOKENS         = 8_192   # hard model limit including CLS + SEP
VARIANT_REGION_TOKENS = 30
MAX_CONTENT_TOKENS = MAX_TOKENS - 3   # 1022 usable token slots
# Approximate token/bp ratio for the GENA-LM tokenizer (used only for initial fetch sizing)
APPROX_BP_PER_TOKEN = 6
# A complete variant-TSS sequence is used when the variant is within this many
# tokenizer tokens of the TSS. This lets an 8192-token context contain both loci.
COMPLETE_TSS_RADIUS_TOKENS = 4_096  # variant must be within 4096 tokenizer tokens of the TSS


def split_csv_cell(cell: str) -> list[str]:
    """Split a comma-separated cell into a list of non-empty strings."""
    return [x.strip() for x in cell.split(",") if x.strip()]

def load_description_text(meta_path: str | Path) -> str:
    """Reads a metadata JSON file and parses it into a unified text description."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    line_texts = []
    for k, v in meta.items():
        k = str(k).replace('_', ' ')
        v = str(v).replace('_', ' ')
        clean_k = re.sub(r'^(Characteristics|Chracteristics|Charateristics|Parameter)\s*', '', k)
        clean_k = re.sub(r'[\[\]]', '', clean_k).strip()
        clean_k = clean_k if clean_k else k
        clean_v = str(v).replace('"', '').strip()
        line_texts.append(f'{clean_k} is {clean_v}.')
    if not line_texts:
        raise ValueError(f"No description text could be generated from file: {meta_path}")
    return " ".join(line_texts)


def parse_variant_position(variant_str: str) -> int | None:
    """Parse a variant string like '7-150499647-G-A' and return the 1-based position."""
    parts = variant_str.strip().split("-")
    if len(parts) < 2:
        return None
    try:
        return parts[0], int(parts[1]), parts[2], parts[3]
    except ValueError:
        return None

def load_variants_table(variants_csv: str | Path) -> pd.DataFrame:
    """
    Load the credible-set variants table. Returns one row per (signal, gene) pair,
    using the all the listed credible-set variants.

    Columns guaranteed in the returned DataFrame:
      signal_id, chrom, variant_pos_1based, gene_id
    """
    df = pd.read_csv(variants_csv)
    records = []
    for _, row in df.iterrows():
        signal_id = row["common_variant_analysis_signal_id"]
        chrom_raw = str(row["chr"])
        # take just the first variant in the credible set
        all_variants = str(row["credible_set_variants_in_1_based_B37"]).split(",")
        gene_set = set(split_csv_cell(str(row.get("genes_with_zero_expression", ""))))
        for variant in all_variants:
            variant = variant.strip()
            chr_number, variant_pos, ref_base, alt_base = parse_variant_position(variant)
            if chr_number != chrom_raw.lstrip("chr"):
                logging.warning(f"Variant chromosome {chr_number} does not match row chromosome {chrom_raw}")
                continue
            if variant_pos is None:
                continue
            # collect all genes (genes + genes_with_zero_expression, deduplicated)
            for gene_id in sorted(gene_set):
                    records.append({
                        "signal_id":        signal_id,
                        "chrom":            chrom_raw,
                        "variant_pos_1based": variant_pos,
                        "ref_base":     ref_base,
                        "alt_base":     alt_base,
                        "gene_id":          gene_id,
                        "type":'ref'
                    })
                    records.append({
                        "signal_id":        signal_id,
                        "chrom":            chrom_raw,
                        "variant_pos_1based": variant_pos,
                        "ref_base":     ref_base,
                        "alt_base":     alt_base,
                        "gene_id":          gene_id,
                        "type":'alt'
                    })
                    records.append({
                        "signal_id":        signal_id,
                        "chrom":            chrom_raw,
                        "variant_pos_1based": variant_pos,
                        "ref_base":     ref_base,
                        "alt_base":     alt_base,
                        "gene_id":          gene_id,
                        "type":'tss_only'
                    })
    
    result_df = pd.DataFrame(records)
    return result_df


class DualRegionInferenceDataset(Dataset):
    """
    For every (variant, gene) pair, fetches:
      - A sequence of VARIANT_REGION_BP bp centered on the variant
      - A sequence of TSS_REGION_BP bp centered on the gene TSS

    Both sequences are tokenized independently. If the combined token count
    exceeds MAX_CONTENT_TOKENS (1022), they are proportionally shortened so
    the total fits exactly.

    The two token sequences are then concatenated (variant first, TSS second)
    and wrapped with CLS/SEP to form a single input of ≤ 1024 tokens.

    The genes_path TSV must contain at minimum:
      gene_id_unversioned, TSS
    """

    def __init__(
        self,
        genome_variants: str | Path,
        genome_genes: str | Path,
        dna_tokenizer,
        text_tokenizer,
        description: str,
        genes_path: str,
        variants_df: pd.DataFrame,
        seed: int = 42,
        token_len_for_fetch = 12
    ):
        self.genome_variants     = genome_variants
        self.genome_genes        = genome_genes
        self.gen_tokenizer       = dna_tokenizer
        self.text_tokenizer      = text_tokenizer
        self.description         = description
        self.logger              = logging.getLogger(__name__)
        self.token_len_for_fetch = token_len_for_fetch
        
        # Gene TSS lookup: gene_id_unversioned → row
        self.genes_df = pd.read_csv(genes_path, sep=',')
        self.genes_df = self.genes_df.dropna()

        self.tss_lookup: dict[str, dict] = {}
        for _, row in self.genes_df.iterrows():
            gid = row["gene_id"]
            self.tss_lookup[gid] = {
                "TSS":        int(row["TSS_B37"]),
                "reverse": False#row['strand']=='-'
            }
        print(f'From initial {variants_df.shape[0]} variant-gene pairs')
        self.variants_df = variants_df[variants_df["gene_id"].isin(self.tss_lookup)].reset_index(drop=True)
        print(f'{self.variants_df.shape[0]} have genes with known location and will be proceeded')
        self.n_keys = 1  # one prediction stream; type distinguishes ref/alt/tss_only
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _fetch_seq(self, genome: str, chrom: str, center_1based: int, window_bp: int) -> str:
        """Fetch `window_bp` bases centered on `center_1based` (1-based coordinate).
        Clamps to chromosome boundaries and pads with Ns if needed."""
        sequences = FastaFile(genome)
        chrom_len = sequences.get_reference_length(chrom)
        half = window_bp // 2
        # convert to 0-based half-open [start, end)
        start = max(0, center_1based - 1 - half)
        end   = min(chrom_len, center_1based - 1 + half)
        seq = sequences.fetch(chrom, start, end).upper()

        return seq
    
    def _fetch_seq_ref_alt(self, chrom_raw: str, center_1based: int, window_bp: int, ref_base: str, alt_base: str) -> str:
        """Fetch `window_bp` bases centered on `center_1based` (1-based coordinate).
        Clamps to chromosome boundaries and pads with Ns if needed."""
        sequences = FastaFile(self.genome_variants)
        chrom = _resolve_chrom(chrom_raw, sequences)
        half = window_bp // 2
        # convert to 0-based half-open [start, end)
        start = center_1based - 1500
        end   = center_1based + 1500
        seq = sequences.fetch(chrom, start, end).upper()
        assert len(seq) == 3000
        seq_ref = seq
        seq_alt = seq  # fallback if anything goes wrong
        # Replace the reference base with the alternate base at the variant position
        variant_pos_0based = half-1  # position of the variant in the fetched sequence 
        # Sanity check: warn if the variant is far from centre, but don't assert
        expected_centre = len(seq) // 2 - 1
        if variant_pos_0based != expected_centre:
            self.logger.warning(
                f"Variant {chrom}:{center_1based} is not near the centre of the fetched "
                f"sequence (pos_0based={variant_pos_0based}, centre={expected_centre}). "
                f"Sequence may have been clamped at a chromosome boundary."
            )
        if 0 <= variant_pos_0based < len(seq):

            if len(ref_base) == 1 and len(alt_base) == 1:
                # SNP: simple single-base substitution
                if seq[variant_pos_0based] != ref_base.upper():
                    real_ref_base = sequences.fetch(chrom, center_1based-5, center_1based+5).upper()
                    self.logger.warning(
                        f"{start} {end} Reference base mismatch at {chrom}:{center_1based}: "
                        f"expected '{ref_base}', found '{seq[variant_pos_0based]}'. -  "
                        f"{real_ref_base} - "
                    f"Using reference base in sequence."
                )
                seq_alt = (
                    seq[:variant_pos_0based] +
                    alt_base.upper() +
                    seq[variant_pos_0based + 1:]
                )
                seq_ref = (
                    seq[:variant_pos_0based] +
                    ref_base.upper() +
                    seq[variant_pos_0based + 1:]
                )
            elif len(ref_base) > len(alt_base):
                # Handle deletion
                seq_ref = seq
                seq_alt = (
                    seq[:variant_pos_0based] +
                    alt_base.upper() +
                    seq[variant_pos_0based + len(ref_base):]
                )
            else:
                # Handle insertion
                seq_ref = (
                    seq[:variant_pos_0based] +
                    ref_base.upper() +
                    seq[variant_pos_0based + len(ref_base):]
                )
                seq_alt = (
                    seq[:variant_pos_0based] +
                    alt_base.upper() +
                    seq[variant_pos_0based + len(ref_base):]
                )
        else:
            self.logger.warning(
                f"Variant position {chrom}:{center_1based} is out of bounds for fetched sequence. "
                f"Expected ref base '{ref_base}' cannot be applied."
            )
        return seq_ref, seq_alt
    
    

    def _fix_offset_mapping(self, ids: list[int], offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Correct offset_mapping entries for the tokenizer's N/gap token (id 5).

        The tokenizer reports an unreliable `start` offset for this token —
        only its `end` offset reflects the true position in the sequence.
        (This is the same quirk already worked around inline in
        `_tokenize_seq` / `tokenize_genome` via
        `length = end_i - mapping[i-1][1]`.)

        Left uncorrected, this breaks the invariant that offsets are
        monotonic and gap-free, which `token_index_for_base` and the
        token<->base-pair coordinate math in `_fetch_complete_ref_alt`
        depend on. Any N-run/ambiguous-base region in the window can then
        cause a token to be mapped to the wrong base-pair span.
        """
        fixed = list(offsets)
        for i, tok_id in enumerate(ids):
            if tok_id == 5:
                start = fixed[i - 1][1] if i > 0 else 0
                end = fixed[i][1]
                fixed[i] = (start, end)
        return fixed

    def _estimate_sequence_will_fit(self, chrom: str, pos1: int, pos2: int) -> int:
        """Return the token distance between two genomic positions.

        The interval is tokenized as one sequence so this uses the actual DNA
        tokenizer rather than the approximate bp/token ratio.
        """
        left = min(pos1, pos2)
        right = max(pos1, pos2)
        if right - left > 200_000:
            return False  # don't bother fetching a huge sequence
            
        # Include a little sequence on both sides so that offsets for tokens
        # covering the two coordinates are available even at token boundaries.
        flank = 16
        fasta = FastaFile(self.genome_variants)
        chrom = _resolve_chrom(chrom, fasta)
        chrom_len = fasta.get_reference_length(chrom)
        start0 = max(0, left - 1 - flank)
        end0 = min(chrom_len, right + flank)
        seq = fasta.fetch(chrom, start0, end0).upper()

        enc = self.gen_tokenizer.encode_plus(seq, return_offsets_mapping=True)
        ids = enc["input_ids"]
        if len (ids) > MAX_CONTENT_TOKENS:
            return False
        else:
            return True

    def _fetch_complete_ref_alt(
        self,
        chrom_raw: str,
        tss_pos: int,
        variant_pos: int,
        ref_base: str,
        alt_base: str,
        reverse: bool,
    ) -> tuple[str, str]:
        """Fetch one shared context containing both TSS and variant.

        The complete context is centred on the TSS and spans approximately the
        whole 8192-token model budget. Ref/alt are generated by applying the
        variant to the same genomic context before tokenization.
        """
        fasta = FastaFile(self.genome_variants)
        chrom = _resolve_chrom(chrom_raw, fasta)
        chrom_len = fasta.get_reference_length(chrom)

        # Initial fetch is deliberately generous. The final DNA context is
        # selected in token space below, so tokenizer variability is respected.
        half_bp = COMPLETE_TSS_RADIUS_TOKENS * APPROX_BP_PER_TOKEN + 256
        start0 = max(0, tss_pos - 1 - half_bp)
        end0 = min(chrom_len, tss_pos - 1 + half_bp)
        seq = fasta.fetch(chrom, start0, end0).upper()

        enc = self.gen_tokenizer.encode_plus(seq, return_offsets_mapping=True)
        ids = enc["input_ids"][1:-1]
        offsets = enc["offset_mapping"][1:-1]
        # Correct the unreliable start offset reported for the N/gap token
        # (id 5) so token<->bp coordinate math below is consistent even
        # when the window contains N-runs / ambiguous bases.
        offsets = self._fix_offset_mapping(ids, offsets)

        tss_rel = (tss_pos - 1) - start0
        variant_rel = (variant_pos - 1) - start0

        def token_index_for_base(rel_pos: int) -> int:
            for i, (st, en) in enumerate(offsets):
                if st <= rel_pos < en:
                    return i
            distances = [min(abs(rel_pos - st), abs(rel_pos - en)) for st, en in offsets]
            return int(np.argmin(distances))

        tss_tok = token_index_for_base(tss_rel)
        variant_tok = token_index_for_base(variant_rel)

        # Keep a MAX_CONTENT_TOKENS-token window centred on the TSS while
        # guaranteeing that the variant remains inside it.
        total_tokens = len(ids)
        half_content = MAX_CONTENT_TOKENS // 2
        start_tok = max(0, tss_tok - half_content)
        end_tok = min(total_tokens, start_tok + MAX_CONTENT_TOKENS)
        start_tok = max(0, end_tok - MAX_CONTENT_TOKENS)

        if not (start_tok <= variant_tok < end_tok):
            raise ValueError(
                f"Variant {chrom_raw}:{variant_pos} was not retained in complete "
                f"context around TSS {tss_pos}; token distance={abs(variant_tok - tss_tok)}"
            )

        dna_start = offsets[start_tok][0]
        dna_end = offsets[end_tok - 1][1]
        complete_seq = seq[dna_start:dna_end]

        # Variant coordinate inside the selected DNA sequence.
        variant_idx = variant_rel - dna_start
        if not (0 <= variant_idx < len(complete_seq)):
            raise ValueError("Variant coordinate is outside the selected complete sequence")

        ref_upper = ref_base.upper()
        alt_upper = alt_base.upper()

        # Apply the variant in genomic orientation first.
        seq_ref = complete_seq
        seq_alt = complete_seq
        if len(ref_upper) == 1 and len(alt_upper) == 1:
            seq_ref = complete_seq[:variant_idx] + ref_upper + complete_seq[variant_idx + 1:]
            seq_alt = complete_seq[:variant_idx] + alt_upper + complete_seq[variant_idx + 1:]
        elif len(ref_upper) > len(alt_upper):
            seq_ref = complete_seq
            seq_alt = (
                complete_seq[:variant_idx]
                + alt_upper
                + complete_seq[variant_idx + len(ref_upper):]
            )
        else:
            seq_ref = (
                complete_seq[:variant_idx]
                + ref_upper
                + complete_seq[variant_idx + len(ref_upper):]
            )
            seq_alt = (
                complete_seq[:variant_idx]
                + alt_upper
                + complete_seq[variant_idx + len(ref_upper):]
            )

        if reverse:
            seq_ref = self.reverse_complement(seq_ref)
            seq_alt = self.reverse_complement(seq_alt)

        return seq_ref, seq_alt

    def _tokenize_seq(self, seq: str, token_budget) -> list[int]:
        token_lengths = []
        num_before = token_budget//2
        center = len(seq)//2
        try:
            sequence = seq[center-(num_before * self.token_len_for_fetch):center]
        except ValueError as e:
            self.logger.error(f"Error sequence {i}")
    
        encoded_sequence = self.gen_tokenizer.encode_plus(sequence, return_offsets_mapping=True)
        encoded_sequence['input_ids'] = encoded_sequence['input_ids'][1:-1]
        encoded_sequence['offset_mapping'] = encoded_sequence['offset_mapping'][1:-1]
        if len(encoded_sequence['input_ids']) < num_before:
            self.logger.warning(f"Trying to tokenize seq before TSS, but it's too short: {len(encoded_sequence['input_ids'])} < {num_before};")
        tokens_before = encoded_sequence['input_ids'][-num_before:]
        mapping = encoded_sequence['offset_mapping'][-num_before:]
        
        for i, (start_i, end_i) in enumerate(mapping):
            token_id = tokens_before[i]
            if (token_id == 5):
                if i > 0:
                    length = end_i - mapping[i-1][1] 
                else:
                    length = end_i
            else:
                length = end_i - start_i  
            token = self.gen_tokenizer.decode([token_id])  
            token_lengths.append((token_id, token, length))    

        try:
            sequence = seq[center:center+(num_before * self.token_len_for_fetch)]
        except ValueError as e:
            self.logger.error(f"Error sequence {i}")
        
        encoded_sequence = self.gen_tokenizer.encode_plus(sequence, return_offsets_mapping=True)
        tokens_before = encoded_sequence['input_ids'][1:-1]
        mapping = encoded_sequence['offset_mapping'][1:-1]
        for i, (start_i, end_i) in enumerate(mapping):
            token_id = tokens_before[i]
            if (token_id == 5):
                if i > 0:
                    length = end_i - mapping[i-1][1] 
                else:
                    length = end_i
            else:
                length = end_i - start_i 
            token = self.gen_tokenizer.decode([token_id])  
            token_lengths.append((token_id, token, length))
            

        token_lengths_df = pd.DataFrame(token_lengths, columns=['token_id', 'token', 'length'])
        return token_lengths_df

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.variants_df)
    
    def wrap_seq(self, ids_variant, ids_tss):
        # --- Wrap with CLS / SEP ---
        cls_id = self.gen_tokenizer.cls_token_id
        sep_id = self.gen_tokenizer.sep_token_id
        gap_id = 5

        assert cls_id is not None and sep_id is not None and gap_id is not None, "Tokenizer must have CLS/SEP"

        tok_tss = torch.tensor(ids_tss, dtype=torch.long)
        tok_variant = torch.tensor(ids_variant, dtype=torch.long)
        try:
            seq_input_ids  = torch.cat([tok_tss.new_tensor([cls_id]), tok_variant, tok_tss.new_tensor([gap_id]) ,tok_tss, tok_tss.new_tensor([sep_id])])
        except:
            print(tok_tss.new_tensor([cls_id]).shape, tok_variant.shape, tok_tss.new_tensor([gap_id]).shape,  tok_tss.shape, tok_tss.new_tensor([sep_id]).shape)
            exit(1)

        assert seq_input_ids.shape[0] <= MAX_TOKENS, (
            f"Token length {seq_input_ids.shape[0]} exceeds {MAX_TOKENS}"   
        )
        
        seq_attn_mask  = torch.ones(seq_input_ids.size(0), dtype=torch.long)
        seq_token_type = torch.zeros(seq_input_ids.size(0), dtype=torch.long)
        return seq_input_ids, seq_attn_mask, seq_token_type

    def reverse_complement(self, sequence):
        complement = str.maketrans('ACGTN', 'TGCAN')
        return sequence.translate(complement)[::-1]

    def tokenize_genome(self, genome, chrom_raw, start, reverse):
        token_lengths = []
        num_before = MAX_TOKENS//2
        
        sequences = FastaFile(genome)
        chrom = _resolve_chrom(chrom_raw, sequences)
        if not reverse:
            sequence = sequences.fetch(chrom, max(start - num_before * self.token_len_for_fetch, 0), start).upper()
        else:
            chrom_length = sequences.get_reference_length(chrom)
            sequence = sequences.fetch(chrom, start, min(start + num_before * self.token_len_for_fetch, chrom_length)).upper()
            sequence = self.reverse_complement(sequence)


        encoded_sequence = self.gen_tokenizer.encode_plus(sequence, return_offsets_mapping=True)
        encoded_sequence['input_ids'] = encoded_sequence['input_ids'][1:-1]
        encoded_sequence['offset_mapping'] = encoded_sequence['offset_mapping'][1:-1]
        if len(encoded_sequence['input_ids']) < num_before:
            self.logger.warning(f"Trying to tokenize seq before TSS, but it's too short: {len(encoded_sequence['input_ids'])} < {num_before}; {chrom}: {start}")
        tokens_before = encoded_sequence['input_ids'][-num_before:]
        mapping = encoded_sequence['offset_mapping'][-num_before:]
        
        for i, (start_i, end_i) in enumerate(mapping):
            token_id = tokens_before[i]
            if (token_id == 5):
                if i > 0:
                    length = end_i - mapping[i-1][1] 
                else:
                    length = end_i
            else:
                length = end_i - start_i  
            token = self.gen_tokenizer.decode([token_id])  
            token_lengths.append((token_id, token, length))

        if not reverse:
            start_gene = start - sum(t[2] for t in token_lengths)
        else:
            start_gene = start + num_before * self.token_len_for_fetch

        if not reverse:
            sequence = sequences.fetch(chrom, start, start + num_before * self.token_len_for_fetch).upper()
        else:
            sequence = sequences.fetch(chrom, start - num_before * self.token_len_for_fetch, start).upper()
            sequence = self.reverse_complement(sequence)
        
        encoded_sequence = self.gen_tokenizer.encode_plus(sequence, return_offsets_mapping=True)
        tokens_before = encoded_sequence['input_ids'][1:-1]
        mapping = encoded_sequence['offset_mapping'][1:-1]
        for i, (start_i, end_i) in enumerate(mapping):
            token_id = tokens_before[i]
            if (token_id == 5):
                if i > 0:
                    length = end_i - mapping[i-1][1] 
                else:
                    length = end_i
            else:
                length = end_i - start_i 
            token = self.gen_tokenizer.decode([token_id])  
            token_lengths.append((token_id, token, length))
            
        if reverse: 
            token_lengths.reverse()
        token_lengths_df = pd.DataFrame(token_lengths, columns=['token_id', 'token', 'length'])
        token_lengths_df['start'] = token_lengths_df['length'].cumsum().shift(fill_value=0) + start_gene 
        token_lengths_df['end'] = token_lengths_df['start'] + token_lengths_df['length']
        token_lengths_df['chrom'] = chrom
        if reverse: 
            token_lengths_df = token_lengths_df[::-1].reset_index(drop=True)
        return token_lengths_df

    def __getitem__(self, idx: int) -> dict:
        row       = self.variants_df.iloc[idx]
        gene_id   = str(row["gene_id"])
        chrom_raw = str(row["chrom"])
        variant_pos = int(row["variant_pos_1based"])
        ref_base = str(row["ref_base"])
        alt_base = str(row["alt_base"])

        return_type = row["type"]

        #print('Started', row['gene_id'], row['signal_id'])
        
        tss_info = self.tss_lookup.get(gene_id)
        if tss_info is None:
            self.logger.warning("No TSS info for gene %s; using Ns for TSS region.", gene_id)
            tss_pos = None
            use_complete = False
            sequence_type = "split"
        else:
            tss_pos = tss_info["TSS"]
            try:
                use_complete = self._estimate_sequence_will_fit(chrom_raw, variant_pos, tss_pos)
            except Exception as exc:
                self.logger.warning(
                    "Could not estimate variant-TSS token distance for %s @ %s:%d: %s",
                    gene_id, chrom_raw, variant_pos, exc,
                )
                use_complete = False
            sequence_type = "complete" if use_complete else "split"

        # The three logical records are still ref / alt / tss_only. The new
        # `sequence_type` says whether ref/alt were obtained from one complete
        # context (variant + TSS together) or from the legacy split construction.
        if use_complete and return_type in ("ref", "alt"):
            try:
                ref_seq, alt_seq = self._fetch_complete_ref_alt(
                    chrom_raw=chrom_raw,
                    tss_pos=tss_pos,
                    variant_pos=variant_pos,
                    ref_base=ref_base,
                    alt_base=alt_base,
                    reverse=tss_info["reverse"],
                )
                # Both alleles must fit in the complete construction. We do
                # not silently truncate an allele because that would turn an
                # "uncut" sequence into a cut sequence while keeping the flag.
                ref_ids = np.asarray(
                    self.gen_tokenizer.encode_plus(ref_seq)["input_ids"][1:-1],
                    dtype=np.int32,
                )
                alt_ids = np.asarray(
                    self.gen_tokenizer.encode_plus(alt_seq)["input_ids"][1:-1],
                    dtype=np.int32,
                )
                if len(ref_ids) > MAX_CONTENT_TOKENS or len(alt_ids) > MAX_CONTENT_TOKENS:
                    raise ValueError(
                        f"Complete ref/alt sequence exceeds {MAX_CONTENT_TOKENS} content tokens: "
                        f"ref={len(ref_ids)}, alt={len(alt_ids)}"
                    )
                ids_complete = ref_ids if return_type == "ref" else alt_ids
                seq_input_ids = torch.tensor(
                    np.concatenate([
                        np.asarray([self.gen_tokenizer.cls_token_id]),
                        ids_complete,
                        np.asarray([self.gen_tokenizer.sep_token_id]),
                    ]),
                    dtype=torch.long,
                )
                if seq_input_ids.shape[0] > MAX_TOKENS:
                    raise ValueError(
                        f"Complete sequence has {seq_input_ids.shape[0]} tokens, exceeds {MAX_TOKENS}"
                    )
                seq_attn_mask = torch.ones(seq_input_ids.size(0), dtype=torch.long)
                seq_token_type = torch.zeros(seq_input_ids.size(0), dtype=torch.long)
            except Exception as exc:
                self.logger.warning(
                    "Falling back to split sequence for %s @ %s:%d: %s",
                    gene_id, chrom_raw, variant_pos, exc,
                )
                use_complete = False
                sequence_type = "split"

        if not use_complete and return_type in ("ref", "alt"):
            # --- Legacy variant-centred sequence (3 kbp) ---
            try:
                ref_seq, alt_seq = self._fetch_seq_ref_alt(
                    chrom_raw, variant_pos, VARIANT_REGION_BP, ref_base, alt_base
                )
            except Exception as exc:
                self.logger.error(
                    "Failed to fetch variant sequence for %s @ %s:%d: %s",
                    gene_id, chrom_raw, variant_pos, exc,
                )
                ref_seq = "N" * VARIANT_REGION_BP
                alt_seq = "N" * VARIANT_REGION_BP

            if return_type == "ref":
                ids_variant = self._tokenize_seq(ref_seq, VARIANT_REGION_TOKENS)["token_id"].values.astype(np.int32)
            else:
                ids_variant = self._tokenize_seq(alt_seq, VARIANT_REGION_TOKENS)["token_id"].values.astype(np.int32)

            ids_variant = ids_variant[: min(ids_variant.shape[0], VARIANT_REGION_TOKENS)]

            # --- TSS-centred sequence (legacy split construction) ---
            if tss_info is None:
                tokens_df = self._tokenize_seq("N" * TSS_REGION_BP, MAX_CONTENT_TOKENS)
            else:
                tokens_df = self.tokenize_genome(
                    self.genome_genes, chrom_raw, tss_pos, tss_info["reverse"]
                )
            ids_tss = tokens_df["token_id"].values.astype(np.int32)
            L = min(ids_tss.shape[0], MAX_CONTENT_TOKENS)
            ids_tss = ids_tss[:L]
            ids_tss = ids_tss[VARIANT_REGION_TOKENS:]
            seq_input_ids, seq_attn_mask, seq_token_type = self.wrap_seq(ids_variant, ids_tss)

        elif return_type == "tss_only":
            # TSS-only remains unchanged; its associated variant is only used
            # to identify which variant/gene row the prediction belongs to.
            if tss_info is None:
                tokens_df = self._tokenize_seq("N" * TSS_REGION_BP, MAX_CONTENT_TOKENS)
            else:
                tokens_df = self.tokenize_genome(
                    self.genome_genes, chrom_raw, tss_pos, tss_info["reverse"]
                )
            ids_tss = tokens_df["token_id"].values.astype(np.int32)
            L = min(ids_tss.shape[0], MAX_CONTENT_TOKENS)
            ids_tss = ids_tss[:L]

            cls_id = self.gen_tokenizer.cls_token_id
            sep_id = self.gen_tokenizer.sep_token_id
            assert cls_id is not None and sep_id is not None, "Tokenizer must have CLS/SEP"
            tok_tss = torch.tensor(ids_tss, dtype=torch.long)
            seq_input_ids = torch.cat([
                tok_tss.new_tensor([cls_id]), tok_tss, tok_tss.new_tensor([sep_id])
            ])
            assert seq_input_ids.shape[0] <= MAX_TOKENS, (
                f"Token length {seq_input_ids.shape[0]} exceeds {MAX_TOKENS}"
            )
            seq_attn_mask = torch.ones(seq_input_ids.size(0), dtype=torch.long)
            seq_token_type = torch.zeros(seq_input_ids.size(0), dtype=torch.long)

        # --- Text description ---
        tokenized_desc = self.text_tokenizer(
            [self.description],
            truncation=True,
            padding="max_length",
            max_length=510,
            return_tensors="pt",
        )
        desc_ids = tokenized_desc["input_ids"].reshape(1,  tokenized_desc["input_ids"].shape[1])
        desc_msk = tokenized_desc["attention_mask"].reshape(1,  tokenized_desc["attention_mask"].shape[1])

        batch_input_ids   = seq_input_ids.unsqueeze(0).expand(self.n_keys, -1)
        batch_attn_mask   = seq_attn_mask.unsqueeze(0).expand(self.n_keys, -1)
        return {
            "gene_id":           gene_id,
            "signal_id":         row["signal_id"],
            "variant_pos": variant_pos,
            "ref_base":ref_base,
            "alt_base":alt_base,
            "input_ids":         batch_input_ids,
            "attention_mask":    batch_attn_mask,
            "dataset_flag":      torch.ones(self.n_keys, dtype=torch.float32),
            "desc_input_ids":    desc_ids,
            "desc_attention_mask": desc_msk,
            "type":             return_type,
            "sequence_type":    sequence_type if return_type != "tss_only" else sequence_type,
        }


# ------------------------------------------------------------------
# Module-level helpers (not dataset methods so they can be tested alone)
# ------------------------------------------------------------------

def _centre_crop(ids: list[int], n: int, shift: int = 0) -> list[int]:
    """Return the central `n` elements of `ids`."""
    start = len(ids) // 2 - n // 2 + shift
    return ids[start: start + n]


def _resolve_chrom(chrom_raw: str, fasta: FastaFile) -> str:
    """
    Try both 'chr7' and '7' forms and return whichever exists in the FASTA.
    Raises ValueError if neither is present.
    """
    references = set(fasta.references)
    if chrom_raw in references:
        return chrom_raw
    # try adding / removing the 'chr' prefix
    alt = chrom_raw[3:] if chrom_raw.startswith("chr") else f"chr{chrom_raw}"
    if alt in references:
        return alt
    raise ValueError(
        f"Chromosome '{chrom_raw}' (also tried '{alt}') not found in FASTA references."
    )


# ------------------------------------------------------------------
# CollateFn — unchanged from original except for signal_id in special_keys
# ------------------------------------------------------------------

class CollateFn:
    tokenizer = None
    text_tokenizer = None

    @staticmethod
    def _pad_1d(x, length, pad_value, pad_left=False):
        pad_len = length - x.size(0)
        if pad_len <= 0:
            return x
        pad = x.new_full((pad_len,), pad_value)
        return torch.cat([pad, x] if pad_left else [x, pad], dim=0)

    @staticmethod
    def _pad_2d(x, max_len, pad_value, dim=0):
        pad_len = max_len - x.size(dim)
        if pad_len <= 0:
            return x
        pad_shape = list(x.shape); pad_shape[dim] = pad_len
        return torch.cat([x, x.new_full(tuple(pad_shape), pad_value)], dim=dim)

    @classmethod
    def collate_fn(cls, batch):
        pad_keys     = ['input_ids', 'attention_mask']
        no_pad_keys  = ['dataset_flag']
        special_keys = ['gene_id', 'signal_id', 'type', 'sequence_type', 'variant_pos', 'ref_base', 'alt_base']

        pad_token_ids = {
            'input_ids':           cls.tokenizer.pad_token_id,
            'attention_mask':      0,
            'desc_input_ids':      cls.text_tokenizer.pad_token_id,
            'desc_attention_mask': 0,
        }

        max_seq_len = 8192
        n_keys = len(batch[0]['desc_input_ids'])
        max_text_seq_len = max(
            ids.size(0)
            for sample in batch
            for ids in sample['desc_input_ids']
        ) or 1

        batch_dict = {k: [] for k in pad_keys + no_pad_keys + special_keys}
        desc_ids_batch, desc_mask_batch = [], []

        for sample in batch:
            sample_ids, sample_masks = [], []
            for k in range(n_keys):
                ids  = CollateFn._pad_1d(sample['desc_input_ids'][k],
                                         max_text_seq_len,
                                         pad_token_ids['desc_input_ids'], pad_left=True)
                mask = CollateFn._pad_1d(sample['desc_attention_mask'][k],
                                         max_text_seq_len,
                                         pad_token_ids['desc_attention_mask'], pad_left=True)
                sample_ids.append(ids); sample_masks.append(mask)
            desc_ids_batch.append(torch.stack(sample_ids))
            desc_mask_batch.append(torch.stack(sample_masks))

        for sample in batch:
            for key in pad_keys:
                x = CollateFn._pad_2d(sample[key], max_seq_len,
                                       pad_token_ids[key], dim=1)
                batch_dict[key].append(x)
            for key in no_pad_keys:
                if key in sample:
                    batch_dict[key].append(sample[key])
            for key in special_keys:
                if key in sample:
                    batch_dict[key].append(sample[key])

        for key in pad_keys:
            batch_dict[key] = torch.stack(batch_dict[key], dim=0)
        for key in no_pad_keys:
            if batch_dict[key]:
                batch_dict[key] = torch.stack(batch_dict[key], dim=0)

        batch_dict['desc_input_ids']      = torch.stack(desc_ids_batch)
        batch_dict['desc_attention_mask'] = torch.stack(desc_mask_batch)
        return batch_dict


# ------------------------------------------------------------------
# Inference loop
# ------------------------------------------------------------------

def _variants_fingerprint(variants_df: pd.DataFrame) -> str:
    """Return a stable fingerprint for the exact inference rows/order."""
    hashed = pd.util.hash_pandas_object(variants_df.reset_index(drop=True), index=True)
    return hashlib.sha256(hashed.values.tobytes()).hexdigest()


def _cache_signature(args, variants_df: pd.DataFrame, total_samples: int) -> dict:
    """Parameters that must match before a checkpoint can be resumed."""
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint_stat = checkpoint_path.stat()

    return {
        "cache_version": 1,
        "variants_fingerprint": _variants_fingerprint(variants_df),
        "total_samples": total_samples,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "description_path": str(Path(args.description_path).expanduser().resolve()),
        "config": str(Path(args.config).expanduser().resolve()),
        "b37_fasta": str(Path(args.b37_fasta).expanduser().resolve()),
        "hg38_fasta": str(Path(args.hg38_fasta).expanduser().resolve()),
        "gene_tss_tsv": str(Path(args.gene_tss_tsv).expanduser().resolve()),
    }


def _atomic_pickle_dump(obj, path: Path) -> None:
    """Atomically replace a checkpoint so a killed node cannot leave a partial cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _load_inference_cache(cache_path: Path, signature: dict) -> tuple[int, dict]:
    """Load a checkpoint and validate that it belongs to this exact inference run."""
    if not cache_path.exists():
        return 0, {}

    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)

        if cache.get("signature") != signature:
            logging.warning(
                "Inference cache exists but does not match the current run; "
                "starting from the beginning: %s",
                cache_path,
            )
            return 0, {}

        next_sample_idx = int(cache["next_sample_idx"])
        results = cache["results"]

        if next_sample_idx < 0 or next_sample_idx > signature["total_samples"]:
            raise ValueError(f"Invalid next_sample_idx={next_sample_idx}")

        logging.info(
            "Resuming from inference cache: %d/%d samples (%.1f%%)",
            next_sample_idx,
            signature["total_samples"],
            100.0 * next_sample_idx / max(signature["total_samples"], 1),
        )
        return next_sample_idx, results

    except Exception as exc:
        # Never destroy an older checkpoint just because the newest one is bad.
        logging.warning(
            "Could not load inference cache %s (%s); starting from the beginning.",
            cache_path,
            exc,
        )
        return 0, {}


def _save_inference_cache(
    cache_path: Path,
    signature: dict,
    next_sample_idx: int,
    results: dict,
) -> None:
    """Save a cumulative checkpoint atomically."""
    payload = {
        "signature": signature,
        "next_sample_idx": next_sample_idx,
        "results": results,
    }
    _atomic_pickle_dump(payload, cache_path)


def run_inference(
    model,
    dataloader,
    device,
    *,
    cache_path: str | Path | None = None,
    cache_signature: dict | None = None,
    resume: bool = True,
) -> dict[tuple[str, str], dict[str, float]]:
    """
    Run inference with a cumulative checkpoint every 10% of the input.

    The checkpoint contains all predictions obtained so far and the index of
    the first unprocessed sample. It is written atomically, so a node failure
    during a checkpoint write cannot corrupt the previous checkpoint.

    On restart, only samples after the latest checkpoint are loaded into the
    DataLoader. Therefore a failed job repeats at most the current 10% chunk
    (and, more precisely, at most the current DataLoader batch beyond the
    last checkpoint).
    """
    total_samples = len(dataloader.dataset)
    results = {}
    start_idx = 0

    if cache_path is not None and cache_signature is not None and resume:
        start_idx, results = _load_inference_cache(Path(cache_path), cache_signature)

    if start_idx >= total_samples:
        logging.info("Inference is already complete according to cache.")
        return results

    # The caller passes a DataLoader over the full dataset.  Rebuild it over
    # the remaining contiguous samples so skipped samples are NOT evaluated
    # by Dataset.__getitem__ (important because sequence construction is
    # expensive).
    if start_idx > 0:
        remaining_dataset = torch.utils.data.Subset(
            dataloader.dataset,
            range(start_idx, total_samples),
        )
        dataloader = DataLoader(
            remaining_dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=getattr(dataloader, "pin_memory", False),
        )

    # Checkpoints are based on the original dataset, not on the remaining data.
    # Move to the first 10% boundary strictly after the resumed position.
    checkpoint_boundaries = [
        math.ceil(total_samples * pct / 10)
        for pct in range(1, 11)
    ]
    checkpoint_boundaries = [x for x in checkpoint_boundaries if x > start_idx]
    next_checkpoint_pos = 0

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    )

    processed = start_idx

    with autocast_context, torch.no_grad():
        for batch in tqdm(
            dataloader,
            total=math.ceil((total_samples - start_idx) / dataloader.batch_size),
            desc=f"Inference ({processed:,}/{total_samples:,})",
        ):
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                desc_input_ids=batch["desc_input_ids"].to(device),
                desc_attention_mask=batch["desc_attention_mask"].to(device),
                dataset_flag=batch["dataset_flag"].to(device),
            )

            logits = output["logits"][:, 0, 0].float().cpu().tolist()

            for signal_id, gene_id, variant_pos, ref_base, alt_base, val, type, sequence_type in zip(
                batch["signal_id"],
                batch["gene_id"],
                batch["variant_pos"],
                batch["ref_base"],
                batch["alt_base"],
                logits,
                batch["type"],
                batch["sequence_type"],
            ):
                column_name = f"predicted_expression_{type}"
                key = (signal_id, gene_id, variant_pos, ref_base, alt_base)

                if key in results:
                    results[key][column_name] = val
                    if type in ("ref", "alt"):
                        results[key]["sequence_type"] = sequence_type
                else:
                    results[key] = {
                        column_name: val,
                        "sequence_type": sequence_type if type in ("ref", "alt") else None,
                    }

            processed += len(batch["signal_id"])

            # A batch may cross one or more 10% boundaries. Save only after
            # the whole batch has completed, so the cache always represents
            # a complete prefix of the dataset.
            crossed = [
                boundary
                for boundary in checkpoint_boundaries
                if processed >= boundary
            ]
            if crossed and cache_path is not None and cache_signature is not None:
                # Save the actual completed prefix, not merely the nominal
                # 10%% boundary. A batch can cross a boundary, and all samples
                # in that batch have already been processed.
                checkpoint_idx = processed
                _save_inference_cache(
                    Path(cache_path),
                    cache_signature,
                    checkpoint_idx,
                    results,
                )
                logging.info(
                    "Saved inference checkpoint: %d/%d samples (%.1f%%) -> %s",
                    checkpoint_idx,
                    total_samples,
                    100.0 * checkpoint_idx / max(total_samples, 1),
                    cache_path,
                )
                checkpoint_boundaries = [
                    boundary
                    for boundary in checkpoint_boundaries
                    if boundary > processed
                ]

    # Do not need the cache after a successful complete run.
    if cache_path is not None and cache_signature is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cache_path.unlink()
            logging.info("Removed completed inference cache: %s", cache_path)

    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-tss-tsv",      required=True,
                        help="TSV with gene_id, TSS columns")
    parser.add_argument("--variants-csv",      required=True,
                        help="Credible-set variants CSV (as described in the docstring)")
    parser.add_argument("--b37-fasta",         required=True)
    parser.add_argument("--hg38-fasta",         required=True)
    parser.add_argument("--out",               required=True, help="Output CSV path")
    parser.add_argument(
        "--cache",
        default=None,
        help=(
            "Path to the inference checkpoint. Defaults to '<out>.cache.pkl'. "
            "The cache is updated every 10%% and used automatically on restart."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing inference cache and start from the beginning.",
    )
    parser.add_argument("--config",            default="notebooks/inference.yaml")
    parser.add_argument("--gena-lm-home",      required=True)
    parser.add_argument("--checkpoint",        required=True)
    parser.add_argument("--description-path",  required=True)
    parser.add_argument("--batch-size",        type=int, default=8)
    parser.add_argument("--device",            default="auto")
    parser.add_argument("--no-amp",            action="store_true")
    args = parser.parse_args()

    os.environ["GENALM_HOME"] = str(Path(args.gena_lm_home).expanduser().resolve())
    local_rank  = 0
    global_rank = 0
    world_size  = 1
    device = torch.device(args.device)

    experiment_config_path = Path(args.config).expanduser().absolute()
    with initialize_config_dir(str(experiment_config_path.parent)):
        experiment_config = compose(config_name=experiment_config_path.name)

    model_kwargs = instantiate(experiment_config["model_kwargs"])
    model = ExpressionCounts(**model_kwargs)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device).eval()
    

    dna_tok  = AutoTokenizer.from_pretrained(experiment_config["args_params"]["gen_tokenizer"])
    text_tok = AutoTokenizer.from_pretrained(
        experiment_config["shared_dataset_params"]["text_tokenizer"], padding_side="left"
    )
    CollateFn.tokenizer      = dna_tok
    CollateFn.text_tokenizer = text_tok

    description_text = load_description_text(args.description_path)

    print(f"Loading variants table from {args.variants_csv} …", flush=True)
    variants_df = load_variants_table(args.variants_csv)
    print(f"  → {len(variants_df):,} (signal, gene) pairs to score", flush=True)

    dataset = DualRegionInferenceDataset(
        genome_variants= args.b37_fasta,
        genome_genes   = args.hg38_fasta,
        dna_tokenizer  = dna_tok,
        text_tokenizer = text_tok,
        description    = description_text,
        genes_path     = args.gene_tss_tsv,
        variants_df    = variants_df,
    )


    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size, # per-GPU batch size
        shuffle     = False,
        num_workers = 1,                 # can increase with DDP
        collate_fn  = CollateFn.collate_fn,
        pin_memory  = True,
    )

    cache_path = Path(args.cache) if args.cache else Path(f"{args.out}.cache.pkl")
    cache_signature = _cache_signature(args, dataset.variants_df, len(dataset))

    print(f"Running inference …", flush=True)
    print(f"Checkpoint cache: {cache_path}", flush=True)
    predictions = run_inference(
        model,
        dataloader,
        device,
        cache_path=cache_path,
        cache_signature=cache_signature,
        resume=not args.no_resume,
    )


    # --- Save (rank 0 only) ---
    if global_rank == 0:
        rows = [
            {
                "common_variant_analysis_signal_id": sig,
                "gene_id": gene,
                "variant_pos": variant_pos,
                "ref_base": ref_base,
                "alt_base": alt_base,
                "sequence_type": vals.get("sequence_type", "split"),
                **vals,
            }
            for (sig, gene, variant_pos, ref_base, alt_base), vals in predictions.items()
        ]
        output_df = pd.DataFrame(rows)
        output_df.to_csv(args.out, index=False)
        complete_count = int((output_df["sequence_type"] == "complete").sum()) if not output_df.empty else 0
        split_count = int((output_df["sequence_type"] == "split").sum()) if not output_df.empty else 0
        print(f"Saved {len(rows):,} predictions → {args.out}", flush=True)
        print(f"Complete (uncut) variant+TSS sequences: {complete_count:,}", flush=True)
        print(f"Split variant/TSS sequences: {split_count:,}", flush=True)

if __name__ == "__main__":
    main()
