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
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from downstream_tasks.expression_prediction.expression_model_final import ExpressionCounts
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import math
from torch.utils.data import Sampler
import torch.distributed as dist

# Target bp window sizes for each region
VARIANT_REGION_BP  = 3_000   # 3 kbp centered on variant
TSS_REGION_BP      = 12_000   # 12 kbp centered on TSS
MAX_TOKENS         = 1_024   # hard model limit including CLS + SEP
MAX_CONTENT_TOKENS = MAX_TOKENS - 2   # 1022 usable token slots
# Approximate token/bp ratio for the GENA-LM tokenizer (used only for initial fetch sizing)
APPROX_BP_PER_TOKEN = 6


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
        clean_k = re.sub(r'\[|\]', '', clean_k).strip()
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
                if gene_id:
                    records.append({
                        "signal_id":        signal_id,
                        "chrom":            chrom_raw,
                        "variant_pos_1based": variant_pos,
                        "ref_base":     ref_base,
                        "alt_base":     alt_base,
                        "gene_id":          gene_id,
                    })
    
    result_df = pd.DataFrame(records)
    result_df = result_df.drop_duplicates(subset=['gene_id'], keep='first')
    return result_df

class PairedDistributedSampler(Sampler):
    """
    A distributed sampler that ensures consecutive pairs of indices (Ref and Alt)
    are always assigned to the exact same process rank.
    """
    def __init__(self, dataset, num_replicas=None, rank=None, drop_last=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        
        # Total number of pairs in the dataset
        assert len(self.dataset) % 2 == 0, "Dataset length must be even for paired tracking."
        self.num_pairs = len(self.dataset) // 2
        
        # Calculate how many pairs this replica gets
        if self.drop_last:
            self.num_samples_per_replica = math.floor(self.num_pairs / self.num_replicas) * 2
            self.total_size = math.floor(self.num_pairs / self.num_replicas) * self.num_replicas * 2
        else:
            self.num_samples_per_replica = math.ceil(self.num_pairs / self.num_replicas) * 2
            self.total_size = math.ceil(self.num_pairs / self.num_replicas) * self.num_replicas * 2

    def __iter__(self):
        # Generate indices group pairs: [[0, 1], [2, 3], [4, 5], ...]
        indices = list(range(len(self.dataset)))
        pairs = [indices[i:i+2] for i in range(0, len(indices), 2)]
        
        # Deterministic padding if total size doesn't distribute evenly
        if not self.drop_last:
            padding_size = (self.total_size // 2) - len(pairs)
            if padding_size > 0:
                pairs += pairs[:padding_size]
        else:
            pairs = pairs[:self.total_size // 2]
            
        assert len(pairs) * 2 == self.total_size

        # Slice out the pairs allocated to THIS rank
        # rank 0 gets pairs[0], pairs[num_replicas], pairs[2*num_replicas]...
        sub_pairs = pairs[self.rank::self.num_replicas]
        
        # Flatten the list of pairs back to linear indices
        final_indices = [idx for pair in sub_pairs for idx in pair]
        return iter(final_indices)

    def __len__(self):
        return self.num_samples_per_replica


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
        genome: str | Path,
        dna_tokenizer,
        text_tokenizer,
        description: str,
        genes_path: str,
        variants_df: pd.DataFrame,
        seed: int = 42,
        token_len_for_fetch = 10
    ):
        self.genome       = str(genome)
        self.gen_tokenizer  = dna_tokenizer
        self.text_tokenizer = text_tokenizer
        self.description  = description
        self.logger = logging.getLogger(__name__)
        self.token_len_for_fetch = token_len_for_fetch
        # Gene TSS lookup: gene_id_unversioned → row
        self.genes_df = pd.read_csv(genes_path)
        self.genes_df = self.genes_df.dropna()

        self.tss_lookup: dict[str, dict] = {}
        for _, row in self.genes_df.iterrows():
            gid = row["gene_id"]
            self.tss_lookup[gid] = {
                "TSS":        int(row["TSS_B37"]),
            }

        self.variants_df = variants_df[variants_df["gene_id"].isin(self.tss_lookup)].reset_index(drop=True)

        self.n_keys = 1  # ref and alt sequences for the variant region
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _fetch_seq(self, chrom: str, center_1based: int, window_bp: int) -> str:
        """Fetch `window_bp` bases centered on `center_1based` (1-based coordinate).
        Clamps to chromosome boundaries and pads with Ns if needed."""
        sequences = FastaFile(self.genome)
        chrom_len = sequences.get_reference_length(chrom)
        half = window_bp // 2
        # convert to 0-based half-open [start, end)
        start = max(0, center_1based - 1 - half)
        end   = min(chrom_len, center_1based - 1 + half)
        seq = sequences.fetch(chrom, start, end).upper()

        return seq
    
    def _fetch_seq_ref_alt(self, chrom: str, center_1based: int, window_bp: int, ref_base: str, alt_base: str) -> str:
        """Fetch `window_bp` bases centered on `center_1based` (1-based coordinate).
        Clamps to chromosome boundaries and pads with Ns if needed."""
        sequences = FastaFile(self.genome)
        chrom_len = sequences.get_reference_length(chrom)
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

    def _tokenize_seq(self, seq: str) -> list[int]:
        """Tokenize a DNA sequence, stripping CLS/SEP added by the tokenizer."""
        enc = self.gen_tokenizer.encode_plus(seq)
        ids = enc["input_ids"]
        # strip leading CLS and trailing SEP if present
        if ids and ids[0]  == self.gen_tokenizer.cls_token_id:
            ids = ids[1:]
        if ids and ids[-1] == self.gen_tokenizer.sep_token_id:
            ids = ids[:-1]
        return ids

    def _proportional_truncate(
        self, ids_variant: list[int], ids_tss: list[int]
    ) -> tuple[list[int], list[int]]:
        """
        Proportionally shorten the two token lists so their combined length
        does not exceed MAX_CONTENT_TOKENS (1022).

        The target split mirrors the requested bp ratio (3 kbp : 6 kbp = 1 : 2),
        meaning the variant sequence gets at most 1/3 of the budget and the TSS
        sequence at most 2/3. Each is then capped at its own actual length so
        shorter sequences do not inflate the other's slice.
        """
        total = len(ids_variant) + len(ids_tss)
        if total <= MAX_CONTENT_TOKENS:
            return ids_variant, ids_tss

        budget_variant = 30
        budget_tss     = MAX_CONTENT_TOKENS - budget_variant



        # Take tokens from the *centre* of each sequence so the anchor point
        # (variant / TSS) stays as close to the middle as possible
        ids_variant = _centre_crop(ids_variant, budget_variant)
        ids_tss     = _centre_crop(ids_tss, budget_tss, budget_variant//2)

        if len(ids_variant) + len(ids_tss) > MAX_CONTENT_TOKENS:
            # floating-point rounding guard: trim one token from the longer list
            if len(ids_tss) > len(ids_variant):
                ids_tss = ids_tss[:len(ids_tss) - 1]
            else:
                ids_variant = ids_variant[:len(ids_variant) - 1]

        return ids_variant, ids_tss

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.variants_df)  #Since we have 2 sequences per one row (ref and alt), we will use "virtual" indexing
    
    def wrap_seq(self, combined_ids):
        # --- Wrap with CLS / SEP ---
        cls_id = self.gen_tokenizer.cls_token_id
        sep_id = self.gen_tokenizer.sep_token_id
        assert cls_id is not None and sep_id is not None, "Tokenizer must have CLS/SEP"

        tok = torch.tensor(combined_ids, dtype=torch.long)
        seq_input_ids  = torch.cat([tok.new_tensor([cls_id]), tok, tok.new_tensor([sep_id])])
        assert seq_input_ids.size(0) <= MAX_TOKENS, (
            f"Token length {seq_input_ids.size(0)} exceeds {MAX_TOKENS}"
        )
        seq_attn_mask  = torch.ones(seq_input_ids.size(0), dtype=torch.long)
        seq_token_type = torch.zeros(seq_input_ids.size(0), dtype=torch.long)
        return seq_input_ids, seq_attn_mask, seq_token_type


    def tokenize_genome(self, chrom, start):
        token_lengths = []
        num_before = MAX_TOKENS//2
        
        sequences = FastaFile(self.genome)
        
        try:
            sequence = sequences.fetch(chrom, max(start - num_before * self.token_len_for_fetch, 0), start).upper()
        except ValueError as e:
            self.logger.error(f"Error sequence {i}")
    
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

        start_gene = start - sum(t[2] for t in token_lengths)
    

        try:
            sequence = sequences.fetch(chrom, start, start + num_before * self.token_len_for_fetch).upper()
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
        token_lengths_df['start'] = token_lengths_df['length'].cumsum().shift(fill_value=0) + start_gene 
        token_lengths_df['end'] = token_lengths_df['start'] + token_lengths_df['length']
        token_lengths_df['chrom'] = chrom

        return token_lengths_df

    def __getitem__(self, idx: int) -> dict:
        #return_ref = idx%2==0 #Even indices for ref, odd for alt
        #idx = idx//2 #Collapse index to thr real one
        return_ref = 0
        row       = self.variants_df.iloc[idx]
        gene_id   = str(row["gene_id"])
        chrom_raw = str(row["chrom"])
        variant_pos = int(row["variant_pos_1based"])
        ref_base = str(row["ref_base"])
        alt_base = str(row["alt_base"])


        sequences = FastaFile(self.genome)
        chrom = _resolve_chrom(chrom_raw, sequences)
        '''
        # --- Variant-centred sequence (3 kbp) ---
        try:
            ref_seq, alt_seq = self._fetch_seq_ref_alt(chrom, variant_pos, VARIANT_REGION_BP, ref_base, alt_base)
        except Exception as exc:
            self.logger.error("Failed to fetch variant sequence for %s @ %s:%d: %s",
                              gene_id, chrom, variant_pos, exc)
            ref_seq = "N" * VARIANT_REGION_BP
            alt_seq = "N" * VARIANT_REGION_BP
        if return_ref:
            ids_variant = self._tokenize_seq(ref_seq)
        else:
            ids_variant = self._tokenize_seq(alt_seq)   
        '''
        # --- TSS-centred sequence (6 kbp) ---
        tss_info = self.tss_lookup.get(gene_id)
        if tss_info is None:
            self.logger.warning("No TSS info for gene %s; using Ns for TSS region.", gene_id)
            ids_tss = self._tokenize_seq("N" * TSS_REGION_BP)
        else:
            tss_pos = tss_info["TSS"]
            '''
            try:
                seq_tss = self._fetch_seq(chrom, tss_pos, TSS_REGION_BP)
            except Exception as exc:
                print(exc)
                self.logger.error("Failed to fetch TSS sequence for %s @ %s:%d: %s",
                                  gene_id, chrom_raw, tss_pos, exc)
                seq_tss = "N" * TSS_REGION_BP
            ids_tss = self._tokenize_seq(seq_tss)
            '''
            tokens_df = self.tokenize_genome(chrom, tss_pos)
        ids_tss = tokens_df["token_id"].values.astype(np.int32)
        L = min(ids_tss.shape[0], MAX_CONTENT_TOKENS)
        ids_tss = ids_tss[:L]

        # --- Proportional truncation then concatenation ---
        #ids_variant, ids_tss = self._proportional_truncate(ids_variant, ids_tss)
        gap_id = self._tokenize_seq("N"*1000)
        pad_id = self.gen_tokenizer.pad_token_id

        #combined_ids = ids_variant + gap_id + ids_tss   # variant first, then gap, then TSS
        #combined_ids =  _centre_crop(ids_tss, 1022)

        # --- Wrap with CLS / SEP ---
        seq_input_ids, seq_attn_mask, seq_token_type = self.wrap_seq(ids_tss)

        # Stack along dim 0 → shape (2, seq_len)
        # key 0 = ref, key 1 = alt
        #batch_input_ids   = torch.stack([seq_input_ids], dim=0)
        #batch_attn_mask   = torch.stack([
        #    (seq_input_ids != pad_id).long(),
        #], dim=0)

        batch_input_ids   = seq_input_ids.unsqueeze(0).expand(self.n_keys, -1)
        batch_attn_mask   = seq_attn_mask.unsqueeze(0).expand(self.n_keys, -1)

        is_ref = torch.tensor(return_ref, dtype=torch.long)
        
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

        return {
            "gene_id":           gene_id,
            "signal_id":         row["signal_id"],
            "input_ids":         batch_input_ids,
            "attention_mask":    batch_attn_mask,
            "dataset_flag":      torch.ones(self.n_keys, dtype=torch.float32),
            "desc_input_ids":    desc_ids,
            "desc_attention_mask": desc_msk,
            "is_ref":             is_ref,
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
        special_keys = ['gene_id', 'signal_id', 'is_ref']

        pad_token_ids = {
            'input_ids':           cls.tokenizer.pad_token_id,
            'attention_mask':      0,
            'desc_input_ids':      cls.text_tokenizer.pad_token_id,
            'desc_attention_mask': 0,
        }

        max_seq_len = 1024
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

def run_inference(model, dataloader, device) -> dict[tuple[str, str], dict[str, float]]:
    """Returns {(signal_id, gene_id): {"predicted_expression_ref": ref_val, "predicted_expression_alt": alt_val}} for every sample."""
    results = {}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
        for batch in dataloader:
            output = model(
                input_ids        = batch["input_ids"].to(device),
                attention_mask   = batch["attention_mask"].to(device),
                desc_input_ids   = batch["desc_input_ids"].to(device),
                desc_attention_mask = batch["desc_attention_mask"].to(device),
                dataset_flag     = batch["dataset_flag"].to(device),
            )
            logits = output["logits"]  # shape (B, n_keys, seq_len, 1) or (B, n_keys, 1)
            # Take the per-sequence summary token (position 0 = CLS)
            logits = output["logits"][:, 0, 0].float().cpu().tolist()
            for signal_id, gene_id, val, is_ref in zip(
                batch["signal_id"], batch["gene_id"], logits, batch["is_ref"]
            ):
                column_name = "predicted_expression_ref" if is_ref.item() == 1 else "predicted_expression_alt"
                if (signal_id, gene_id) in results:
                    results[(signal_id, gene_id)][column_name] = val
                else:
                    results[(signal_id, gene_id)] = {
                        column_name: val
                    }
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
    parser.add_argument("--out",               required=True, help="Output CSV path")
    parser.add_argument("--config",            default="notebooks/inference.yaml")
    parser.add_argument("--gena-lm-home",      required=True)
    parser.add_argument("--checkpoint",        required=True)
    parser.add_argument("--description-path",  required=True)
    parser.add_argument("--batch-size",        type=int, default=8)
    parser.add_argument("--device",            default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-amp",            action="store_true")
    args = parser.parse_args()

    os.environ["GENALM_HOME"] = str(Path(args.gena_lm_home).expanduser().resolve())
    use_ddp = "LOCAL_RANK" in os.environ
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size  = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        local_rank  = 0
        global_rank = 0
        world_size  = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_config_path = Path(args.config).expanduser().absolute()
    with initialize_config_dir(str(experiment_config_path.parent)):
        experiment_config = compose(config_name=experiment_config_path.name)

    model_kwargs = instantiate(experiment_config["model_kwargs"])
    model = ExpressionCounts(**model_kwargs)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device).eval()
    
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )



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
        genome        = args.b37_fasta,
        dna_tokenizer = dna_tok,
        text_tokenizer= text_tok,
        description   = description_text,
        genes_path    = args.gene_tss_tsv,
        variants_df   = variants_df,
    )

    if use_ddp:
        sampler = PairedDistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            drop_last=False,  # Keep stable for full evaluation
        )
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size, # per-GPU batch size
        shuffle     = False,
        num_workers = 6,                 # can increase with DDP
        collate_fn  = CollateFn.collate_fn,
        sampler     = sampler,
        pin_memory  = True,
    )

    print("Running inference …", flush=True)
    predictions = run_inference(model, dataloader, device)

    # --- Gather results from all ranks to rank 0 ---
    if use_ddp:
        # each rank has a dict; convert to list of dicts, all_gather, merge
        all_preds = [None] * world_size
        dist.all_gather_object(all_preds, predictions)
        if global_rank == 0:
            merged = {}
            for d in all_preds:
                merged.update(d)
            predictions = merged
        else:
            predictions = {}   # only rank 0 writes output
    

    # --- Save (rank 0 only) ---
    if global_rank == 0:
        rows = [
            {"common_variant_analysis_signal_id": sig, "gene_id": gene, **vals}
            for (sig, gene), vals in predictions.items()
        ]
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"Saved {len(rows):,} predictions → {args.out}", flush=True)

    if use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()