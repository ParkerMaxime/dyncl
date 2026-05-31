"""
Sparse Transformer seq2seq with dynamic synapses.

Goal:
- provide a Transformer model that keeps the dynamic-synapse API used in 3B/3C
- expose weights/masks/replay tags and dynamic create/prune ops
- keep compatibility with existing mechanism code paths (crystallization, tags, newsyn)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SynapseStats:
    """Connectivity and energy summary."""

    total_possible: int
    total_active: int
    overall_density: float
    energy_consumed: float
    energy_budget: float
    by_connection: Dict[str, Dict[str, float]]


class SparseTransformerSeq2Seq(nn.Module):
    """
    Seq2seq Transformer with sparse dynamic matrices.

    Dynamic interface (compatible with 3B mechanisms):
    - `weights`: ParameterDict of sparse matrices
    - `masks`: mapping matrix key -> mask buffer name
    - `_get_replay_tag`, `_get_usage`, `_get_age`
    - `gradient_guided_synaptogenesis`, `utility_based_pruning_with_protection`
    - `increment_synapse_ages`, `reset_tracking`, `get_connectivity_stats`
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_enc_layers: int = 1,
        n_dec_layers: int = 1,
        max_src_len: int = 160,
        max_tgt_len: int = 30,
        pad_id: int = 0,
        initial_connectivity_attn: float = 0.10,
        initial_connectivity_ffn: float = 0.05,
        dropout: float = 0.0,
        ffn_activation: str = "relu",
        energy_budget: float = 1e9,
        synapse_creation_cost: float = 0.0,
        synapse_maintenance_cost: float = 0.001,
        hebbian_threshold: float = 0.20,
        pruning_utility_threshold: float = 0.01,
        protection_period: int = 15,
        tie_embeddings: bool = False,
        creation_score_mode: str = "legacy",
        creation_score_normalization: str = "batch",
        creation_score_alpha: float = 0.10,
        creation_policy: str = "uniform_quota",
        creation_floor_per_connection: int = 0,
        creation_active_ref_quantile: float = 0.10,
        creation_gap_rel_margin: float = 0.10,
        creation_gap_abs_margin: float = 1e-6,
        creation_quality_mad_factor: float = 1.5,
        creation_coherence_threshold: float = 0.85,
        creation_need_ema_alpha: float = 0.30,
        creation_priority_topk: int = 128,
        creation_ref_mature_only: bool = True,
        creation_group_min_share: float = 0.40,
        creation_group_mix: float = 0.50,
        creation_group_prior_enc_share: float = 0.50,
        creation_group_share_ema_alpha: float = 1.0,
        creation_selection_weight: str = "raw",
        creation_auto_budget_from_role_need: bool = False,
        creation_role_need_ema_alpha: float = 0.20,
        creation_auto_budget_min_ratio: float = 0.25,
        channel_creation_enabled: bool = False,
        channel_fan_in: int = 3,
        channel_fan_out: int = 3,
        channel_bundle_size: int = 6,
        channel_candidate_pool: int = 24,
        channel_bundle_init_scale: float = 0.005 / math.sqrt(6.0),
        channel_underused_max_quantile: float = 0.75,
        channel_require_full_bundle: bool = False,
        block_density_cap: float = 1.0,
        channel_budget_fraction: float = 1.0,
        channel_probe_period: int = 25,
        max_probationary_channels_per_prefix: int = 64,
        max_probationary_channels_global: int = 192,
        channel_probe_threshold_ema_alpha: float = 0.05,
        max_new_probationary_channels_per_epoch: int = 0,
        phase1_registry_enabled: bool = False,
        phase1_reselection_decay: float = 0.5,
        phase1_unexplored_bonus: float = 1.0,
        phase1_reexplored_bonus: float = 0.7,
        tenured_channel_grace_period: int = 0,
        pathway_dropout_enabled: bool = False,
        pathway_dropout_prefixes: Tuple[str, ...] = (),
        pathway_dropout_top_frac: float = 0.10,
        pathway_dropout_rate: float = 0.25,
        pathway_dropout_warmup_epochs: int = 0,
        pathway_dropout_end_epoch: int = 0,
        local_reinforce_enabled: bool = False,
        local_reinforce_epochs: int = 0,
        local_reinforce_max_edges_per_epoch: int = 0,
        local_reinforce_max_edges_per_channel_total: int = 0,
        local_reinforce_edges_per_channel_per_epoch: int = 0,
        local_reinforce_top_pool: int = 16,
        ffn_channel_complete_k: int = 0,
        pruning_policy: str = "global_usage",
        pruning_active_ref_quantile: float = 0.10,
        pruning_priority_topk: int = 128,
        pruning_ref_mature_only: bool = True,
        pruning_couple_creation_need: bool = False,
        pruning_creation_need_ema_alpha: float = 0.20,
        pruning_creation_coupling_alpha: float = 1.0,
        tenure_enabled: bool = False,
        tenure_score_ema_alpha: float = 0.10,
        tenure_min_age: int = 30,
        tenure_acquire_quantile: float = 0.80,
        tenure_acquire_streak: int = 20,
        tenure_max_share_per_matrix: float = 0.20,
        tenure_decay_on_miss: int = 1,
        tenure_release_enabled: bool = False,
        enable_pruning_persistence_diagnostic: bool = False,
        pruning_persistence_long_alpha: float = 0.02,
        pruning_persistence_quantile: float = 0.10,
        enable_pruning_natural_volume_diagnostic: bool = False,
        enable_pruning_relative_median_diagnostic: bool = False,
    ) -> None:
        super().__init__()

        if d_model <= 0 or n_heads <= 0 or d_ff <= 0:
            raise ValueError("d_model, n_heads, d_ff must be > 0.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if vocab_size <= 4:
            raise ValueError("vocab_size too small.")

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = int(d_model // n_heads)
        self.d_ff = int(d_ff)
        self.n_enc_layers = int(n_enc_layers)
        self.n_dec_layers = int(n_dec_layers)
        self.max_src_len = int(max_src_len)
        self.max_tgt_len = int(max_tgt_len)
        self.pad_id = int(pad_id)
        activation = str(ffn_activation).strip().lower()
        if activation not in {"relu", "gelu"}:
            raise ValueError("ffn_activation must be one of: relu, gelu.")
        self.ffn_activation = activation

        # Dynamic parameters / thresholds
        self.energy_budget = float(energy_budget)
        self.creation_cost = float(synapse_creation_cost)
        self.maintenance_cost = float(synapse_maintenance_cost)
        self.hebbian_threshold = float(hebbian_threshold)
        self.pruning_threshold = float(pruning_utility_threshold)
        self.protection_period = int(protection_period)
        score_mode = str(creation_score_mode).strip().lower()
        if score_mode not in {"legacy", "mlp_seq"}:
            raise ValueError("creation_score_mode must be one of: legacy, mlp_seq.")
        self.creation_score_mode = score_mode
        score_norm = str(creation_score_normalization).strip().lower()
        if score_norm not in {"batch", "valid_tokens"}:
            raise ValueError("creation_score_normalization must be one of: batch, valid_tokens.")
        self.creation_score_normalization = score_norm
        self.creation_score_alpha = float(max(1e-4, min(1.0, creation_score_alpha)))
        policy = str(creation_policy).strip().lower()
        if policy not in {
            "uniform_quota",
            "hybrid_global",
            "effect_gap_global",
            "group_balanced_effect_gap",
            "group_calibrated_weighted",
            "group_frustration_weighted",
            "need_quality_filtered",
            "need_quality_inverse_coverage",
            "paired_ffn_channel_random",
            "paired_ffn_channel_probe",
        }:
            raise ValueError(
                "creation_policy must be one of: uniform_quota, hybrid_global, effect_gap_global, group_balanced_effect_gap, group_calibrated_weighted, group_frustration_weighted, need_quality_filtered, need_quality_inverse_coverage, paired_ffn_channel_random, paired_ffn_channel_probe."
            )
        self.creation_policy = policy
        self.creation_floor_per_connection = int(max(0, creation_floor_per_connection))
        self.creation_active_ref_quantile = float(max(0.0, min(1.0, creation_active_ref_quantile)))
        self.creation_gap_rel_margin = float(max(0.0, creation_gap_rel_margin))
        self.creation_gap_abs_margin = float(max(0.0, creation_gap_abs_margin))
        self.creation_quality_mad_factor = float(max(0.0, creation_quality_mad_factor))
        self.creation_coherence_threshold = float(max(0.0, min(1.0, creation_coherence_threshold)))
        self.creation_need_ema_alpha = float(max(1e-6, min(1.0, creation_need_ema_alpha)))
        self.creation_priority_topk = int(max(1, creation_priority_topk))
        self.creation_ref_mature_only = bool(creation_ref_mature_only)
        self.creation_group_min_share = float(max(0.0, min(0.5, creation_group_min_share)))
        self.creation_group_mix = float(max(0.0, min(1.0, creation_group_mix)))
        self.creation_group_prior_enc_share = float(max(0.0, min(1.0, creation_group_prior_enc_share)))
        self.creation_group_share_ema_alpha = float(max(1e-6, min(1.0, creation_group_share_ema_alpha)))
        selection_weight = str(creation_selection_weight).strip().lower()
        if selection_weight not in {"raw", "score_gap", "sqrt_score_gap"}:
            raise ValueError("creation_selection_weight must be one of: raw, score_gap, sqrt_score_gap.")
        self.creation_selection_weight = selection_weight
        self.creation_auto_budget_from_role_need = bool(creation_auto_budget_from_role_need)
        self.creation_role_need_ema_alpha = float(max(1e-6, min(1.0, creation_role_need_ema_alpha)))
        self.creation_auto_budget_min_ratio = float(max(0.0, min(1.0, creation_auto_budget_min_ratio)))
        self.channel_creation_enabled = bool(channel_creation_enabled)
        self.channel_fan_in = int(max(1, channel_fan_in))
        self.channel_fan_out = int(max(1, channel_fan_out))
        self.channel_bundle_size = int(max(1, channel_bundle_size))
        self.channel_candidate_pool = int(max(1, channel_candidate_pool))
        self.channel_bundle_init_scale = float(max(0.0, channel_bundle_init_scale))
        self.channel_underused_max_quantile = float(max(0.0, min(1.0, channel_underused_max_quantile)))
        self.channel_require_full_bundle = bool(channel_require_full_bundle)
        self.block_density_cap = float(max(0.0, min(1.0, block_density_cap)))
        self.channel_budget_fraction = float(max(0.0, min(1.0, channel_budget_fraction)))
        self.channel_probe_period = int(max(1, channel_probe_period))
        self.max_probationary_channels_per_prefix = int(max(1, max_probationary_channels_per_prefix))
        self.max_probationary_channels_global = int(max(1, max_probationary_channels_global))
        self.channel_probe_threshold_ema_alpha = float(
            max(1e-6, min(1.0, channel_probe_threshold_ema_alpha))
        )
        self.max_new_probationary_channels_per_epoch = int(max(0, max_new_probationary_channels_per_epoch))
        self.phase1_registry_enabled = bool(phase1_registry_enabled)
        self.phase1_reselection_decay = float(max(0.0, min(1.0, phase1_reselection_decay)))
        self.phase1_unexplored_bonus = float(max(0.0, phase1_unexplored_bonus))
        self.phase1_reexplored_bonus = float(max(0.0, phase1_reexplored_bonus))
        self.tenured_channel_grace_period = int(max(0, tenured_channel_grace_period))
        self.pathway_dropout_enabled = bool(pathway_dropout_enabled)
        self.pathway_dropout_prefixes = tuple(str(v) for v in pathway_dropout_prefixes)
        self.pathway_dropout_top_frac = float(max(0.0, min(1.0, pathway_dropout_top_frac)))
        self.pathway_dropout_rate = float(max(0.0, min(1.0, pathway_dropout_rate)))
        self.pathway_dropout_warmup_epochs = int(max(0, pathway_dropout_warmup_epochs))
        self.pathway_dropout_end_epoch = int(max(0, pathway_dropout_end_epoch))
        self.local_reinforce_enabled = bool(local_reinforce_enabled)
        self.local_reinforce_epochs = int(max(0, local_reinforce_epochs))
        self.local_reinforce_max_edges_per_epoch = int(max(0, local_reinforce_max_edges_per_epoch))
        self.local_reinforce_max_edges_per_channel_total = int(max(0, local_reinforce_max_edges_per_channel_total))
        self.local_reinforce_edges_per_channel_per_epoch = int(max(0, local_reinforce_edges_per_channel_per_epoch))
        self.local_reinforce_top_pool = int(max(1, local_reinforce_top_pool))
        self.ffn_channel_complete_k = int(max(0, ffn_channel_complete_k))
        pruning_policy = str(pruning_policy).strip().lower()
        if pruning_policy not in {"global_usage", "matrix_survival_mix"}:
            raise ValueError("pruning_policy must be one of: global_usage, matrix_survival_mix.")
        self.pruning_policy = pruning_policy
        self.pruning_active_ref_quantile = float(max(0.0, min(1.0, pruning_active_ref_quantile)))
        self.pruning_priority_topk = int(max(1, pruning_priority_topk))
        self.pruning_ref_mature_only = bool(pruning_ref_mature_only)
        self.pruning_couple_creation_need = bool(pruning_couple_creation_need)
        self.pruning_creation_need_ema_alpha = float(max(1e-6, min(1.0, pruning_creation_need_ema_alpha)))
        self.pruning_creation_coupling_alpha = float(max(0.0, pruning_creation_coupling_alpha))
        self.tenure_enabled = bool(tenure_enabled)
        self.tenure_score_ema_alpha = float(max(1e-6, min(1.0, tenure_score_ema_alpha)))
        self.tenure_min_age = int(max(1, tenure_min_age))
        self.tenure_acquire_quantile = float(max(0.0, min(1.0, tenure_acquire_quantile)))
        self.tenure_acquire_streak = int(max(1, tenure_acquire_streak))
        self.tenure_max_share_per_matrix = float(max(0.0, min(1.0, tenure_max_share_per_matrix)))
        self.tenure_decay_on_miss = int(max(0, tenure_decay_on_miss))
        self.tenure_release_enabled = bool(tenure_release_enabled)
        self.enable_pruning_persistence_diagnostic = bool(enable_pruning_persistence_diagnostic)
        self.pruning_persistence_long_alpha = float(max(1e-6, min(1.0, pruning_persistence_long_alpha)))
        self.pruning_persistence_quantile = float(max(0.0, min(1.0, pruning_persistence_quantile)))
        self.enable_pruning_natural_volume_diagnostic = bool(enable_pruning_natural_volume_diagnostic)
        self.enable_pruning_relative_median_diagnostic = bool(enable_pruning_relative_median_diagnostic)
        self._pruning_dead_thresholds: Tuple[Tuple[str, float], ...] = (
            ("1e5", 1e-5),
            ("1e4", 1e-4),
            ("5e4", 5e-4),
            ("1e3", 1e-3),
            ("5e3", 5e-3),
        )
        self._creation_group_share_ema: Dict[str, Optional[float]] = {"enc": None, "dec": None}
        self._creation_role_need_ema: Dict[str, Optional[float]] = {"enc": None, "dec": None, "other": None}
        self._creation_need_ema: Dict[str, Optional[float]] = {}
        self._pruning_creation_need_ema: Dict[str, Optional[float]] = {}
        self.last_pruning_stats: Dict[str, float] = {
            "policy": self.pruning_policy,
            "survival_score_mode": "usage_mean",
            "pruned_tagged": 0.0,
            "pruned_untagged": 0.0,
            "protected_tagged": 0.0,
            "protected_young": 0.0,
            "released_tagged": 0.0,
            "tag_threshold_mean": 0.0,
            "tagged_active_total": 0.0,
            "total_prunable": 0.0,
            "prunable_count_by_key": {},
            "usage_mean_by_key": {},
            "usage_peak_by_key": {},
            "survival_mean_by_key": {},
            "survival_p95_by_key": {},
            "survival_max_by_key": {},
            "threshold_by_key": {},
            "prune_gap_mass_by_key": {},
            "effective_prune_mass_by_key": {},
            "creation_need_by_key": {},
            "creation_need_ema_by_key": {},
            "prune_pressure_by_key": {},
            "positive_gap_count_by_key": {},
            "pruned_by_key": {},
            "pruned_persistent_low_epoch": 0.0,
            "pruned_short_only_low_epoch": 0.0,
            "pruned_persistent_low_ratio_epoch": 0.0,
            "pruned_short_only_low_ratio_epoch": 0.0,
            "pruned_persistent_low_total": 0.0,
            "pruned_short_only_low_total": 0.0,
            "pruned_short_only_low_ratio_total": 0.0,
            "pruned_persistent_low_by_key": {},
            "pruned_short_only_low_by_key": {},
            "pruned_age_mean_epoch": 0.0,
            "pruned_age_p25_epoch": 0.0,
            "pruned_age_p75_epoch": 0.0,
        }
        for dead_label, _ in self._pruning_dead_thresholds:
            self.last_pruning_stats[f"dead_count_lt_{dead_label}_epoch"] = 0.0
            self.last_pruning_stats[f"dead_count_lt_{dead_label}_total"] = 0.0
            self.last_pruning_stats[f"dead_to_pruned_ratio_lt_{dead_label}_epoch"] = 0.0
            self.last_pruning_stats[f"dead_to_pruned_ratio_lt_{dead_label}_total"] = 0.0
            self.last_pruning_stats[f"dead_count_lt_{dead_label}_by_key"] = {}
        self.last_pruning_stats["median_dead_count_k001_epoch"] = 0.0
        self.last_pruning_stats["median_dead_count_k001_total"] = 0.0
        self.last_pruning_stats["median_dead_to_pruned_ratio_k001_epoch"] = 0.0
        self.last_pruning_stats["median_dead_to_pruned_ratio_k001_total"] = 0.0
        self.last_pruning_stats["median_dead_count_k002_epoch"] = 0.0
        self.last_pruning_stats["median_dead_count_k002_total"] = 0.0
        self.last_pruning_stats["median_dead_to_pruned_ratio_k002_epoch"] = 0.0
        self.last_pruning_stats["median_dead_to_pruned_ratio_k002_total"] = 0.0
        self.last_pruning_stats["median_usage_by_key"] = {}
        self.last_pruning_stats["median_dead_count_k001_by_key"] = {}
        self.last_pruning_stats["median_dead_count_k002_by_key"] = {}
        self.last_pruning_stats["median_prune_candidate_count_epoch"] = 0.0
        self.last_pruning_stats["median_prune_effective_count_epoch"] = 0.0
        self.last_pruning_stats["median_prune_threshold_by_key"] = {}
        self.last_pruning_stats["median_prune_candidate_count_by_key"] = {}
        self.last_pruning_stats["median_prune_effective_count_by_key"] = {}
        self.last_tenure_stats: Dict[str, Any] = {
            "enabled": bool(self.tenure_enabled),
            "score_source": "survival_sqrt_mean_peak",
            "new_tenures_total": 0.0,
            "tenured_total": 0.0,
            "tenured_share_global": 0.0,
            "active_count_by_key": {},
            "mature_count_by_key": {},
            "tenured_count_by_key": {},
            "tenured_share_by_key": {},
            "tenure_threshold_by_key": {},
            "tenure_norm_anchor_by_key": {},
            "eligible_to_tenure_by_key": {},
            "new_tenures_by_key": {},
            "max_tenured_by_key": {},
        }
        # Counters
        self.creation_count = 0
        self.pruning_count = 0
        self.protected_from_pruning = 0
        self.created_channels_total = 0
        self.created_channel_edges_total = 0
        self.created_channels_total_by_prefix: Dict[str, int] = {}
        self.tenured_channels_total = 0
        self.rejected_channels_total = 0
        self.local_reinforce_channels_total = 0
        self.local_reinforce_edges_total = 0
        self.current_epoch = 0
        self.pruned_persistent_low_total = 0
        self.pruned_short_only_low_total = 0
        self._dead_count_totals_by_label: Dict[str, int] = {
            dead_label: 0 for dead_label, _ in self._pruning_dead_thresholds
        }
        self._dead_pruned_observed_total = 0
        self._median_dead_count_k001_total = 0
        self._median_dead_count_k002_total = 0
        self._median_dead_pruned_observed_total = 0
        self._epoch_runtime_tag = -1
        self._epoch_pathway_dropout_active = False
        self._epoch_pathway_dropout_rate_effective = 0.0
        self._epoch_pathway_dropout_candidate_channels_by_prefix: Dict[str, int] = {}
        self._epoch_pathway_dropout_dropped_channels_by_prefix: Dict[str, int] = {}
        self._epoch_pathway_dropout_protected_channels_by_prefix: Dict[str, int] = {}
        self._local_reinforce_queue: Dict[int, Dict[str, Any]] = {}
        self._epoch_local_reinforce_channels: Dict[int, bool] = {}
        self._epoch_local_reinforce_edges_up_by_prefix: Dict[str, int] = {}
        self._epoch_local_reinforce_edges_down_by_prefix: Dict[str, int] = {}
        self._epoch_local_reinforce_edges = 0
        self._epoch_local_reinforce_budget_used = 0
        self._channel_activity_tenured_sum = 0.0
        self._channel_activity_tenured_count = 0
        self._channel_activity_rejected_sum = 0.0
        self._channel_activity_rejected_count = 0
        self._next_probationary_channel_id = 1
        self._probationary_channels: Dict[int, Dict[str, Any]] = {}
        self._tenured_grace_channels: Dict[int, Dict[str, Any]] = {}
        self._channel_probe_threshold_ema: Optional[float] = None
        self._phase1_exploration_registry: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._phase1_eligible_hidden_ids_by_prefix: Dict[str, List[int]] = {}
        self._phase1_new_hidden_attempts_epoch = 0
        self._phase1_retested_hidden_attempts_epoch = 0
        self._phase1_evaluated_channels_total_current_level = 0
        self._phase1_tenured_channels_total_current_level = 0
        self._phase1_rejected_channels_total_current_level = 0
        self._created_channels_total_current_level = 0
        self.last_channel_probe_stats: Dict[str, Any] = self._empty_channel_probe_stats()
        self.last_creation_stats: Dict[str, Any] = self._empty_creation_stats()
        self.last_ffn_hidden_stats: Dict[str, Any] = {
            "by_prefix": {},
            "by_role_group": {},
        }

        # Embeddings + positional encodings
        self.src_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.tgt_embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        if tie_embeddings:
            self.tgt_embedding.weight = self.src_embedding.weight

        src_pe = self._build_sinusoidal_positional_encoding(self.max_src_len, self.d_model)
        tgt_pe = self._build_sinusoidal_positional_encoding(self.max_tgt_len, self.d_model)
        self.register_buffer("src_positional_encoding", src_pe, persistent=False)
        self.register_buffer("tgt_positional_encoding", tgt_pe, persistent=False)

        # Layer norms (dense, standard Transformer)
        self.enc_norm1 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_enc_layers)])
        self.enc_norm2 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_enc_layers)])
        self.dec_norm1 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_dec_layers)])
        self.dec_norm2 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_dec_layers)])
        self.dec_norm3 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_dec_layers)])
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

        self.output_proj = nn.Linear(self.d_model, self.vocab_size)

        # Sparse matrices and metadata
        self.weights = nn.ParameterDict()
        self.masks: Dict[str, str] = {}
        self.connection_meta: Dict[str, Dict[str, int]] = {}
        self.replay_tags: Dict[str, str] = {}
        self._dst_groups: List[int] = []

        self._init_sparse_transformer_connectivity(
            initial_connectivity_attn=float(initial_connectivity_attn),
            initial_connectivity_ffn=float(initial_connectivity_ffn),
        )
        self._init_tracking_buffers()
        self._creation_need_ema = {key: None for key in self.weights.keys()}
        self._pruning_creation_need_ema = {key: None for key in self.weights.keys()}

        # Approx "n_layers" for quota compatibility in existing code.
        self.n_layers = len(self._dst_groups) + 1
        self.layer_sizes = [self.d_model for _ in range(self.n_layers)]

        stats = self.get_connectivity_stats()
        logger.info(
            "SparseTransformerSeq2Seq created: d_model=%d, heads=%d, enc=%d, dec=%d",
            self.d_model,
            self.n_heads,
            self.n_enc_layers,
            self.n_dec_layers,
        )
        logger.info(
            "  Sparse synapses: %d/%d (%.2f%%)",
            stats.total_active,
            stats.total_possible,
            100.0 * stats.overall_density,
        )
        logger.info(
            "  Creation score mode: %s (norm=%s, alpha=%.2f)",
            self.creation_score_mode,
            self.creation_score_normalization,
            self.creation_score_alpha,
        )
        if self.ffn_channel_complete_k > 0:
            logger.info(
                "  FFN channel-complete init: enabled (k=%d)",
                self.ffn_channel_complete_k,
            )
        if self.creation_policy in {"paired_ffn_channel_random", "paired_ffn_channel_probe"}:
            logger.info(
                "  FFN paired-channel creation: enabled=%s fan_in=%d fan_out=%d bundle=%d qdeg<=%.2f full=%s dens<=%.2f frac=%.2f init=%.4f",
                self.channel_creation_enabled,
                self.channel_fan_in,
                self.channel_fan_out,
                self.channel_bundle_size,
                self.channel_underused_max_quantile,
                self.channel_require_full_bundle,
                self.block_density_cap,
                self.channel_budget_fraction,
                self.channel_bundle_init_scale,
            )
            if self.creation_policy == "paired_ffn_channel_probe":
                logger.info(
                    "  FFN channel probe: period=%d max_prefix=%d max_global=%d thr_ema_alpha=%.3f",
                    self.channel_probe_period,
                    self.max_probationary_channels_per_prefix,
                    self.max_probationary_channels_global,
                    self.channel_probe_threshold_ema_alpha,
                )
        logger.info(
            "  Pruning policy: %s (qref=%.2f, topk=%d, mature_only=%s)",
            self.pruning_policy,
            self.pruning_active_ref_quantile,
            self.pruning_priority_topk,
            self.pruning_ref_mature_only,
        )
        if self.tenure_enabled:
            logger.info(
                "  Tenure: enabled (alpha=%.2f, min_age=%d, q=%.2f, streak=%d, max_share=%.2f, decay=%d, release=%s)",
                self.tenure_score_ema_alpha,
                self.tenure_min_age,
                self.tenure_acquire_quantile,
                self.tenure_acquire_streak,
                self.tenure_max_share_per_matrix,
                self.tenure_decay_on_miss,
                self.tenure_release_enabled,
            )
        if self.pruning_couple_creation_need:
            logger.info(
                "  Pruning creation coupling: enabled (need_alpha=%.2f, coupling_alpha=%.2f)",
                self.pruning_creation_need_ema_alpha,
                self.pruning_creation_coupling_alpha,
            )

    # ---------------------------------------------------------------------
    # Initialization helpers
    # ---------------------------------------------------------------------
    def _build_sinusoidal_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / float(d_model))
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def _init_sparse_transformer_connectivity(
        self,
        initial_connectivity_attn: float,
        initial_connectivity_ffn: float,
    ) -> None:
        """
        Register all sparse matrices:
        - encoder: Q, K, V, O, FFN_up, FFN_down
        - decoder: self-Q/K/V/O, cross-Q/K/V/O, FFN_up/down
        """
        group = 1

        for l in range(self.n_enc_layers):
            p = f"enc{l}"
            self._add_sparse_weight(f"{p}_Q", self.d_model, self.d_model, initial_connectivity_attn, group)
            self._add_sparse_weight(f"{p}_K", self.d_model, self.d_model, initial_connectivity_attn, group)
            self._add_sparse_weight(f"{p}_V", self.d_model, self.d_model, initial_connectivity_attn, group)
            self._add_sparse_weight(f"{p}_O", self.d_model, self.d_model, initial_connectivity_attn, group)
            self._add_sparse_weight(f"{p}_FFN_up", self.d_model, self.d_ff, initial_connectivity_ffn, group)
            self._add_sparse_weight(f"{p}_FFN_down", self.d_ff, self.d_model, initial_connectivity_ffn, group)
            self._apply_ffn_channel_complete(p)
            self._dst_groups.append(group)
            group += 1

        for l in range(self.n_dec_layers):
            p = f"dec{l}"
            self._add_sparse_weight(f"{p}_self_Q", self.d_model, self.d_model, initial_connectivity_attn, group)
            self._add_sparse_weight(f"{p}_self_K", self.d_model, self.d_model, initial_connectivity_attn, group)
            self._add_sparse_weight(f"{p}_self_V", self.d_model, self.d_model, initial_connectivity_attn, group)
            self._add_sparse_weight(f"{p}_self_O", self.d_model, self.d_model, initial_connectivity_attn, group)
            self._add_sparse_weight(f"{p}_cross_Q", self.d_model, self.d_model, initial_connectivity_attn, group)
            self._add_sparse_weight(f"{p}_cross_K", self.d_model, self.d_model, initial_connectivity_attn, group)
            self._add_sparse_weight(f"{p}_cross_V", self.d_model, self.d_model, initial_connectivity_attn, group)
            self._add_sparse_weight(f"{p}_cross_O", self.d_model, self.d_model, initial_connectivity_attn, group)
            self._add_sparse_weight(f"{p}_FFN_up", self.d_model, self.d_ff, initial_connectivity_ffn, group)
            self._add_sparse_weight(f"{p}_FFN_down", self.d_ff, self.d_model, initial_connectivity_ffn, group)
            self._apply_ffn_channel_complete(p)
            self._dst_groups.append(group)
            group += 1

    def _add_sparse_weight(
        self,
        key: str,
        in_dim: int,
        out_dim: int,
        density: float,
        dst_group: int,
    ) -> None:
        density = float(max(1e-4, min(1.0, density)))
        std = math.sqrt(2.0 / float(max(1, in_dim + out_dim)))
        w = torch.randn(in_dim, out_dim) * std
        self.weights[key] = nn.Parameter(w)

        mask = self._make_coverage_mask(in_dim, out_dim, density)
        mask_name = f"mask_{key}"
        self.register_buffer(mask_name, mask)
        self.masks[key] = mask_name
        self.connection_meta[key] = {"in_dim": int(in_dim), "out_dim": int(out_dim), "dst_group": int(dst_group)}

    def _make_coverage_mask(self, in_dim: int, out_dim: int, density: float) -> torch.Tensor:
        """
        Sparse mask with guaranteed destination coverage:
        at least one incoming synapse for each destination unit.
        """
        n_total = int(in_dim * out_dim)
        n_target = max(out_dim, int(round(n_total * density)))
        n_target = min(n_total, n_target)

        mask = torch.zeros(in_dim, out_dim, dtype=torch.float32)
        src_indices = torch.randint(0, in_dim, (out_dim,))
        for dst in range(out_dim):
            mask[src_indices[dst], dst] = 1.0

        remaining = int(n_target - int(mask.sum().item()))
        if remaining > 0:
            inactive = (mask == 0).nonzero(as_tuple=False)
            if int(inactive.size(0)) > 0:
                perm = torch.randperm(int(inactive.size(0)))[:remaining]
                chosen = inactive[perm]
                mask[chosen[:, 0], chosen[:, 1]] = 1.0

        return mask

    def _rewire_mask_min_column_degree(self, mask: torch.Tensor, min_degree: int) -> torch.Tensor:
        min_degree = int(max(0, min_degree))
        if min_degree <= 0:
            return mask
        out_dim = int(mask.size(1))
        total_active = int(mask.sum().item())
        feasible_degree = int(total_active // max(1, out_dim))
        target_degree = int(min(min_degree, feasible_degree))
        if target_degree <= 0:
            return mask

        out = mask.clone()
        for dst in range(out_dim):
            while int(out[:, dst].sum().item()) < target_degree:
                donors = (out.sum(dim=0) > target_degree).nonzero(as_tuple=False).flatten()
                inactive = (out[:, dst] == 0).nonzero(as_tuple=False).flatten()
                if int(donors.numel()) == 0 or int(inactive.numel()) == 0:
                    break
                donor_col = int(donors[torch.randint(0, int(donors.numel()), (1,))].item())
                donor_rows = (out[:, donor_col] > 0).nonzero(as_tuple=False).flatten()
                if int(donor_rows.numel()) == 0:
                    break
                donor_row = int(donor_rows[torch.randint(0, int(donor_rows.numel()), (1,))].item())
                new_row = int(inactive[torch.randint(0, int(inactive.numel()), (1,))].item())
                out[donor_row, donor_col] = 0.0
                out[new_row, dst] = 1.0
        return out

    def _rewire_mask_min_row_degree(self, mask: torch.Tensor, min_degree: int) -> torch.Tensor:
        min_degree = int(max(0, min_degree))
        if min_degree <= 0:
            return mask
        in_dim = int(mask.size(0))
        total_active = int(mask.sum().item())
        feasible_degree = int(total_active // max(1, in_dim))
        target_degree = int(min(min_degree, feasible_degree))
        if target_degree <= 0:
            return mask

        out = mask.clone()
        for src in range(in_dim):
            while int(out[src, :].sum().item()) < target_degree:
                donors = (out.sum(dim=1) > target_degree).nonzero(as_tuple=False).flatten()
                inactive = (out[src, :] == 0).nonzero(as_tuple=False).flatten()
                if int(donors.numel()) == 0 or int(inactive.numel()) == 0:
                    break
                donor_row = int(donors[torch.randint(0, int(donors.numel()), (1,))].item())
                donor_cols = (out[donor_row, :] > 0).nonzero(as_tuple=False).flatten()
                if int(donor_cols.numel()) == 0:
                    break
                donor_col = int(donor_cols[torch.randint(0, int(donor_cols.numel()), (1,))].item())
                new_col = int(inactive[torch.randint(0, int(inactive.numel()), (1,))].item())
                out[donor_row, donor_col] = 0.0
                out[src, new_col] = 1.0
        return out

    def _apply_ffn_channel_complete(self, prefix: str) -> None:
        k = int(max(0, self.ffn_channel_complete_k))
        if k <= 0:
            return
        up_key = f"{prefix}_FFN_up"
        down_key = f"{prefix}_FFN_down"
        if up_key not in self.masks or down_key not in self.masks:
            return
        up_mask = getattr(self, self.masks[up_key])
        down_mask = getattr(self, self.masks[down_key])
        up_before = up_mask.sum(dim=0)
        down_before = down_mask.sum(dim=1)
        up_rewired = self._rewire_mask_min_column_degree(up_mask, k)
        down_rewired = self._rewire_mask_min_row_degree(down_mask, k)
        up_mask.copy_(up_rewired)
        down_mask.copy_(down_rewired)
        up_after = up_mask.sum(dim=0)
        down_after = down_mask.sum(dim=1)
        logger.info(
            "FFN channel-complete init [%s]: k=%d | up min %.0f->%.0f | down min %.0f->%.0f",
            prefix,
            k,
            float(up_before.min().item()) if up_before.numel() > 0 else 0.0,
            float(up_after.min().item()) if up_after.numel() > 0 else 0.0,
            float(down_before.min().item()) if down_before.numel() > 0 else 0.0,
            float(down_after.min().item()) if down_after.numel() > 0 else 0.0,
        )

    def _init_tracking_buffers(self) -> None:
        for key in self.weights.keys():
            mask = getattr(self, self.masks[key])
            self.register_buffer(f"usage_{key}", torch.zeros_like(mask))
            # Non-persistent to keep backward compatibility when bootstrapping
            # from pre-3c4.3.2b checkpoints that do not contain this buffer.
            self.register_buffer(f"usage_long_{key}", torch.zeros_like(mask), persistent=False)
            self.register_buffer(f"usage_peak_{key}", torch.zeros_like(mask))
            self.register_buffer(f"contribution_{key}", torch.zeros_like(mask))
            self.register_buffer(f"age_{key}", torch.zeros_like(mask))
            self.register_buffer(f"historical_importance_{key}", torch.zeros_like(mask))
            self.register_buffer(f"fisher_{key}", torch.zeros_like(mask))
            self.register_buffer(f"create_score_{key}", torch.zeros_like(mask))
            self.register_buffer(f"create_direction_{key}", torch.zeros_like(mask))
            self.register_buffer(f"tenure_mask_{key}", torch.zeros_like(mask))
            self.register_buffer(f"tenure_score_ema_{key}", torch.zeros_like(mask))
            self.register_buffer(f"tenure_streak_{key}", torch.zeros_like(mask, dtype=torch.int16))

            tag_name = f"replay_tag_{key}"
            self.register_buffer(tag_name, torch.zeros_like(mask))
            self.replay_tags[key] = tag_name

            if str(key).endswith("_FFN_up"):
                prefix = str(key).rsplit("_FFN_up", 1)[0]
                out_dim = int(self.connection_meta[key]["out_dim"])
                self.register_buffer(f"hidden_usage_mean_{prefix}", torch.zeros(out_dim))
                self.register_buffer(f"hidden_usage_peak_{prefix}", torch.zeros(out_dim))
                self.register_buffer(f"hidden_fire_rate_{prefix}", torch.zeros(out_dim))

    def _creation_role_group(self, key: str) -> str:
        if str(key).startswith("enc"):
            return "enc"
        if str(key).startswith("dec"):
            return "dec"
        return "other"

    def _register_gradient_hooks(self) -> None:
        """
        Update creation score buffers from weight gradients.
        This is architecture-agnostic and robust for Transformer matrices.
        """

        def _make_hook(key: str):
            def _hook(grad: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    g = grad.detach()
                    mask = getattr(self, self.masks[key]).to(g.device)
                    score = g.abs() * mask
                    direction = g * mask
                    alpha = 0.10
                    s_buf = self._get_creation_score(key).to(g.device)
                    d_buf = self._get_creation_direction(key).to(g.device)
                    s_buf.copy_((1.0 - alpha) * s_buf + alpha * score)
                    d_buf.copy_((1.0 - alpha) * d_buf + alpha * direction)
                return grad

            return _hook

        for key, p in self.weights.items():
            p.register_hook(_make_hook(key))

    # ---------------------------------------------------------------------
    # Buffer getters
    # ---------------------------------------------------------------------
    def _get_usage(self, key: str) -> torch.Tensor:
        return getattr(self, f"usage_{key}")

    def _get_usage_long(self, key: str) -> torch.Tensor:
        return getattr(self, f"usage_long_{key}")

    def _get_usage_peak(self, key: str) -> torch.Tensor:
        return getattr(self, f"usage_peak_{key}")

    def _get_contribution(self, key: str) -> torch.Tensor:
        return getattr(self, f"contribution_{key}")

    def _get_age(self, key: str) -> torch.Tensor:
        return getattr(self, f"age_{key}")

    def _get_historical_importance(self, key: str) -> torch.Tensor:
        return getattr(self, f"historical_importance_{key}")

    def _get_fisher(self, key: str) -> torch.Tensor:
        return getattr(self, f"fisher_{key}")

    def _get_creation_score(self, key: str) -> torch.Tensor:
        return getattr(self, f"create_score_{key}")

    def _get_creation_direction(self, key: str) -> torch.Tensor:
        return getattr(self, f"create_direction_{key}")

    def _get_tenure_mask(self, key: str) -> torch.Tensor:
        return getattr(self, f"tenure_mask_{key}")

    def _get_tenure_score_ema(self, key: str) -> torch.Tensor:
        return getattr(self, f"tenure_score_ema_{key}")

    def _get_tenure_streak(self, key: str) -> torch.Tensor:
        return getattr(self, f"tenure_streak_{key}")

    def _get_replay_tag(self, key: str) -> torch.Tensor:
        return getattr(self, self.replay_tags[key])

    def _get_hidden_usage_mean(self, prefix: str) -> torch.Tensor:
        return getattr(self, f"hidden_usage_mean_{prefix}")

    def _get_hidden_usage_peak(self, prefix: str) -> torch.Tensor:
        return getattr(self, f"hidden_usage_peak_{prefix}")

    def _get_hidden_fire_rate(self, prefix: str) -> torch.Tensor:
        return getattr(self, f"hidden_fire_rate_{prefix}")

    # ---------------------------------------------------------------------
    # Sparse linear and attention
    # ---------------------------------------------------------------------
    def _track_synapse_usage(self, key: str, src_activation: torch.Tensor, effective_weight: torch.Tensor) -> None:
        with torch.no_grad():
            dims = tuple(range(src_activation.dim() - 1))
            src_mean = src_activation.detach().abs().mean(dim=dims).unsqueeze(1)
            src_peak = src_activation.detach().abs().amax(dim=dims).unsqueeze(1)
            weight_abs = effective_weight.detach().abs()
            usage = src_mean * weight_abs
            usage_peak = src_peak * weight_abs
            alpha = 0.10
            buf = self._get_usage(key)
            long_buf = self._get_usage_long(key)
            peak_buf = self._get_usage_peak(key)
            buf.copy_((1.0 - alpha) * buf + alpha * usage)
            long_alpha = float(self.pruning_persistence_long_alpha)
            long_buf.copy_((1.0 - long_alpha) * long_buf + long_alpha * usage)
            peak_buf.copy_((1.0 - alpha) * peak_buf + alpha * usage_peak)

    def _track_ffn_hidden_usage(
        self,
        prefix: str,
        hidden_activation: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> None:
        with torch.no_grad():
            h = hidden_activation.detach().float()
            h2 = h.reshape(-1, h.size(-1))
            if token_mask is not None and hidden_activation.dim() >= 3:
                valid = token_mask.detach().reshape(-1)
                if valid.dtype != torch.bool:
                    valid = valid > 0
                if int(valid.numel()) == int(h2.size(0)):
                    if int(valid.sum().item()) <= 0:
                        return
                    h2 = h2[valid]
            if int(h2.numel()) <= 0:
                return
            h_abs = h2.abs()
            mean_usage = h_abs.mean(dim=0)
            peak_usage = h_abs.amax(dim=0)
            fire_rate = (h2 > 0).float().mean(dim=0)
            alpha = 0.10
            mean_buf = self._get_hidden_usage_mean(prefix)
            peak_buf = self._get_hidden_usage_peak(prefix)
            fire_buf = self._get_hidden_fire_rate(prefix)
            mean_buf.copy_((1.0 - alpha) * mean_buf + alpha * mean_usage.to(mean_buf.device))
            peak_buf.copy_((1.0 - alpha) * peak_buf + alpha * peak_usage.to(peak_buf.device))
            fire_buf.copy_((1.0 - alpha) * fire_buf + alpha * fire_rate.to(fire_buf.device))

    @staticmethod
    def _gini_coefficient(values: torch.Tensor) -> float:
        if values.numel() <= 0:
            return 0.0
        x = values.detach().float().flatten().clamp_min(0.0)
        total = float(x.sum().item())
        if total <= 0.0:
            return 0.0
        xs, _ = torch.sort(x)
        n = int(xs.numel())
        idx = torch.arange(1, n + 1, device=xs.device, dtype=xs.dtype)
        gini = (2.0 * float((idx * xs).sum().item()) / (float(n) * total)) - ((float(n) + 1.0) / float(n))
        return float(max(0.0, min(1.0, gini)))

    @staticmethod
    def _normalized_entropy(values: torch.Tensor) -> float:
        if values.numel() <= 1:
            return 0.0
        x = values.detach().float().flatten().clamp_min(0.0)
        total = float(x.sum().item())
        if total <= 0.0:
            return 0.0
        p = x / total
        p = p[p > 0]
        if int(p.numel()) <= 1:
            return 0.0
        entropy = float((-(p * p.log()).sum()).item())
        denom = math.log(float(x.numel()))
        if denom <= 0.0:
            return 0.0
        return float(max(0.0, min(1.0, entropy / denom)))

    @staticmethod
    def _topk_share(values: torch.Tensor, ratio: float) -> float:
        if values.numel() <= 0:
            return 0.0
        x = values.detach().float().flatten().clamp_min(0.0)
        total = float(x.sum().item())
        if total <= 0.0:
            return 0.0
        k = max(1, int(math.ceil(float(x.numel()) * float(ratio))))
        topk = torch.topk(x, k=min(k, int(x.numel()))).values
        return float(topk.sum().item() / total)

    @staticmethod
    def _ratio_above_relative_threshold(values: torch.Tensor, rel: float) -> float:
        if values.numel() <= 0:
            return 0.0
        x = values.detach().float().flatten().clamp_min(0.0)
        max_val = float(x.max().item()) if int(x.numel()) > 0 else 0.0
        if max_val <= 0.0:
            return 0.0
        return float((x >= (float(rel) * max_val)).float().mean().item())

    def get_ffn_hidden_stats(self, include_raw: bool = False) -> Dict[str, Any]:
        by_prefix: Dict[str, Any] = {}
        role_agg: Dict[str, Dict[str, float]] = {}
        role_counts: Dict[str, int] = {}
        for key in [str(k) for k in self.weights.keys() if str(k).endswith("_FFN_up")]:
            prefix = key.rsplit("_FFN_up", 1)[0]
            role = self._creation_role_group(prefix)
            mean_buf = self._get_hidden_usage_mean(prefix).detach().cpu().float().clamp_min(0.0)
            peak_buf = self._get_hidden_usage_peak(prefix).detach().cpu().float().clamp_min(0.0)
            fire_buf = self._get_hidden_fire_rate(prefix).detach().cpu().float().clamp(0.0, 1.0)
            summary = {
                "n_units": int(mean_buf.numel()),
                "mean_usage_mean": float(mean_buf.mean().item()) if int(mean_buf.numel()) > 0 else 0.0,
                "mean_usage_p95": float(torch.quantile(mean_buf, 0.95).item()) if int(mean_buf.numel()) > 0 else 0.0,
                "peak_usage_mean": float(peak_buf.mean().item()) if int(peak_buf.numel()) > 0 else 0.0,
                "fire_rate_mean": float(fire_buf.mean().item()) if int(fire_buf.numel()) > 0 else 0.0,
                "fire_rate_p95": float(torch.quantile(fire_buf, 0.95).item()) if int(fire_buf.numel()) > 0 else 0.0,
                "gini_mean_usage": self._gini_coefficient(mean_buf),
                "entropy_mean_usage": self._normalized_entropy(mean_buf),
                "top10_share_mean_usage": self._topk_share(mean_buf, 0.10),
                "top20_share_mean_usage": self._topk_share(mean_buf, 0.20),
                "ratio_units_above_10pct_max": self._ratio_above_relative_threshold(mean_buf, 0.10),
                "ratio_units_above_1pct_max": self._ratio_above_relative_threshold(mean_buf, 0.01),
            }
            if include_raw:
                summary["raw"] = {
                    "mean_usage": mean_buf.tolist(),
                    "peak_usage": peak_buf.tolist(),
                    "fire_rate": fire_buf.tolist(),
                }
            by_prefix[prefix] = summary

            role_entry = role_agg.setdefault(role, {
                "mean_usage_mean": 0.0,
                "fire_rate_mean": 0.0,
                "gini_mean_usage": 0.0,
                "entropy_mean_usage": 0.0,
                "top10_share_mean_usage": 0.0,
                "ratio_units_above_10pct_max": 0.0,
            })
            role_counts[role] = role_counts.get(role, 0) + 1
            for metric in role_entry.keys():
                role_entry[metric] += float(summary[metric])

        by_role_group: Dict[str, Any] = {}
        for role, vals in role_agg.items():
            cnt = max(1, int(role_counts.get(role, 1)))
            by_role_group[role] = {metric: float(value) / float(cnt) for metric, value in vals.items()}

        stats = {
            "by_prefix": by_prefix,
            "by_role_group": by_role_group,
        }
        self.last_ffn_hidden_stats = stats
        return stats

    def _should_use_mlp_like_creation_score(self, key: str) -> bool:
        return self.creation_score_mode == "mlp_seq" and "_FFN_" in str(key)

    def _compute_creation_signal(
        self,
        src_activation: torch.Tensor,
        grad: torch.Tensor,
        key: str,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        x2 = src_activation.reshape(-1, src_activation.size(-1)).float()
        g2 = grad.reshape(-1, grad.size(-1)).float()

        if self._should_use_mlp_like_creation_score(key):
            valid_count = int(x2.size(0))
            if token_mask is not None and src_activation.dim() >= 3:
                valid = token_mask.detach().reshape(-1)
                if valid.dtype != torch.bool:
                    valid = valid > 0
                if int(valid.numel()) == int(x2.size(0)):
                    valid_count = int(valid.sum().item())
                    if valid_count <= 0:
                        return None, None
                    x2 = x2[valid]
                    g2 = g2[valid]
            if self.creation_score_normalization == "valid_tokens":
                norm = max(1, int(valid_count))
            else:
                norm = max(1, int(src_activation.size(0)))
            direction = (x2.transpose(0, 1) @ g2) / float(norm)
            score = (x2.abs().transpose(0, 1) @ g2.abs()) / float(norm)
            return score, direction

        n = max(1, int(x2.size(0)))
        signed = (x2.transpose(0, 1) @ g2) / float(n)
        return signed.abs(), signed

    def _sparse_linear(
        self,
        x: torch.Tensor,
        key: str,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w = self.weights[key]
        mask = getattr(self, self.masks[key]).to(w.device)
        effective_w = w * mask
        if self.training:
            self._track_synapse_usage(key, x, effective_w)
        out = torch.matmul(x, effective_w)

        # Build creation scores from local pre-activation gradient signal.
        # This provides a score on all potential edges (including currently inactive ones).
        if self.training and torch.is_grad_enabled():
            src_detached = x.detach()
            mask_detached = token_mask.detach() if token_mask is not None else None

            def _hook(grad: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    # Keep creation signal accumulation in float32 for numerical stability
                    # and AMP compatibility (half gradients + fp32 activations).
                    score, direction = self._compute_creation_signal(
                        src_activation=src_detached,
                        grad=grad.detach(),
                        key=key,
                        token_mask=mask_detached,
                    )
                    if score is None or direction is None:
                        return grad

                    alpha = self.creation_score_alpha
                    s_buf = self._get_creation_score(key).to(score.device)
                    d_buf = self._get_creation_direction(key).to(direction.device)
                    s_buf.copy_((1.0 - alpha) * s_buf + alpha * score)
                    d_buf.copy_((1.0 - alpha) * d_buf + alpha * direction)
                return grad

            out.register_hook(_hook)

        return out

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Dh]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, Dh]
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.d_model)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,      # [1,1,Tq,Tk], bool
        key_padding_mask: Optional[torch.Tensor] = None,  # [B,1,1,Tk], bool (True=valid)
    ) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))  # [B,H,Tq,Tk]

        valid_mask: Optional[torch.Tensor] = None
        if causal_mask is not None:
            valid_mask = causal_mask if valid_mask is None else (valid_mask & causal_mask)
        if key_padding_mask is not None:
            valid_mask = key_padding_mask if valid_mask is None else (valid_mask & key_padding_mask)

        if valid_mask is not None:
            mask_fill = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~valid_mask, mask_fill)

        attn = torch.softmax(scores, dim=-1)
        if valid_mask is not None:
            vm = valid_mask.to(attn.dtype)
            attn = attn * vm
            denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            attn = attn / denom

        attn = self.dropout(attn)
        return torch.matmul(attn, v)

    def _encoder_self_attention(
        self,
        x: torch.Tensor,
        layer_idx: int,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        p = f"enc{layer_idx}"
        q = self._split_heads(self._sparse_linear(x, f"{p}_Q"))
        k = self._split_heads(self._sparse_linear(x, f"{p}_K"))
        v = self._split_heads(self._sparse_linear(x, f"{p}_V"))
        out = self._scaled_dot_product_attention(q, k, v, causal_mask=None, key_padding_mask=key_padding_mask)
        out = self._merge_heads(out)
        return self._sparse_linear(out, f"{p}_O")

    def _decoder_self_attention(
        self,
        x: torch.Tensor,
        layer_idx: int,
        causal_mask: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        p = f"dec{layer_idx}_self"
        q = self._split_heads(self._sparse_linear(x, f"{p}_Q"))
        k = self._split_heads(self._sparse_linear(x, f"{p}_K"))
        v = self._split_heads(self._sparse_linear(x, f"{p}_V"))
        out = self._scaled_dot_product_attention(
            q, k, v, causal_mask=causal_mask, key_padding_mask=key_padding_mask
        )
        out = self._merge_heads(out)
        return self._sparse_linear(out, f"{p}_O")

    def _decoder_cross_attention(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        layer_idx: int,
        memory_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        p = f"dec{layer_idx}_cross"
        q = self._split_heads(self._sparse_linear(x, f"{p}_Q"))
        k = self._split_heads(self._sparse_linear(memory, f"{p}_K"))
        v = self._split_heads(self._sparse_linear(memory, f"{p}_V"))
        out = self._scaled_dot_product_attention(q, k, v, causal_mask=None, key_padding_mask=memory_padding_mask)
        out = self._merge_heads(out)
        return self._sparse_linear(out, f"{p}_O")

    def _ffn(
        self,
        x: torch.Tensor,
        prefix: str,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        up = self._sparse_linear(x, f"{prefix}_FFN_up", token_mask=token_mask)
        if self.ffn_activation == "gelu":
            up = F.gelu(up)
        else:
            up = F.relu(up)
        up = self._apply_channel_pathway_dropout(prefix, up, token_mask=token_mask)
        if self.training:
            self._track_ffn_hidden_usage(prefix, up, token_mask=token_mask)
        down = self._sparse_linear(up, f"{prefix}_FFN_down", token_mask=token_mask)
        return down

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward seq2seq.

        Args:
            src_tokens: [B, T_src], token ids
            tgt_tokens: [B, T_tgt], decoder input token ids

        Returns:
            logits: [B, T_tgt, vocab_size]
        """
        if src_tokens.dim() != 2 or tgt_tokens.dim() != 2:
            raise ValueError("Expected src/tgt token tensors with shape [B, T].")
        if src_tokens.dtype != torch.long:
            src_tokens = src_tokens.long()
        if tgt_tokens.dtype != torch.long:
            tgt_tokens = tgt_tokens.long()

        bsz, src_len = src_tokens.shape
        _b2, tgt_len = tgt_tokens.shape
        if src_len > self.max_src_len or tgt_len > self.max_tgt_len:
            raise ValueError(
                f"Sequence length exceeds configured maximums: src={src_len}/{self.max_src_len}, "
                f"tgt={tgt_len}/{self.max_tgt_len}"
            )

        dev = src_tokens.device
        src_pos = self.src_positional_encoding[:src_len].to(dev)
        tgt_pos = self.tgt_positional_encoding[:tgt_len].to(dev)

        src = self.src_embedding(src_tokens) + src_pos.unsqueeze(0)
        tgt = self.tgt_embedding(tgt_tokens) + tgt_pos.unsqueeze(0)

        # Masks: True means valid/allowed.
        src_token_mask = (src_tokens != self.pad_id)
        tgt_token_mask = (tgt_tokens != self.pad_id)
        src_valid = src_token_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T_src]
        tgt_valid = tgt_token_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T_tgt]
        causal = torch.tril(torch.ones((tgt_len, tgt_len), device=dev, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

        # Encoder
        memory = src
        for l in range(self.n_enc_layers):
            h = self.enc_norm1[l](memory)
            memory = memory + self.dropout(self._encoder_self_attention(h, l, key_padding_mask=src_valid))
            h = self.enc_norm2[l](memory)
            memory = memory + self.dropout(self._ffn(h, prefix=f"enc{l}", token_mask=src_token_mask))

        # Decoder
        out = tgt
        for l in range(self.n_dec_layers):
            h = self.dec_norm1[l](out)
            out = out + self.dropout(
                self._decoder_self_attention(h, l, causal_mask=causal, key_padding_mask=tgt_valid)
            )
            h = self.dec_norm2[l](out)
            out = out + self.dropout(self._decoder_cross_attention(h, memory, l, memory_padding_mask=src_valid))
            h = self.dec_norm3[l](out)
            out = out + self.dropout(self._ffn(h, prefix=f"dec{l}", token_mask=tgt_token_mask))

        logits = self.output_proj(out)  # [B, T_tgt, vocab]
        return logits

    # ---------------------------------------------------------------------
    # Dynamic sparse ops
    # ---------------------------------------------------------------------
    def compute_energy(self) -> float:
        total_active = 0.0
        for key in self.weights.keys():
            mask = getattr(self, self.masks[key])
            total_active += float(mask.sum().item())
        return total_active

    def compute_utility_scores(self) -> Dict[str, torch.Tensor]:
        utilities: Dict[str, torch.Tensor] = {}
        for key in self.weights.keys():
            mask = getattr(self, self.masks[key])
            usage = self._get_usage(key).to(mask.device)
            utilities[key] = usage * mask
        return utilities

    def compute_survival_scores(self) -> Dict[str, torch.Tensor]:
        survival_scores: Dict[str, torch.Tensor] = {}
        for key in self.weights.keys():
            mask = getattr(self, self.masks[key])
            usage_mean = self._get_usage(key).to(mask.device).clamp_min(0.0)
            usage_peak = self._get_usage_peak(key).to(mask.device).clamp_min(0.0)
            survival = torch.sqrt(usage_mean * usage_peak)
            survival_scores[key] = survival * mask
        return survival_scores

    def update_tenure_state(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "enabled": bool(self.tenure_enabled),
            "score_source": "survival_sqrt_mean_peak",
            "new_tenures_total": 0.0,
            "tenured_total": 0.0,
            "tenured_share_global": 0.0,
            "active_count_by_key": {},
            "mature_count_by_key": {},
            "tenured_count_by_key": {},
            "tenured_share_by_key": {},
            "tenure_threshold_by_key": {},
            "tenure_norm_anchor_by_key": {},
            "eligible_to_tenure_by_key": {},
            "new_tenures_by_key": {},
            "max_tenured_by_key": {},
        }
        if not self.tenure_enabled:
            self.last_tenure_stats = stats
            return stats

        survival_by_key = self.compute_survival_scores()
        alpha = float(self.tenure_score_ema_alpha)
        decay = int(self.tenure_decay_on_miss)
        total_active = 0
        total_tenured = 0
        total_new = 0

        for key in self.weights.keys():
            mask = getattr(self, self.masks[key])
            active = (mask > 0).detach()
            age = self._get_age(key).detach().float()
            mature = active & (age >= float(self.tenure_min_age))
            tenure_mask = self._get_tenure_mask(key)
            tenure_mask_bool = (tenure_mask > 0).detach()
            score_ema = self._get_tenure_score_ema(key)
            streak = self._get_tenure_streak(key)
            survival = survival_by_key[key].detach().to(mask.device).float()

            active_count = int(active.sum().item())
            mature_count = int(mature.sum().item())
            total_active += active_count

            if active_count <= 0:
                score_ema.zero_()
                streak.zero_()
                tenure_mask.zero_()
                stats["active_count_by_key"][key] = 0
                stats["mature_count_by_key"][key] = 0
                stats["tenured_count_by_key"][key] = 0
                stats["tenured_share_by_key"][key] = 0.0
                stats["tenure_threshold_by_key"][key] = 0.0
                stats["tenure_norm_anchor_by_key"][key] = 0.0
                stats["eligible_to_tenure_by_key"][key] = 0
                stats["new_tenures_by_key"][key] = 0
                stats["max_tenured_by_key"][key] = 0
                continue

            ref_mask = mature if mature_count > 0 else active
            ref_scores = survival[ref_mask]
            if int(ref_scores.numel()) > 0:
                raw_threshold = float(torch.quantile(ref_scores, self.tenure_acquire_quantile).item())
                norm_quantile = max(self.tenure_acquire_quantile, 0.95)
                norm_anchor = float(torch.quantile(ref_scores, norm_quantile).item())
            else:
                raw_threshold = 0.0
                norm_anchor = 0.0
            norm_anchor = max(norm_anchor, 1e-8)
            threshold_rel = float(max(0.0, raw_threshold / norm_anchor))
            relative_score = (survival / norm_anchor).clamp_min(0.0)

            score_ema.mul_(1.0 - alpha)
            score_ema[active] += alpha * relative_score[active].to(score_ema.dtype)
            score_ema[~active] = 0.0

            candidate = active & mature & ~tenure_mask_bool
            meets = candidate & (score_ema >= threshold_rel)
            misses = candidate & ~meets
            if int(meets.sum().item()) > 0:
                streak[meets] += 1
            if decay > 0 and int(misses.sum().item()) > 0:
                streak[misses] = torch.clamp(streak[misses] - int(decay), min=0)
            streak[~candidate] = 0

            tenured_active = tenure_mask_bool & active
            current_tenured = int(tenured_active.sum().item())
            max_tenured = int(math.floor(float(active_count) * float(self.tenure_max_share_per_matrix)))
            available_slots = max(0, max_tenured - current_tenured)
            eligible = candidate & (streak >= int(self.tenure_acquire_streak))
            eligible_count = int(eligible.sum().item())
            new_tenures = 0
            if available_slots > 0 and eligible_count > 0:
                idx = eligible.nonzero(as_tuple=False)
                vals = score_ema[eligible]
                topk = min(available_slots, eligible_count)
                chosen = idx[torch.topk(vals, k=topk).indices]
                tenure_mask[chosen[:, 0], chosen[:, 1]] = 1.0
                new_tenures = int(chosen.size(0))
                total_new += new_tenures

            tenured_active = (tenure_mask > 0) & active
            tenured_count = int(tenured_active.sum().item())
            total_tenured += tenured_count

            stats["active_count_by_key"][key] = active_count
            stats["mature_count_by_key"][key] = mature_count
            stats["tenured_count_by_key"][key] = tenured_count
            stats["tenured_share_by_key"][key] = float(tenured_count) / float(max(1, active_count))
            stats["tenure_threshold_by_key"][key] = float(threshold_rel)
            stats["tenure_norm_anchor_by_key"][key] = float(norm_anchor)
            stats["eligible_to_tenure_by_key"][key] = eligible_count
            stats["new_tenures_by_key"][key] = new_tenures
            stats["max_tenured_by_key"][key] = max_tenured

        stats["new_tenures_total"] = float(total_new)
        stats["tenured_total"] = float(total_tenured)
        stats["tenured_share_global"] = float(total_tenured) / float(max(1, total_active))
        self.last_tenure_stats = stats
        return stats

    def _empty_pruning_persistence_stats(self) -> Dict[str, Any]:
        ratio_total = float(
            float(self.pruned_short_only_low_total)
            / max(1.0, float(self.pruned_short_only_low_total + self.pruned_persistent_low_total))
        )
        out: Dict[str, Any] = {
            "pruned_persistent_low_epoch": 0.0,
            "pruned_short_only_low_epoch": 0.0,
            "pruned_persistent_low_ratio_epoch": 0.0,
            "pruned_short_only_low_ratio_epoch": 0.0,
            "pruned_persistent_low_total": float(self.pruned_persistent_low_total),
            "pruned_short_only_low_total": float(self.pruned_short_only_low_total),
            "pruned_short_only_low_ratio_total": ratio_total,
            "pruned_persistent_low_by_key": {},
            "pruned_short_only_low_by_key": {},
            "pruned_age_mean_epoch": 0.0,
            "pruned_age_p25_epoch": 0.0,
            "pruned_age_p75_epoch": 0.0,
        }
        for dead_label, _ in self._pruning_dead_thresholds:
            dead_total = int(self._dead_count_totals_by_label.get(dead_label, 0))
            out[f"dead_count_lt_{dead_label}_epoch"] = 0.0
            out[f"dead_count_lt_{dead_label}_total"] = float(dead_total)
            out[f"dead_to_pruned_ratio_lt_{dead_label}_epoch"] = 0.0
            out[f"dead_to_pruned_ratio_lt_{dead_label}_total"] = float(
                dead_total / max(1.0, float(self._dead_pruned_observed_total))
            )
            out[f"dead_count_lt_{dead_label}_by_key"] = {}
        out["median_dead_count_k001_epoch"] = 0.0
        out["median_dead_count_k001_total"] = float(self._median_dead_count_k001_total)
        out["median_dead_to_pruned_ratio_k001_epoch"] = 0.0
        out["median_dead_to_pruned_ratio_k001_total"] = float(
            float(self._median_dead_count_k001_total)
            / max(1.0, float(self._median_dead_pruned_observed_total))
        )
        out["median_dead_count_k002_epoch"] = 0.0
        out["median_dead_count_k002_total"] = float(self._median_dead_count_k002_total)
        out["median_dead_to_pruned_ratio_k002_epoch"] = 0.0
        out["median_dead_to_pruned_ratio_k002_total"] = float(
            float(self._median_dead_count_k002_total)
            / max(1.0, float(self._median_dead_pruned_observed_total))
        )
        out["median_usage_by_key"] = {}
        out["median_dead_count_k001_by_key"] = {}
        out["median_dead_count_k002_by_key"] = {}
        out["median_prune_candidate_count_epoch"] = 0.0
        out["median_prune_effective_count_epoch"] = 0.0
        out["median_prune_threshold_by_key"] = {}
        out["median_prune_candidate_count_by_key"] = {}
        out["median_prune_effective_count_by_key"] = {}
        return out

    def _compute_pruning_persistence_stats(
        self,
        pending_rows: Dict[str, List[int]],
        pending_cols: Dict[str, List[int]],
    ) -> Dict[str, Any]:
        if (
            not self.enable_pruning_persistence_diagnostic
            and not self.enable_pruning_natural_volume_diagnostic
            and not self.enable_pruning_relative_median_diagnostic
        ):
            return self._empty_pruning_persistence_stats()

        q = float(self.pruning_persistence_quantile)
        persistent_by_key: Dict[str, int] = {}
        short_only_by_key: Dict[str, int] = {}
        dead_by_label_by_key: Dict[str, Dict[str, int]] = {
            dead_label: {} for dead_label, _ in self._pruning_dead_thresholds
        }
        dead_epoch_by_label: Dict[str, int] = {
            dead_label: 0 for dead_label, _ in self._pruning_dead_thresholds
        }
        median_usage_by_key: Dict[str, float] = {}
        median_dead_count_k001_by_key: Dict[str, int] = {}
        median_dead_count_k002_by_key: Dict[str, int] = {}
        median_dead_count_k001_epoch = 0
        median_dead_count_k002_epoch = 0
        pruned_age_values: List[torch.Tensor] = []
        pruned_epoch_total = 0

        for key, rows in pending_rows.items():
            if not rows:
                continue
            cols = pending_cols.get(key, [])
            n_pairs = min(len(rows), len(cols))
            pruned_epoch_total += int(max(0, n_pairs))
            if len(cols) != len(rows):
                continue

            active = (getattr(self, self.masks[key]).detach().cpu() > 0)
            age = self._get_age(key).detach().cpu().float()
            usage_short = self._get_usage(key).detach().cpu().float().clamp_min(0.0)
            usage_long = self._get_usage_long(key).detach().cpu().float().clamp_min(0.0)

            mature_active = active & (age >= float(self.protection_period))
            if self.enable_pruning_natural_volume_diagnostic:
                for dead_label, dead_threshold in self._pruning_dead_thresholds:
                    dead_count = int((mature_active & (usage_short < float(dead_threshold))).sum().item())
                    dead_by_label_by_key[dead_label][key] = dead_count
                    dead_epoch_by_label[dead_label] += dead_count
            if self.enable_pruning_relative_median_diagnostic:
                mature_usage = usage_short[mature_active]
                if int(mature_usage.numel()) > 0:
                    median_usage = float(torch.median(mature_usage).item())
                    dead_k001 = int((mature_usage < (0.01 * median_usage)).sum().item())
                    dead_k002 = int((mature_usage < (0.02 * median_usage)).sum().item())
                else:
                    median_usage = 0.0
                    dead_k001 = 0
                    dead_k002 = 0
                median_usage_by_key[key] = float(median_usage)
                median_dead_count_k001_by_key[key] = int(dead_k001)
                median_dead_count_k002_by_key[key] = int(dead_k002)
                median_dead_count_k001_epoch += int(dead_k001)
                median_dead_count_k002_epoch += int(dead_k002)

            if not self.enable_pruning_persistence_diagnostic:
                continue

            ref_mask = mature_active if int(mature_active.sum().item()) > 0 else active
            ref_short = usage_short[ref_mask]
            ref_long = usage_long[ref_mask]
            if int(ref_short.numel()) <= 0 or int(ref_long.numel()) <= 0:
                continue

            short_thr = float(torch.quantile(ref_short, q).item())
            long_thr = float(torch.quantile(ref_long, q).item())

            pruned_mask = torch.zeros_like(active, dtype=torch.bool)
            rows_t = torch.tensor(rows, dtype=torch.long)
            cols_t = torch.tensor(cols, dtype=torch.long)
            pruned_mask[rows_t, cols_t] = True
            mature_pruned = pruned_mask & (age >= float(self.protection_period))
            if int(mature_pruned.sum().item()) <= 0:
                continue

            low_short = usage_short <= short_thr
            low_long = usage_long <= long_thr
            persistent_mask = mature_pruned & low_short & low_long
            short_only_mask = mature_pruned & low_short & (~low_long)

            persistent_count = int(persistent_mask.sum().item())
            short_only_count = int(short_only_mask.sum().item())
            persistent_by_key[key] = persistent_count
            short_only_by_key[key] = short_only_count

            pruned_age_values.append(age[mature_pruned].reshape(-1))

        persistent_epoch = int(sum(persistent_by_key.values()))
        short_only_epoch = int(sum(short_only_by_key.values()))
        denom = max(1.0, float(persistent_epoch + short_only_epoch))

        age_mean = 0.0
        age_p25 = 0.0
        age_p75 = 0.0
        if pruned_age_values:
            ages = torch.cat(pruned_age_values, dim=0).float()
            if int(ages.numel()) > 0:
                age_mean = float(ages.mean().item())
                age_p25 = float(torch.quantile(ages, 0.25).item())
                age_p75 = float(torch.quantile(ages, 0.75).item())

        if self.enable_pruning_persistence_diagnostic:
            self.pruned_persistent_low_total += int(persistent_epoch)
            self.pruned_short_only_low_total += int(short_only_epoch)
        if self.enable_pruning_natural_volume_diagnostic:
            self._dead_pruned_observed_total += int(pruned_epoch_total)
            for dead_label, _ in self._pruning_dead_thresholds:
                self._dead_count_totals_by_label[dead_label] = int(
                    self._dead_count_totals_by_label.get(dead_label, 0)
                    + int(dead_epoch_by_label.get(dead_label, 0))
                )
        if self.enable_pruning_relative_median_diagnostic:
            self._median_dead_pruned_observed_total += int(pruned_epoch_total)
            self._median_dead_count_k001_total += int(median_dead_count_k001_epoch)
            self._median_dead_count_k002_total += int(median_dead_count_k002_epoch)

        total_denom = max(1.0, float(self.pruned_persistent_low_total + self.pruned_short_only_low_total))
        short_only_ratio_total = float(float(self.pruned_short_only_low_total) / total_denom)

        out: Dict[str, Any] = {
            "pruned_persistent_low_epoch": float(persistent_epoch),
            "pruned_short_only_low_epoch": float(short_only_epoch),
            "pruned_persistent_low_ratio_epoch": float(persistent_epoch / denom),
            "pruned_short_only_low_ratio_epoch": float(short_only_epoch / denom),
            "pruned_persistent_low_total": float(self.pruned_persistent_low_total),
            "pruned_short_only_low_total": float(self.pruned_short_only_low_total),
            "pruned_short_only_low_ratio_total": float(short_only_ratio_total),
            "pruned_persistent_low_by_key": {
                str(k): int(v) for k, v in persistent_by_key.items()
            },
            "pruned_short_only_low_by_key": {
                str(k): int(v) for k, v in short_only_by_key.items()
            },
            "pruned_age_mean_epoch": float(age_mean),
            "pruned_age_p25_epoch": float(age_p25),
            "pruned_age_p75_epoch": float(age_p75),
        }
        for dead_label, _ in self._pruning_dead_thresholds:
            dead_epoch = int(dead_epoch_by_label.get(dead_label, 0))
            dead_total = int(self._dead_count_totals_by_label.get(dead_label, 0))
            out[f"dead_count_lt_{dead_label}_epoch"] = float(dead_epoch)
            out[f"dead_count_lt_{dead_label}_total"] = float(dead_total)
            out[f"dead_to_pruned_ratio_lt_{dead_label}_epoch"] = float(
                dead_epoch / max(1.0, float(pruned_epoch_total))
            )
            out[f"dead_to_pruned_ratio_lt_{dead_label}_total"] = float(
                dead_total / max(1.0, float(self._dead_pruned_observed_total))
            )
            out[f"dead_count_lt_{dead_label}_by_key"] = {
                str(k): int(v)
                for k, v in dead_by_label_by_key.get(dead_label, {}).items()
            }
        out["median_dead_count_k001_epoch"] = float(median_dead_count_k001_epoch)
        out["median_dead_count_k001_total"] = float(self._median_dead_count_k001_total)
        out["median_dead_to_pruned_ratio_k001_epoch"] = float(
            float(median_dead_count_k001_epoch) / max(1.0, float(pruned_epoch_total))
        )
        out["median_dead_to_pruned_ratio_k001_total"] = float(
            float(self._median_dead_count_k001_total)
            / max(1.0, float(self._median_dead_pruned_observed_total))
        )
        out["median_dead_count_k002_epoch"] = float(median_dead_count_k002_epoch)
        out["median_dead_count_k002_total"] = float(self._median_dead_count_k002_total)
        out["median_dead_to_pruned_ratio_k002_epoch"] = float(
            float(median_dead_count_k002_epoch) / max(1.0, float(pruned_epoch_total))
        )
        out["median_dead_to_pruned_ratio_k002_total"] = float(
            float(self._median_dead_count_k002_total)
            / max(1.0, float(self._median_dead_pruned_observed_total))
        )
        out["median_usage_by_key"] = {str(k): float(v) for k, v in median_usage_by_key.items()}
        out["median_dead_count_k001_by_key"] = {
            str(k): int(v) for k, v in median_dead_count_k001_by_key.items()
        }
        out["median_dead_count_k002_by_key"] = {
            str(k): int(v) for k, v in median_dead_count_k002_by_key.items()
        }
        return out

    def _matrix_survival_pruning_with_protection(
        self,
        keep_ratio: float = 0.9,
        max_prune_count: Optional[int] = None,
        enable_replay_tag_protection: bool = False,
        replay_tag_quantile: float = 0.90,
        replay_tag_min: float = 0.0,
        tagged_prune_pressure: float = 0.0,
        tagged_prune_pressure_slope: float = 0.0,
        max_tagged_prune_ratio: float = 0.0,
    ) -> int:
        protected_count = 0
        protected_young_total = 0
        protected_tagged_total = 0
        protected_tenured_total = 0
        tagged_active_total = 0
        released_tagged_total = 0
        tag_thresholds: List[float] = []

        survival_by_key = self.compute_survival_scores()
        candidate_meta: Dict[str, Dict[str, torch.Tensor]] = {}
        prunable_count_by_key: Dict[str, int] = {}
        usage_mean_by_key: Dict[str, float] = {}
        usage_peak_by_key: Dict[str, float] = {}
        survival_mean_by_key: Dict[str, float] = {}
        survival_p95_by_key: Dict[str, float] = {}
        survival_max_by_key: Dict[str, float] = {}
        threshold_by_key: Dict[str, float] = {}
        prune_gap_mass_by_key: Dict[str, float] = {}
        effective_prune_mass_by_key: Dict[str, float] = {}
        creation_need_by_key: Dict[str, float] = {}
        creation_need_ema_by_key: Dict[str, float] = {}
        prune_pressure_by_key: Dict[str, float] = {}
        positive_gap_count_by_key: Dict[str, int] = {}
        tenure_excluded_by_key: Dict[str, int] = {}

        p = float(max(0.0, min(1.0, tagged_prune_pressure)))
        slope = float(max(0.0, tagged_prune_pressure_slope))
        max_ratio = float(max(0.0, min(1.0, max_tagged_prune_ratio)))
        tagged_release_ratio = float(max(0.0, min(max_ratio, slope * p)))

        q = float(max(0.0, min(1.0, replay_tag_quantile)))
        tmin = float(max(0.0, replay_tag_min))

        for key in self.weights.keys():
            active = (getattr(self, self.masks[key]).detach().cpu() > 0)
            if int(active.sum().item()) == 0:
                continue

            age = self._get_age(key).detach().cpu()
            usage_mean = self._get_usage(key).detach().cpu().float()
            usage_peak = self._get_usage_peak(key).detach().cpu().float()
            survival = survival_by_key[key].detach().cpu().float()

            active_survival = survival[active]
            usage_mean_by_key[key] = float(usage_mean[active].mean().item()) if int(active_survival.numel()) > 0 else 0.0
            usage_peak_by_key[key] = float(usage_peak[active].mean().item()) if int(active_survival.numel()) > 0 else 0.0
            survival_mean_by_key[key] = float(active_survival.mean().item()) if int(active_survival.numel()) > 0 else 0.0
            survival_p95_by_key[key] = (
                float(torch.quantile(active_survival, 0.95).item()) if int(active_survival.numel()) > 0 else 0.0
            )
            survival_max_by_key[key] = float(active_survival.max().item()) if int(active_survival.numel()) > 0 else 0.0

            young_protected = (age < self.protection_period) & active
            protected_young_total += int(young_protected.sum().item())

            tagged_mask = torch.zeros_like(active)
            released_tagged = torch.zeros_like(active)

            if enable_replay_tag_protection:
                tag = self._get_replay_tag(key).detach().cpu().float()
                vals = tag[active]
                if int(vals.numel()) > 0:
                    thr = max(tmin, float(torch.quantile(vals, q).item()))
                    tag_thresholds.append(thr)
                    tagged_mask = (tag >= thr) & active & ~young_protected
                    tagged_active_total += int(tagged_mask.sum().item())

                    if tagged_release_ratio > 0.0 and int(tagged_mask.sum().item()) > 0:
                        n_release = int(math.ceil(tagged_release_ratio * int(tagged_mask.sum().item())))
                        n_release = max(0, min(int(tagged_mask.sum().item()), n_release))
                        if n_release > 0:
                            idx_tagged = tagged_mask.nonzero(as_tuple=False)
                            vals_tagged = tag[tagged_mask]
                            order = torch.argsort(vals_tagged, descending=False)
                            chosen = idx_tagged[order[:n_release]]
                            released_tagged[chosen[:, 0], chosen[:, 1]] = True
                            released_tagged_total += int(n_release)

            protected_tagged = tagged_mask & ~released_tagged
            protected_tagged_total += int(protected_tagged.sum().item())

            probationary_protected = self._probationary_edge_mask(key, active)
            tenured_grace_protected = self._tenured_grace_edge_mask(key, active)
            tenured_mask = (self._get_tenure_mask(key).detach().cpu() > 0) & active
            protected_tenured = tenured_mask & active & ~young_protected & ~protected_tagged
            protected_tenured_total += int(protected_tenured.sum().item())
            tenure_excluded_by_key[key] = int(protected_tenured.sum().item())
            protected = (
                young_protected
                | protected_tagged
                | probationary_protected
                | tenured_grace_protected
                | protected_tenured
            )
            prunable = active & ~protected
            protected_count += int(protected.sum().item())
            prunable_count = int(prunable.sum().item())
            prunable_count_by_key[key] = prunable_count
            if prunable_count <= 0:
                threshold_by_key[key] = 0.0
                prune_gap_mass_by_key[key] = 0.0
                effective_prune_mass_by_key[key] = 0.0
                creation_need_by_key[key] = 0.0
                creation_need_ema_by_key[key] = float(self._pruning_creation_need_ema.get(key) or 0.0)
                prune_pressure_by_key[key] = 1.0
                positive_gap_count_by_key[key] = 0
                continue

            ref_mask = active
            if self.pruning_ref_mature_only:
                mature_active = active & (age >= float(self.protection_period))
                if int(mature_active.sum().item()) > 0:
                    ref_mask = mature_active
            ref_scores = survival[ref_mask]
            threshold = float(torch.quantile(ref_scores, self.pruning_active_ref_quantile).item()) if int(ref_scores.numel()) > 0 else 0.0
            threshold_by_key[key] = threshold

            idx = prunable.nonzero(as_tuple=False)
            survival_vals = survival[prunable]
            gap_vals = (threshold - survival_vals).clamp_min(0.0)
            tag_flags = tagged_mask[prunable]
            positive_mask = gap_vals > 0
            positive_count = int(positive_mask.sum().item())
            positive_gap_count_by_key[key] = positive_count
            if positive_count > 0:
                topk = min(self.pruning_priority_topk, positive_count)
                prune_gap_mass_by_key[key] = float(torch.topk(gap_vals[positive_mask], k=topk).values.sum().item())
            else:
                prune_gap_mass_by_key[key] = 0.0
            effective_prune_mass_by_key[key] = float(prune_gap_mass_by_key[key])
            creation_need_by_key[key] = 0.0
            creation_need_ema_by_key[key] = float(self._pruning_creation_need_ema.get(key) or 0.0)
            prune_pressure_by_key[key] = 1.0

            candidate_meta[key] = {
                "idx": idx.to(dtype=torch.long),
                "survival": survival_vals,
                "gap": gap_vals,
                "tagged": tag_flags.reshape(-1).to(dtype=torch.bool),
                "selected": torch.zeros(prunable_count, dtype=torch.bool),
            }

        self.protected_from_pruning = protected_count
        total_prunable = int(sum(prunable_count_by_key.values()))
        keep_count = int(round(float(keep_ratio) * total_prunable))
        keep_count = max(0, min(total_prunable, keep_count))
        base_prune_count = max(0, total_prunable - keep_count)
        effective_max_prune_count = None
        prune_count = int(base_prune_count)
        if max_prune_count is not None:
            effective_max_prune_count = max(0, min(total_prunable, int(max_prune_count)))
            prune_count = min(prune_count, effective_max_prune_count)

        if self.pruning_couple_creation_need and candidate_meta:
            creation_stats = getattr(self, "last_creation_stats", {}) or {}
            raw_need_map = creation_stats.get("relative_gap_mass_by_key", {}) or {}
            need_keys = list(candidate_meta.keys())
            alpha = float(self.pruning_creation_need_ema_alpha)
            for key in need_keys:
                need_val = float(raw_need_map.get(key, 0.0))
                creation_need_by_key[key] = need_val
                prev = self._pruning_creation_need_ema.get(key)
                if prev is None:
                    ema = need_val
                else:
                    ema = alpha * need_val + (1.0 - alpha) * float(prev)
                self._pruning_creation_need_ema[key] = float(ema)
                creation_need_ema_by_key[key] = float(ema)

            max_need = max([float(v) for v in creation_need_ema_by_key.values()], default=0.0)
            coupling_alpha = float(self.pruning_creation_coupling_alpha)
            for key in need_keys:
                if max_need > 0.0 and coupling_alpha > 0.0:
                    need_norm = float(creation_need_ema_by_key.get(key, 0.0)) / max_need
                    pressure = 1.0 / (1.0 + coupling_alpha * need_norm)
                else:
                    pressure = 1.0
                prune_pressure_by_key[key] = float(pressure)
                effective_prune_mass_by_key[key] = float(prune_gap_mass_by_key.get(key, 0.0)) * float(pressure)

        if total_prunable <= 0 or prune_count <= 0 or not candidate_meta:
            self.last_pruning_stats = {
                "policy": self.pruning_policy,
                "survival_score_mode": "sqrt_mean_peak",
                "base_prune_count": float(base_prune_count),
                "effective_prune_count": float(prune_count),
                "max_prune_count": (
                    float(effective_max_prune_count) if effective_max_prune_count is not None else None
                ),
                "pruned_tagged": 0.0,
                "pruned_untagged": 0.0,
                "protected_tagged": float(protected_tagged_total),
                "protected_young": float(protected_young_total),
                "protected_tenured": float(protected_tenured_total),
                "released_tagged": float(released_tagged_total),
                "tag_threshold_mean": float(sum(tag_thresholds) / max(1, len(tag_thresholds))) if tag_thresholds else 0.0,
                "tagged_active_total": float(tagged_active_total),
                "total_prunable": float(total_prunable),
                "tenured_total": float((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_total", 0.0)),
                "tenured_count_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_count_by_key", {})),
                "tenured_share_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_share_by_key", {})),
                "tenure_excluded_by_key": tenure_excluded_by_key,
                "prunable_count_by_key": prunable_count_by_key,
                "usage_mean_by_key": usage_mean_by_key,
                "usage_peak_by_key": usage_peak_by_key,
                "survival_mean_by_key": survival_mean_by_key,
                "survival_p95_by_key": survival_p95_by_key,
                "survival_max_by_key": survival_max_by_key,
                "threshold_by_key": threshold_by_key,
                "prune_gap_mass_by_key": prune_gap_mass_by_key,
                "effective_prune_mass_by_key": effective_prune_mass_by_key,
                "creation_need_by_key": creation_need_by_key,
                "creation_need_ema_by_key": creation_need_ema_by_key,
                "prune_pressure_by_key": prune_pressure_by_key,
                "positive_gap_count_by_key": positive_gap_count_by_key,
                "pruned_by_key": {},
                **self._empty_pruning_persistence_stats(),
            }
            return 0

        masses = {
            key: float(effective_prune_mass_by_key.get(key, 0.0))
            for key in candidate_meta.keys()
            if int(positive_gap_count_by_key.get(key, 0)) > 0
        }
        capacities = {
            key: int(positive_gap_count_by_key.get(key, 0))
            for key in candidate_meta.keys()
        }
        planned_by_key = {key: 0 for key in candidate_meta.keys()}
        total_mass = float(sum(masses.values()))
        remaining = int(prune_count)

        if total_mass > 0.0:
            raw_targets: Dict[str, float] = {
                key: float(prune_count) * float(masses[key]) / total_mass for key in masses.keys()
            }
            fractional: List[Tuple[float, str]] = []
            for key, raw in raw_targets.items():
                base = min(capacities[key], int(math.floor(raw)))
                planned_by_key[key] = base
                remaining -= base
                fractional.append((raw - float(base), key))
            fractional.sort(key=lambda item: item[0], reverse=True)
            for _, key in fractional:
                if remaining <= 0:
                    break
                if planned_by_key[key] < capacities[key]:
                    planned_by_key[key] += 1
                    remaining -= 1
            if remaining > 0:
                refill_keys = sorted(masses.keys(), key=lambda key: masses[key], reverse=True)
                guard = 0
                while remaining > 0 and refill_keys:
                    progressed = False
                    for key in refill_keys:
                        if remaining <= 0:
                            break
                        if planned_by_key[key] < capacities[key]:
                            planned_by_key[key] += 1
                            remaining -= 1
                            progressed = True
                    guard += 1
                    if not progressed or guard > max(1, len(refill_keys)) * 4:
                        break

        def _order_gap_then_survival(gap_vals: torch.Tensor, survival_vals: torch.Tensor) -> torch.Tensor:
            try:
                secondary = torch.argsort(survival_vals, descending=False, stable=True)
            except TypeError:
                secondary = torch.argsort(survival_vals, descending=False)
            gap_secondary = gap_vals[secondary]
            try:
                primary = torch.argsort(gap_secondary, descending=True, stable=True)
            except TypeError:
                primary = torch.argsort(gap_secondary, descending=True)
            return secondary[primary]

        pruned_by_key: Dict[str, int] = {}
        pruned_tagged = 0
        pruned_untagged = 0
        pending_rows: Dict[str, List[int]] = {}
        pending_cols: Dict[str, List[int]] = {}

        selected_total = 0
        for key, quota in planned_by_key.items():
            if quota <= 0:
                continue
            meta = candidate_meta[key]
            positive_local = (meta["gap"] > 0).nonzero(as_tuple=False).reshape(-1)
            if int(positive_local.numel()) <= 0:
                continue
            quota = min(int(quota), int(positive_local.numel()))
            local_order = _order_gap_then_survival(meta["gap"][positive_local], meta["survival"][positive_local])
            chosen_local = positive_local[local_order[:quota]]
            meta["selected"][chosen_local] = True
            selected_total += int(chosen_local.numel())

        remaining = max(0, prune_count - selected_total)
        if remaining > 0:
            fallback_survival: List[torch.Tensor] = []
            fallback_key_ids: List[torch.Tensor] = []
            fallback_local_idx: List[torch.Tensor] = []
            key_names: List[str] = []
            for key, meta in candidate_meta.items():
                available = ~meta["selected"]
                if not bool(available.any()):
                    continue
                local_idx = available.nonzero(as_tuple=False).reshape(-1)
                if int(local_idx.numel()) <= 0:
                    continue
                kid = len(key_names)
                key_names.append(key)
                fallback_survival.append(meta["survival"][local_idx].reshape(-1))
                fallback_key_ids.append(torch.full((int(local_idx.numel()),), kid, dtype=torch.long))
                fallback_local_idx.append(local_idx.to(dtype=torch.long))
            if fallback_survival:
                survival_all = torch.cat(fallback_survival, dim=0)
                key_ids_all = torch.cat(fallback_key_ids, dim=0)
                local_idx_all = torch.cat(fallback_local_idx, dim=0)
                try:
                    fallback_order = torch.argsort(survival_all, descending=False, stable=True)
                except TypeError:
                    fallback_order = torch.argsort(survival_all, descending=False)
                fallback_sel = fallback_order[:remaining]
                for kid, local_idx in zip(key_ids_all[fallback_sel].tolist(), local_idx_all[fallback_sel].tolist()):
                    candidate_meta[key_names[int(kid)]]["selected"][int(local_idx)] = True

        for key, meta in candidate_meta.items():
            chosen = meta["selected"].nonzero(as_tuple=False).reshape(-1)
            pruned_count_key = int(chosen.numel())
            if pruned_count_key <= 0:
                continue
            pruned_by_key[key] = pruned_count_key
            rows = meta["idx"][chosen, 0].tolist()
            cols = meta["idx"][chosen, 1].tolist()
            pending_rows[key] = rows
            pending_cols[key] = cols
            tagged_sel = meta["tagged"][chosen]
            tagged_count = int(tagged_sel.sum().item())
            pruned_tagged += tagged_count
            pruned_untagged += int(pruned_count_key - tagged_count)

        prune_count = int(sum(pruned_by_key.values()))
        if prune_count <= 0:
            self.last_pruning_stats = {
                "policy": self.pruning_policy,
                "survival_score_mode": "sqrt_mean_peak",
                "base_prune_count": float(base_prune_count),
                "effective_prune_count": float(prune_count),
                "max_prune_count": (
                    float(effective_max_prune_count) if effective_max_prune_count is not None else None
                ),
                "pruned_tagged": 0.0,
                "pruned_untagged": 0.0,
                "protected_tagged": float(protected_tagged_total),
                "protected_young": float(protected_young_total),
                "protected_tenured": float(protected_tenured_total),
                "released_tagged": float(released_tagged_total),
                "tag_threshold_mean": float(sum(tag_thresholds) / max(1, len(tag_thresholds))) if tag_thresholds else 0.0,
                "tagged_active_total": float(tagged_active_total),
                "total_prunable": float(total_prunable),
                "tenured_total": float((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_total", 0.0)),
                "tenured_count_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_count_by_key", {})),
                "tenured_share_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_share_by_key", {})),
                "tenure_excluded_by_key": tenure_excluded_by_key,
                "prunable_count_by_key": prunable_count_by_key,
                "usage_mean_by_key": usage_mean_by_key,
                "usage_peak_by_key": usage_peak_by_key,
                "survival_mean_by_key": survival_mean_by_key,
                "survival_p95_by_key": survival_p95_by_key,
                "survival_max_by_key": survival_max_by_key,
                "threshold_by_key": threshold_by_key,
                "prune_gap_mass_by_key": prune_gap_mass_by_key,
                "effective_prune_mass_by_key": effective_prune_mass_by_key,
                "creation_need_by_key": creation_need_by_key,
                "creation_need_ema_by_key": creation_need_ema_by_key,
                "prune_pressure_by_key": prune_pressure_by_key,
                "positive_gap_count_by_key": positive_gap_count_by_key,
                "pruned_by_key": {},
                **self._empty_pruning_persistence_stats(),
            }
            return 0

        pruning_persistence_stats = self._compute_pruning_persistence_stats(
            pending_rows=pending_rows,
            pending_cols=pending_cols,
        )
        with torch.no_grad():
            for key, rows in pending_rows.items():
                if not rows:
                    continue
                cols = pending_cols[key]
                dev = self.weights[key].device
                rows_t = torch.tensor(rows, device=dev, dtype=torch.long)
                cols_t = torch.tensor(cols, device=dev, dtype=torch.long)

                mask = getattr(self, self.masks[key])
                mask[rows_t, cols_t] = 0.0
                self.weights[key].data[rows_t, cols_t] = 0.0

                self._get_replay_tag(key)[rows_t, cols_t] = 0.0
                self._get_usage(key)[rows_t, cols_t] = 0.0
                self._get_usage_peak(key)[rows_t, cols_t] = 0.0
                self._get_age(key)[rows_t, cols_t] = 0.0
                self._get_tenure_mask(key)[rows_t, cols_t] = 0.0
                self._get_tenure_score_ema(key)[rows_t, cols_t] = 0.0
                self._get_tenure_streak(key)[rows_t, cols_t] = 0

        self.pruning_count += int(prune_count)
        ordered_keys = sorted(prune_gap_mass_by_key.keys(), key=lambda key: prune_gap_mass_by_key[key], reverse=True)
        self.last_pruning_stats = {
            "policy": self.pruning_policy,
            "survival_score_mode": "sqrt_mean_peak",
            "base_prune_count": float(base_prune_count),
            "effective_prune_count": float(prune_count),
            "max_prune_count": (
                float(effective_max_prune_count) if effective_max_prune_count is not None else None
            ),
            "pruned_tagged": float(pruned_tagged),
            "pruned_untagged": float(pruned_untagged),
            "protected_tagged": float(protected_tagged_total),
            "protected_young": float(protected_young_total),
            "protected_tenured": float(protected_tenured_total),
            "released_tagged": float(released_tagged_total),
            "tag_threshold_mean": float(sum(tag_thresholds) / max(1, len(tag_thresholds))) if tag_thresholds else 0.0,
            "tagged_active_total": float(tagged_active_total),
            "total_prunable": float(total_prunable),
            "tenured_total": float((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_total", 0.0)),
            "tenured_count_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_count_by_key", {})),
            "tenured_share_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_share_by_key", {})),
            "tenure_excluded_by_key": tenure_excluded_by_key,
            "ordered_keys": ordered_keys,
            "prunable_count_by_key": prunable_count_by_key,
            "usage_mean_by_key": usage_mean_by_key,
            "usage_peak_by_key": usage_peak_by_key,
            "survival_mean_by_key": survival_mean_by_key,
            "survival_p95_by_key": survival_p95_by_key,
            "survival_max_by_key": survival_max_by_key,
            "threshold_by_key": threshold_by_key,
            "prune_gap_mass_by_key": prune_gap_mass_by_key,
            "effective_prune_mass_by_key": effective_prune_mass_by_key,
            "creation_need_by_key": creation_need_by_key,
            "creation_need_ema_by_key": creation_need_ema_by_key,
            "prune_pressure_by_key": prune_pressure_by_key,
            "positive_gap_count_by_key": positive_gap_count_by_key,
            "pruned_by_key": pruned_by_key,
            **pruning_persistence_stats,
        }
        return int(prune_count)

    def _weighted_sample_indices(self, scores: torch.Tensor, num_samples: int) -> torch.Tensor:
        if num_samples <= 0 or int(scores.numel()) == 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        num_samples = min(int(num_samples), int(scores.numel()))
        weights = scores.detach().float().clamp_min(0.0)
        if float(weights.sum().item()) <= 0.0:
            perm = torch.randperm(int(scores.numel()), device=scores.device)
            return perm[:num_samples]
        probs = weights / weights.sum()
        return torch.multinomial(probs, num_samples=num_samples, replacement=False)

    def _ensure_epoch_runtime_state(self) -> None:
        epoch_tag = int(getattr(self, "current_epoch", 0))
        if epoch_tag == int(self._epoch_runtime_tag):
            return
        self._epoch_runtime_tag = int(epoch_tag)
        self._epoch_pathway_dropout_active = False
        self._epoch_pathway_dropout_rate_effective = 0.0
        self._epoch_pathway_dropout_candidate_channels_by_prefix = {}
        self._epoch_pathway_dropout_dropped_channels_by_prefix = {}
        self._epoch_pathway_dropout_protected_channels_by_prefix = {}
        self._epoch_local_reinforce_channels = {}
        self._epoch_local_reinforce_edges_up_by_prefix = {}
        self._epoch_local_reinforce_edges_down_by_prefix = {}
        self._epoch_local_reinforce_edges = 0
        self._epoch_local_reinforce_budget_used = 0
        self._phase1_new_hidden_attempts_epoch = 0
        self._phase1_retested_hidden_attempts_epoch = 0

    def _pathway_dropout_rate_for_epoch(self) -> float:
        if not self.pathway_dropout_enabled:
            return 0.0
        epoch = int(getattr(self, "current_epoch", 0))
        if epoch <= 0:
            return 0.0
        if self.pathway_dropout_end_epoch > 0 and epoch > int(self.pathway_dropout_end_epoch):
            return 0.0
        if self.pathway_dropout_warmup_epochs > 0:
            warm = min(1.0, float(epoch) / float(max(1, self.pathway_dropout_warmup_epochs)))
        else:
            warm = 1.0
        return float(self.pathway_dropout_rate) * float(max(0.0, min(1.0, warm)))

    def _protected_channel_ids_for_dropout(self, prefix: str) -> Dict[int, bool]:
        protected: Dict[int, bool] = {}
        for channel in self._probationary_channels.values():
            if str(channel.get("prefix", "")) == str(prefix):
                protected[int(channel["hidden_idx"])] = True
        for channel in self._tenured_grace_channels.values():
            if str(channel.get("prefix", "")) == str(prefix):
                protected[int(channel["hidden_idx"])] = True
        for channel in self._local_reinforce_queue.values():
            if str(channel.get("prefix", "")) == str(prefix):
                protected[int(channel["hidden_idx"])] = True
        return protected

    def _compute_local_need_scores(
        self,
        key: str,
        idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if int(idx.numel()) <= 0:
            empty = torch.empty(0, dtype=torch.float32)
            return empty, empty
        scores = self._get_creation_score(key).detach().cpu().float()
        dirs = self._get_creation_direction(key).detach().cpu().float()
        raw_scores = scores[idx[:, 0], idx[:, 1]]
        cand_dirs = dirs[idx[:, 0], idx[:, 1]]
        align_scores = torch.minimum(raw_scores, cand_dirs.abs())
        interference_scores = (raw_scores - align_scores).clamp_min(0.0)
        active = (getattr(self, self.masks[key]).detach().cpu() > 0)
        age = self._get_age(key).detach().cpu()
        mature_active = active & (age >= float(self.protection_period))
        support_mask = mature_active if int(mature_active.sum().item()) > 0 else active
        usage = self._get_usage(key).detach().cpu().float()
        support_by_col = (usage * support_mask.float()).sum(dim=0)
        support_positive = support_by_col[support_by_col > 0]
        support_scale = (
            float(torch.quantile(support_positive, 0.75).item())
            if int(support_positive.numel()) > 0
            else 1.0
        )
        scarcity_by_col = (support_scale / (1e-8 + support_by_col)).clamp(0.25, 4.0)
        candidate_scarcity = scarcity_by_col[idx[:, 1]]
        need_scores = raw_scores * candidate_scarcity
        need_interference = interference_scores * candidate_scarcity
        return need_scores, need_interference

    def _create_edge_immediately(self, key: str, src_n: int, dst_n: int, val: float) -> bool:
        with torch.no_grad():
            mask = getattr(self, self.masks[key])
            if bool(mask[src_n, dst_n].item() > 0):
                return False
            mask[src_n, dst_n] = 1.0
            self.weights[key].data[src_n, dst_n] = float(val)
            self._get_usage(key)[src_n, dst_n] = 0.0
            self._get_usage_peak(key)[src_n, dst_n] = 0.0
            self._get_age(key)[src_n, dst_n] = 0.0
            self._get_replay_tag(key)[src_n, dst_n] = 0.0
            self._get_creation_score(key)[src_n, dst_n] = 0.0
            self._get_creation_direction(key)[src_n, dst_n] = 0.0
        self.creation_count += 1
        return True

    def _apply_channel_pathway_dropout(
        self,
        prefix: str,
        hidden_activation: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._ensure_epoch_runtime_state()
        if (
            (not self.training)
            or (not self.pathway_dropout_enabled)
            or str(prefix) not in set(self.pathway_dropout_prefixes)
        ):
            return hidden_activation
        rate_effective = float(self._pathway_dropout_rate_for_epoch())
        if rate_effective <= 0.0 or self.pathway_dropout_top_frac <= 0.0:
            return hidden_activation

        dominance = self._get_hidden_usage_mean(prefix).detach().to(hidden_activation.device).float().clamp_min(0.0)
        fire = self._get_hidden_fire_rate(prefix).detach().to(hidden_activation.device).float().clamp(0.0, 1.0)
        dominance = dominance * (0.5 + fire)
        if int(dominance.numel()) <= 0:
            return hidden_activation

        protected_ids = self._protected_channel_ids_for_dropout(prefix)
        protected_mask = torch.zeros_like(dominance, dtype=torch.bool)
        for hidden_idx in protected_ids.keys():
            if 0 <= int(hidden_idx) < int(protected_mask.numel()):
                protected_mask[int(hidden_idx)] = True
        eligible_mask = ~protected_mask
        if int(eligible_mask.sum().item()) <= 0:
            self._epoch_pathway_dropout_active = True
            self._epoch_pathway_dropout_rate_effective = max(
                float(self._epoch_pathway_dropout_rate_effective), rate_effective
            )
            self._epoch_pathway_dropout_protected_channels_by_prefix[prefix] = int(protected_mask.sum().item())
            return hidden_activation

        quant = max(0.0, min(1.0, 1.0 - float(self.pathway_dropout_top_frac)))
        eligible_vals = dominance[eligible_mask]
        threshold = float(torch.quantile(eligible_vals, quant).item()) if int(eligible_vals.numel()) > 0 else 0.0
        candidate_mask = eligible_mask & (dominance >= threshold)
        candidate_count = int(candidate_mask.sum().item())
        protected_count = int(protected_mask.sum().item())
        dropped_count = 0
        if candidate_count > 0:
            draws = torch.rand(candidate_count, device=hidden_activation.device) < rate_effective
            if bool(draws.any().item()):
                candidate_idx = candidate_mask.nonzero(as_tuple=False).flatten()
                dropped_idx = candidate_idx[draws]
                gate = torch.ones(int(hidden_activation.size(-1)), device=hidden_activation.device, dtype=hidden_activation.dtype)
                gate[dropped_idx] = 0.0
                view_shape = [1] * hidden_activation.dim()
                view_shape[-1] = int(hidden_activation.size(-1))
                hidden_activation = hidden_activation * gate.view(*view_shape)
                dropped_count = int(dropped_idx.numel())

        self._epoch_pathway_dropout_active = True
        self._epoch_pathway_dropout_rate_effective = max(float(self._epoch_pathway_dropout_rate_effective), rate_effective)
        self._epoch_pathway_dropout_candidate_channels_by_prefix[prefix] = (
            int(self._epoch_pathway_dropout_candidate_channels_by_prefix.get(prefix, 0)) + candidate_count
        )
        self._epoch_pathway_dropout_dropped_channels_by_prefix[prefix] = (
            int(self._epoch_pathway_dropout_dropped_channels_by_prefix.get(prefix, 0)) + dropped_count
        )
        self._epoch_pathway_dropout_protected_channels_by_prefix[prefix] = (
            int(self._epoch_pathway_dropout_protected_channels_by_prefix.get(prefix, 0)) + protected_count
        )
        return hidden_activation

    def run_local_channel_reinforcement(self, max_edges: int) -> Dict[str, Any]:
        self._ensure_epoch_runtime_state()
        if (
            (not self.local_reinforce_enabled)
            or int(max_edges) <= 0
            or self.local_reinforce_max_edges_per_epoch <= 0
            or self.local_reinforce_edges_per_channel_per_epoch <= 0
            or self.local_reinforce_max_edges_per_channel_total <= 0
            or not self._local_reinforce_queue
        ):
            return {
                "local_reinforce_queue_size_epoch": float(len(self._local_reinforce_queue)),
                "local_reinforce_channels_epoch": 0.0,
                "local_reinforce_edges_epoch": 0.0,
                "local_reinforce_edges_up_by_prefix": {},
                "local_reinforce_edges_down_by_prefix": {},
                "local_reinforce_budget_used_epoch": 0.0,
                "local_reinforce_channels_total": float(self.local_reinforce_channels_total),
                "local_reinforce_edges_total": float(self.local_reinforce_edges_total),
            }

        remaining_budget = min(int(max_edges), int(self.local_reinforce_max_edges_per_epoch))
        per_side_limit = max(1, int(math.ceil(float(self.local_reinforce_edges_per_channel_per_epoch) / 2.0)))
        expired_ids: List[int] = []

        for channel_id in sorted(self._local_reinforce_queue.keys()):
            if remaining_budget <= 0:
                break
            channel = self._local_reinforce_queue[channel_id]
            if int(channel.get("remaining_epochs", 0)) <= 0:
                expired_ids.append(int(channel_id))
                continue
            if int(channel.get("reinforce_total_edges_added", 0)) >= int(self.local_reinforce_max_edges_per_channel_total):
                expired_ids.append(int(channel_id))
                continue

            prefix = str(channel["prefix"])
            h = int(channel["hidden_idx"])
            up_key = str(channel["up_key"])
            down_key = str(channel["down_key"])
            up_shadow = (getattr(self, self.masks[up_key]).detach().cpu() > 0)
            down_shadow = (getattr(self, self.masks[down_key]).detach().cpu() > 0)

            added_this_channel = 0
            for side in ("up", "down"):
                if remaining_budget <= 0:
                    break
                added_side = 0
                if added_side >= per_side_limit:
                    continue
                total_added = int(channel.get("reinforce_total_edges_added", 0))
                if total_added >= int(self.local_reinforce_max_edges_per_channel_total):
                    break

                if side == "up":
                    inactive = (~up_shadow[:, h]).nonzero(as_tuple=False).flatten().detach().cpu()
                    if int(inactive.numel()) <= 0:
                        continue
                    idx = torch.stack(
                        [inactive, torch.full_like(inactive, int(h))],
                        dim=1,
                    )
                    key = up_key
                else:
                    inactive = (~down_shadow[h, :]).nonzero(as_tuple=False).flatten().detach().cpu()
                    if int(inactive.numel()) <= 0:
                        continue
                    idx = torch.stack(
                        [torch.full_like(inactive, int(h)), inactive],
                        dim=1,
                    )
                    key = down_key

                need_scores, _ = self._compute_local_need_scores(key, idx)
                top_pool = min(int(idx.size(0)), int(self.local_reinforce_top_pool))
                pick_local = None
                if top_pool > 0 and int(need_scores.numel()) > 0 and float(need_scores.max().item()) > 0.0:
                    pool_idx = torch.topk(need_scores, k=top_pool, largest=True, sorted=False).indices
                    weighted = self._weighted_sample_indices(need_scores[pool_idx], 1)
                    if int(weighted.numel()) > 0:
                        pick_local = int(pool_idx[int(weighted[0].item())].item())
                if pick_local is None:
                    pick_local = int(torch.randint(int(idx.size(0)), (1,)).item())

                src_n = int(idx[pick_local, 0].item())
                dst_n = int(idx[pick_local, 1].item())
                direction = float(self._get_creation_direction(key).detach().cpu().float()[src_n, dst_n].item())
                init_scale = float(self.channel_bundle_init_scale)
                if abs(direction) < 1e-8:
                    val = float(torch.randn(1).item()) * init_scale
                else:
                    val = max(-3.0, min(3.0, direction)) * init_scale
                if not self._create_edge_immediately(key, src_n, dst_n, val):
                    continue

                if side == "up":
                    channel.setdefault("up_rows", []).append(int(src_n))
                    self._epoch_local_reinforce_edges_up_by_prefix[prefix] = (
                        int(self._epoch_local_reinforce_edges_up_by_prefix.get(prefix, 0)) + 1
                    )
                else:
                    channel.setdefault("down_cols", []).append(int(dst_n))
                    self._epoch_local_reinforce_edges_down_by_prefix[prefix] = (
                        int(self._epoch_local_reinforce_edges_down_by_prefix.get(prefix, 0)) + 1
                    )
                if int(channel_id) in self._tenured_grace_channels:
                    if side == "up":
                        self._tenured_grace_channels[int(channel_id)].setdefault("up_rows", []).append(int(src_n))
                    else:
                        self._tenured_grace_channels[int(channel_id)].setdefault("down_cols", []).append(int(dst_n))
                channel[f"reinforce_edges_added_{side}"] = int(channel.get(f"reinforce_edges_added_{side}", 0)) + 1
                channel["reinforce_total_edges_added"] = int(channel.get("reinforce_total_edges_added", 0)) + 1
                added_this_channel += 1
                added_side += 1
                remaining_budget -= 1
                self.local_reinforce_edges_total += 1
                self._epoch_local_reinforce_edges += 1
                self._epoch_local_reinforce_budget_used += 1

            channel["remaining_epochs"] = int(channel.get("remaining_epochs", 0)) - 1
            if added_this_channel > 0:
                self._epoch_local_reinforce_channels[int(channel_id)] = True
                if not bool(channel.get("reinforced_once", False)):
                    channel["reinforced_once"] = True
                    self.local_reinforce_channels_total += 1
            if (
                int(channel.get("remaining_epochs", 0)) <= 0
                or int(channel.get("reinforce_total_edges_added", 0)) >= int(self.local_reinforce_max_edges_per_channel_total)
            ):
                expired_ids.append(int(channel_id))

        for channel_id in expired_ids:
            self._local_reinforce_queue.pop(int(channel_id), None)

        return {
            "local_reinforce_queue_size_epoch": float(len(self._local_reinforce_queue)),
            "local_reinforce_channels_epoch": float(len(self._epoch_local_reinforce_channels)),
            "local_reinforce_edges_epoch": float(self._epoch_local_reinforce_edges),
            "local_reinforce_edges_up_by_prefix": {
                str(k): int(v) for k, v in self._epoch_local_reinforce_edges_up_by_prefix.items()
            },
            "local_reinforce_edges_down_by_prefix": {
                str(k): int(v) for k, v in self._epoch_local_reinforce_edges_down_by_prefix.items()
            },
            "local_reinforce_budget_used_epoch": float(self._epoch_local_reinforce_budget_used),
            "local_reinforce_channels_total": float(self.local_reinforce_channels_total),
            "local_reinforce_edges_total": float(self.local_reinforce_edges_total),
        }

    def _probationary_counts_by_prefix(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for channel in self._probationary_channels.values():
            prefix = str(channel.get("prefix", ""))
            counts[prefix] = counts.get(prefix, 0) + 1
        return counts

    def _probationary_edge_mask(self, key: str, reference: torch.Tensor) -> torch.Tensor:
        protected = torch.zeros_like(reference, dtype=torch.bool)
        for channel in self._probationary_channels.values():
            if str(channel.get("up_key")) == key:
                h = int(channel["hidden_idx"])
                for src_n in channel.get("up_rows", []):
                    protected[int(src_n), h] = True
            elif str(channel.get("down_key")) == key:
                h = int(channel["hidden_idx"])
                for dst_n in channel.get("down_cols", []):
                    protected[h, int(dst_n)] = True
        return protected

    def _channel_min_age_if_fully_active(self, channel: Dict[str, Any]) -> Optional[float]:
        up_key = str(channel["up_key"])
        down_key = str(channel["down_key"])
        h = int(channel["hidden_idx"])
        up_rows = [int(v) for v in channel.get("up_rows", [])]
        down_cols = [int(v) for v in channel.get("down_cols", [])]
        if not up_rows or not down_cols:
            return None

        up_mask = (getattr(self, self.masks[up_key]).detach().cpu() > 0)
        down_mask = (getattr(self, self.masks[down_key]).detach().cpu() > 0)
        up_rows_t = torch.tensor(up_rows, dtype=torch.long)
        down_cols_t = torch.tensor(down_cols, dtype=torch.long)
        up_active = up_mask[up_rows_t, h]
        down_active = down_mask[h, down_cols_t]
        if int(up_active.sum().item()) < len(up_rows) or int(down_active.sum().item()) < len(down_cols):
            return None

        up_age = self._get_age(up_key).detach().cpu().float()[up_rows_t, h]
        down_age = self._get_age(down_key).detach().cpu().float()[h, down_cols_t]
        return min(
            float(up_age.min().item()) if int(up_age.numel()) > 0 else 0.0,
            float(down_age.min().item()) if int(down_age.numel()) > 0 else 0.0,
        )

    def _cleanup_tenured_grace_channels(self) -> None:
        if not getattr(self, "_tenured_grace_channels", None):
            return
        grace_limit = float(self.channel_probe_period + self.tenured_channel_grace_period)
        expired_ids: List[int] = []
        for channel_id, channel in self._tenured_grace_channels.items():
            min_age = self._channel_min_age_if_fully_active(channel)
            if min_age is None or min_age >= grace_limit:
                expired_ids.append(int(channel_id))
        for channel_id in expired_ids:
            self._tenured_grace_channels.pop(channel_id, None)

    def _tenured_grace_edge_mask(self, key: str, reference: torch.Tensor) -> torch.Tensor:
        protected = torch.zeros_like(reference, dtype=torch.bool)
        if self.tenured_channel_grace_period <= 0:
            return protected
        self._cleanup_tenured_grace_channels()
        for channel in self._tenured_grace_channels.values():
            if str(channel.get("up_key")) == key:
                h = int(channel["hidden_idx"])
                for src_n in channel.get("up_rows", []):
                    protected[int(src_n), h] = True
            elif str(channel.get("down_key")) == key:
                h = int(channel["hidden_idx"])
                for dst_n in channel.get("down_cols", []):
                    protected[h, int(dst_n)] = True
        return protected

    def _register_probationary_channel(
        self,
        prefix: str,
        hidden_idx: int,
        up_key: str,
        down_key: str,
        up_rows: List[int],
        down_cols: List[int],
        epoch_created: int,
    ) -> None:
        self._ensure_epoch_runtime_state()
        if self.phase1_registry_enabled:
            registry_entry = self._phase1_registry_entry(prefix, hidden_idx, create=True)
            if registry_entry is not None:
                attempts_before = int(registry_entry.get("n_attempts_current_level", 0))
                registry_entry["n_attempts_total"] = int(registry_entry.get("n_attempts_total", 0)) + 1
                registry_entry["n_attempts_current_level"] = attempts_before + 1
                registry_entry["last_attempt_epoch"] = int(epoch_created)
                if attempts_before <= 0:
                    self._phase1_new_hidden_attempts_epoch += 1
                else:
                    self._phase1_retested_hidden_attempts_epoch += 1

        channel_id = int(self._next_probationary_channel_id)
        self._next_probationary_channel_id += 1
        self._probationary_channels[channel_id] = {
            "channel_id": channel_id,
            "prefix": str(prefix),
            "hidden_idx": int(hidden_idx),
            "up_key": str(up_key),
            "down_key": str(down_key),
            "up_rows": [int(v) for v in up_rows],
            "down_cols": [int(v) for v in down_cols],
            "epoch_created": int(epoch_created),
            "status": "probationary",
        }

    def _remove_channel_edges(self, channel: Dict[str, Any]) -> None:
        with torch.no_grad():
            up_key = str(channel["up_key"])
            down_key = str(channel["down_key"])
            h = int(channel["hidden_idx"])
            up_rows = [int(v) for v in channel.get("up_rows", [])]
            down_cols = [int(v) for v in channel.get("down_cols", [])]

            if up_rows:
                dev = self.weights[up_key].device
                rows_t = torch.tensor(up_rows, device=dev, dtype=torch.long)
                cols_t = torch.full((len(up_rows),), h, device=dev, dtype=torch.long)
                getattr(self, self.masks[up_key])[rows_t, cols_t] = 0.0
                self.weights[up_key].data[rows_t, cols_t] = 0.0
                self._get_usage(up_key)[rows_t, cols_t] = 0.0
                self._get_usage_peak(up_key)[rows_t, cols_t] = 0.0
                self._get_age(up_key)[rows_t, cols_t] = 0.0
                self._get_replay_tag(up_key)[rows_t, cols_t] = 0.0
                self._get_creation_score(up_key)[rows_t, cols_t] = 0.0
                self._get_creation_direction(up_key)[rows_t, cols_t] = 0.0

            if down_cols:
                dev = self.weights[down_key].device
                rows_t = torch.full((len(down_cols),), h, device=dev, dtype=torch.long)
                cols_t = torch.tensor(down_cols, device=dev, dtype=torch.long)
                getattr(self, self.masks[down_key])[rows_t, cols_t] = 0.0
                self.weights[down_key].data[rows_t, cols_t] = 0.0
                self._get_usage(down_key)[rows_t, cols_t] = 0.0
                self._get_usage_peak(down_key)[rows_t, cols_t] = 0.0
                self._get_age(down_key)[rows_t, cols_t] = 0.0
                self._get_replay_tag(down_key)[rows_t, cols_t] = 0.0
                self._get_creation_score(down_key)[rows_t, cols_t] = 0.0
                self._get_creation_direction(down_key)[rows_t, cols_t] = 0.0

    def _compute_channel_activity(self, channel: Dict[str, Any]) -> float:
        up_key = str(channel["up_key"])
        down_key = str(channel["down_key"])
        h = int(channel["hidden_idx"])
        up_rows = [int(v) for v in channel.get("up_rows", [])]
        down_cols = [int(v) for v in channel.get("down_cols", [])]
        if not up_rows or not down_cols:
            return 0.0

        up_usage = self._get_usage(up_key).detach().cpu().float()
        down_usage = self._get_usage(down_key).detach().cpu().float()
        up_vals = up_usage[torch.tensor(up_rows, dtype=torch.long), h]
        down_vals = down_usage[h, torch.tensor(down_cols, dtype=torch.long)]
        up_activity = float(up_vals.mean().item()) if int(up_vals.numel()) > 0 else 0.0
        down_activity = float(down_vals.mean().item()) if int(down_vals.numel()) > 0 else 0.0
        return float(math.sqrt(max(up_activity, 0.0) * max(down_activity, 0.0)))

    def _phase1_registry_entry(
        self,
        prefix: str,
        hidden_idx: int,
        create: bool = False,
    ) -> Optional[Dict[str, Any]]:
        prefix_key = str(prefix)
        hidden = int(hidden_idx)
        by_prefix = self._phase1_exploration_registry.get(prefix_key)
        if by_prefix is None:
            if not create:
                return None
            by_prefix = {}
            self._phase1_exploration_registry[prefix_key] = by_prefix
        entry = by_prefix.get(hidden)
        if entry is None and create:
            entry = {
                "n_attempts_total": 0,
                "n_attempts_current_level": 0,
                # "Promoted" is the canonical wording for phase1 probe channels.
                # Keep legacy n_tenured_* aliases for backward compatibility.
                "n_promoted_total": 0,
                "n_promoted_current_level": 0,
                "n_tenured_total": 0,
                "n_tenured_current_level": 0,
                "n_rejected_total": 0,
                "n_rejected_current_level": 0,
                "last_attempt_epoch": -1,
                "best_activity": 0.0,
            }
            by_prefix[hidden] = entry
        return entry

    def _phase1_registry_coverage(self) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, float], float]:
        eligible_ids_by_prefix = {
            str(key): {int(v) for v in values}
            for key, values in self._phase1_eligible_hidden_ids_by_prefix.items()
        }
        eligible_by_prefix = {
            str(key): int(len(values))
            for key, values in eligible_ids_by_prefix.items()
        }
        explored_by_prefix: Dict[str, int] = {}
        coverage_by_prefix: Dict[str, float] = {}
        total_eligible = 0
        total_explored = 0

        if eligible_by_prefix:
            for prefix, eligible_count in eligible_by_prefix.items():
                registry = self._phase1_exploration_registry.get(prefix, {})
                eligible_hidden_ids = eligible_ids_by_prefix.get(prefix, set())
                explored_count = 0
                if eligible_count > 0 and registry:
                    for hidden_idx, entry in registry.items():
                        if int(hidden_idx) not in eligible_hidden_ids:
                            continue
                        if int(entry.get("n_attempts_current_level", 0)) > 0:
                            explored_count += 1
                explored_count = min(int(explored_count), int(eligible_count))
                explored_by_prefix[prefix] = int(explored_count)
                coverage_by_prefix[prefix] = float(explored_count / max(1, eligible_count))
                total_eligible += int(eligible_count)
                total_explored += int(explored_count)
        else:
            # Fallback for epochs where no structural eligibility map is available yet.
            for prefix, registry in self._phase1_exploration_registry.items():
                explored_count = sum(
                    1
                    for entry in registry.values()
                    if int(entry.get("n_attempts_current_level", 0)) > 0
                )
                if explored_count <= 0:
                    continue
                explored_by_prefix[str(prefix)] = int(explored_count)
                eligible_by_prefix[str(prefix)] = int(explored_count)
                coverage_by_prefix[str(prefix)] = 1.0
                total_eligible += int(explored_count)
                total_explored += int(explored_count)

        coverage_global = float(total_explored / max(1, total_eligible)) if total_eligible > 0 else 0.0
        return eligible_by_prefix, explored_by_prefix, coverage_by_prefix, coverage_global

    def _empty_channel_probe_stats(self) -> Dict[str, Any]:
        (
            eligible_hidden_count_by_prefix,
            explored_hidden_count_by_prefix,
            coverage_by_prefix,
            coverage_global,
        ) = self._phase1_registry_coverage()
        return {
            "policy": self.creation_policy,
            "probationary_channels_epoch": 0.0,
            "probationary_channels_by_prefix": {},
            "ready_channels_epoch": 0.0,
            "evaluated_channels_epoch": 0.0,
            # "Promoted" is the canonical wording for probe channels that graduate
            # from probation. Keep legacy tenured_* aliases for compatibility.
            "promoted_probe_channels_epoch": 0.0,
            "tenured_channels_epoch": 0.0,
            "rejected_channels_epoch": 0.0,
            "probe_promotion_rate_epoch": 0.0,
            "tenure_rate_epoch": 0.0,
            "max_probation_age_epoch": 0.0,
            "oldest_probation_age_epoch": 0.0,
            "channel_activity_mean_promoted_by_prefix": {},
            "channel_activity_mean_tenured_by_prefix": {},
            "channel_activity_mean_rejected_by_prefix": {},
            "channel_activity_p95_by_prefix": {},
            "selected_channel_signal_mean_by_prefix": {},
            "selected_channel_struct_fallback_count_by_prefix": {},
            "eligible_signal_present_by_prefix": {},
            "pathway_dropout_active_epoch": 0.0,
            "pathway_dropout_rate_effective": 0.0,
            "pathway_dropout_dropped_channels_by_prefix": {},
            "pathway_dropout_candidate_channels_by_prefix": {},
            "pathway_dropout_protected_channels_by_prefix": {},
            "local_reinforce_queue_size_epoch": float(len(self._local_reinforce_queue)),
            "local_reinforce_channels_epoch": 0.0,
            "local_reinforce_edges_epoch": 0.0,
            "local_reinforce_edges_up_by_prefix": {},
            "local_reinforce_edges_down_by_prefix": {},
            "local_reinforce_budget_used_epoch": 0.0,
            "max_probationary_channels_per_prefix": float(self.max_probationary_channels_per_prefix),
            "max_probationary_channels_global": float(self.max_probationary_channels_global),
            "channel_probe_period": float(self.channel_probe_period),
            "max_new_probationary_channels_per_epoch": float(self.max_new_probationary_channels_per_epoch),
            "new_hidden_attempts_epoch": float(self._phase1_new_hidden_attempts_epoch),
            "retested_hidden_attempts_epoch": float(self._phase1_retested_hidden_attempts_epoch),
            "eligible_hidden_count_by_prefix": {
                str(key): int(value) for key, value in eligible_hidden_count_by_prefix.items()
            },
            "explored_hidden_count_by_prefix": {
                str(key): int(value) for key, value in explored_hidden_count_by_prefix.items()
            },
            "coverage_by_prefix": {
                str(key): float(value) for key, value in coverage_by_prefix.items()
            },
            "coverage_global": float(coverage_global),
            "evaluated_channels_total_current_level": float(self._phase1_evaluated_channels_total_current_level),
            "promoted_probe_channels_total_current_level": float(self._phase1_tenured_channels_total_current_level),
            "tenured_channels_total_current_level": float(self._phase1_tenured_channels_total_current_level),
            "rejected_channels_total_current_level": float(self._phase1_rejected_channels_total_current_level),
            "created_channels_total_current_level": float(self._created_channels_total_current_level),
            "channel_probe_threshold": (
                float(self._channel_probe_threshold_ema)
                if self._channel_probe_threshold_ema is not None
                else 0.0
            ),
            "created_channels_total": float(self.created_channels_total),
            "promoted_probe_channels_total": float(self.tenured_channels_total),
            "tenured_channels_total": float(self.tenured_channels_total),
            "rejected_channels_total": float(self.rejected_channels_total),
            "local_reinforce_channels_total": float(self.local_reinforce_channels_total),
            "local_reinforce_edges_total": float(self.local_reinforce_edges_total),
            "probe_promotion_rate_total": (
                float(self.tenured_channels_total / max(1, self.tenured_channels_total + self.rejected_channels_total))
                if (self.tenured_channels_total + self.rejected_channels_total) > 0
                else 0.0
            ),
            "tenure_rate_total": (
                float(self.tenured_channels_total / max(1, self.tenured_channels_total + self.rejected_channels_total))
                if (self.tenured_channels_total + self.rejected_channels_total) > 0
                else 0.0
            ),
            "avg_channel_activity_promoted": (
                float(self._channel_activity_tenured_sum / max(1, self._channel_activity_tenured_count))
                if self._channel_activity_tenured_count > 0
                else 0.0
            ),
            "avg_channel_activity_tenured": (
                float(self._channel_activity_tenured_sum / max(1, self._channel_activity_tenured_count))
                if self._channel_activity_tenured_count > 0
                else 0.0
            ),
            "avg_channel_activity_rejected": (
                float(self._channel_activity_rejected_sum / max(1, self._channel_activity_rejected_count))
                if self._channel_activity_rejected_count > 0
                else 0.0
            ),
        }

    def update_probationary_channels(self) -> Dict[str, Any]:
        self._ensure_epoch_runtime_state()
        if not self._probationary_channels:
            self.last_channel_probe_stats = self._empty_channel_probe_stats()
            return dict(self.last_channel_probe_stats)

        ready_ids: List[int] = []
        activities: List[float] = []
        activity_by_channel: Dict[int, float] = {}
        activity_by_prefix: Dict[str, List[float]] = {}
        max_probation_age = 0.0
        for channel_id, channel in self._probationary_channels.items():
            up_key = str(channel["up_key"])
            down_key = str(channel["down_key"])
            h = int(channel["hidden_idx"])
            up_rows = [int(v) for v in channel.get("up_rows", [])]
            down_cols = [int(v) for v in channel.get("down_cols", [])]
            if not up_rows or not down_cols:
                ready = True
                min_age = float(self.channel_probe_period)
            else:
                up_age = self._get_age(up_key).detach().cpu().float()[
                    torch.tensor(up_rows, dtype=torch.long), h
                ]
                down_age = self._get_age(down_key).detach().cpu().float()[
                    h, torch.tensor(down_cols, dtype=torch.long)
                ]
                min_age = min(
                    float(up_age.min().item()) if int(up_age.numel()) > 0 else 0.0,
                    float(down_age.min().item()) if int(down_age.numel()) > 0 else 0.0,
                )
                ready = min_age >= float(self.channel_probe_period)
            max_probation_age = max(max_probation_age, float(min_age))
            if not ready:
                continue
            ready_ids.append(int(channel_id))
            activity = self._compute_channel_activity(channel)
            activity_by_channel[int(channel_id)] = float(activity)
            activities.append(float(activity))
            prefix = str(channel["prefix"])
            activity_by_prefix.setdefault(prefix, []).append(float(activity))

        tenured_ids: List[int] = []
        rejected_ids: List[int] = []
        tenured_by_prefix: Dict[str, List[float]] = {}
        rejected_by_prefix: Dict[str, List[float]] = {}
        threshold = float(self._channel_probe_threshold_ema) if self._channel_probe_threshold_ema is not None else 0.0
        if activities:
            median_activity = float(torch.tensor(activities, dtype=torch.float32).median().item())
            if self._channel_probe_threshold_ema is None:
                threshold = median_activity
            else:
                alpha = float(self.channel_probe_threshold_ema_alpha)
                threshold = alpha * median_activity + (1.0 - alpha) * float(self._channel_probe_threshold_ema)
            self._channel_probe_threshold_ema = float(threshold)

            for channel_id in ready_ids:
                channel = self._probationary_channels[channel_id]
                prefix = str(channel["prefix"])
                activity = float(activity_by_channel[channel_id])
                if self.phase1_registry_enabled:
                    registry_entry = self._phase1_registry_entry(prefix, int(channel["hidden_idx"]), create=True)
                    if registry_entry is not None:
                        registry_entry["best_activity"] = float(
                            max(float(registry_entry.get("best_activity", 0.0)), activity)
                        )
                if activity >= threshold:
                    channel["status"] = "tenured"
                    tenured_ids.append(channel_id)
                    if self.tenured_channel_grace_period > 0:
                        self._tenured_grace_channels[int(channel_id)] = dict(channel)
                    if self.local_reinforce_enabled and self.local_reinforce_epochs > 0:
                        self._local_reinforce_queue[int(channel_id)] = {
                            **dict(channel),
                            "remaining_epochs": int(self.local_reinforce_epochs),
                            "reinforce_edges_added_up": 0,
                            "reinforce_edges_added_down": 0,
                            "reinforce_total_edges_added": 0,
                        }
                    tenured_by_prefix.setdefault(prefix, []).append(activity)
                    self.tenured_channels_total += 1
                    self._channel_activity_tenured_sum += activity
                    self._channel_activity_tenured_count += 1
                    if self.phase1_registry_enabled:
                        registry_entry = self._phase1_registry_entry(prefix, int(channel["hidden_idx"]), create=True)
                        if registry_entry is not None:
                            registry_entry["n_promoted_total"] = int(
                                registry_entry.get("n_promoted_total", 0)
                            ) + 1
                            registry_entry["n_promoted_current_level"] = int(
                                registry_entry.get("n_promoted_current_level", 0)
                            ) + 1
                            registry_entry["n_tenured_total"] = int(
                                registry_entry.get("n_tenured_total", 0)
                            ) + 1
                            registry_entry["n_tenured_current_level"] = int(
                                registry_entry.get("n_tenured_current_level", 0)
                            ) + 1
                else:
                    self._remove_channel_edges(channel)
                    channel["status"] = "rejected"
                    rejected_ids.append(channel_id)
                    rejected_by_prefix.setdefault(prefix, []).append(activity)
                    self.rejected_channels_total += 1
                    self._channel_activity_rejected_sum += activity
                    self._channel_activity_rejected_count += 1
                    if self.phase1_registry_enabled:
                        registry_entry = self._phase1_registry_entry(prefix, int(channel["hidden_idx"]), create=True)
                        if registry_entry is not None:
                            registry_entry["n_rejected_total"] = int(
                                registry_entry.get("n_rejected_total", 0)
                            ) + 1
                            registry_entry["n_rejected_current_level"] = int(
                                registry_entry.get("n_rejected_current_level", 0)
                            ) + 1

            for channel_id in tenured_ids + rejected_ids:
                self._probationary_channels.pop(channel_id, None)

        probationary_counts = self._probationary_counts_by_prefix()
        evaluated_total = len(ready_ids)
        self._phase1_evaluated_channels_total_current_level += int(evaluated_total)
        self._phase1_tenured_channels_total_current_level += int(len(tenured_ids))
        self._phase1_rejected_channels_total_current_level += int(len(rejected_ids))
        (
            eligible_hidden_count_by_prefix,
            explored_hidden_count_by_prefix,
            coverage_by_prefix,
            coverage_global,
        ) = self._phase1_registry_coverage()
        self.last_channel_probe_stats = {
            "policy": self.creation_policy,
            "probationary_channels_epoch": float(len(self._probationary_channels)),
            "probationary_channels_by_prefix": {
                str(key): int(value) for key, value in probationary_counts.items()
            },
            "ready_channels_epoch": float(len(ready_ids)),
            "evaluated_channels_epoch": float(evaluated_total),
            "promoted_probe_channels_epoch": float(len(tenured_ids)),
            "tenured_channels_epoch": float(len(tenured_ids)),
            "rejected_channels_epoch": float(len(rejected_ids)),
            "probe_promotion_rate_epoch": (
                float(len(tenured_ids) / max(1, evaluated_total)) if evaluated_total > 0 else 0.0
            ),
            "tenure_rate_epoch": (
                float(len(tenured_ids) / max(1, evaluated_total)) if evaluated_total > 0 else 0.0
            ),
            "max_probation_age_epoch": float(max_probation_age),
            "oldest_probation_age_epoch": float(max_probation_age),
            "channel_activity_mean_promoted_by_prefix": {
                str(key): float(sum(vals) / max(1, len(vals))) for key, vals in tenured_by_prefix.items()
            },
            "channel_activity_mean_tenured_by_prefix": {
                str(key): float(sum(vals) / max(1, len(vals))) for key, vals in tenured_by_prefix.items()
            },
            "channel_activity_mean_rejected_by_prefix": {
                str(key): float(sum(vals) / max(1, len(vals))) for key, vals in rejected_by_prefix.items()
            },
            "channel_activity_p95_by_prefix": {
                str(key): float(torch.quantile(torch.tensor(vals, dtype=torch.float32), 0.95).item())
                for key, vals in activity_by_prefix.items()
                if vals
            },
            "selected_channel_signal_mean_by_prefix": {},
            "selected_channel_struct_fallback_count_by_prefix": {},
            "eligible_signal_present_by_prefix": {},
            "pathway_dropout_active_epoch": float(self._epoch_pathway_dropout_active),
            "pathway_dropout_rate_effective": float(self._epoch_pathway_dropout_rate_effective),
            "pathway_dropout_dropped_channels_by_prefix": {
                str(k): int(v) for k, v in self._epoch_pathway_dropout_dropped_channels_by_prefix.items()
            },
            "pathway_dropout_candidate_channels_by_prefix": {
                str(k): int(v) for k, v in self._epoch_pathway_dropout_candidate_channels_by_prefix.items()
            },
            "pathway_dropout_protected_channels_by_prefix": {
                str(k): int(v) for k, v in self._epoch_pathway_dropout_protected_channels_by_prefix.items()
            },
            "local_reinforce_queue_size_epoch": float(len(self._local_reinforce_queue)),
            "local_reinforce_channels_epoch": float(len(self._epoch_local_reinforce_channels)),
            "local_reinforce_edges_epoch": float(self._epoch_local_reinforce_edges),
            "local_reinforce_edges_up_by_prefix": {
                str(k): int(v) for k, v in self._epoch_local_reinforce_edges_up_by_prefix.items()
            },
            "local_reinforce_edges_down_by_prefix": {
                str(k): int(v) for k, v in self._epoch_local_reinforce_edges_down_by_prefix.items()
            },
            "local_reinforce_budget_used_epoch": float(self._epoch_local_reinforce_budget_used),
            "max_probationary_channels_per_prefix": float(self.max_probationary_channels_per_prefix),
            "max_probationary_channels_global": float(self.max_probationary_channels_global),
            "channel_probe_period": float(self.channel_probe_period),
            "max_new_probationary_channels_per_epoch": float(self.max_new_probationary_channels_per_epoch),
            "new_hidden_attempts_epoch": float(self._phase1_new_hidden_attempts_epoch),
            "retested_hidden_attempts_epoch": float(self._phase1_retested_hidden_attempts_epoch),
            "eligible_hidden_count_by_prefix": {
                str(key): int(value) for key, value in eligible_hidden_count_by_prefix.items()
            },
            "explored_hidden_count_by_prefix": {
                str(key): int(value) for key, value in explored_hidden_count_by_prefix.items()
            },
            "coverage_by_prefix": {
                str(key): float(value) for key, value in coverage_by_prefix.items()
            },
            "coverage_global": float(coverage_global),
            "evaluated_channels_total_current_level": float(self._phase1_evaluated_channels_total_current_level),
            "promoted_probe_channels_total_current_level": float(self._phase1_tenured_channels_total_current_level),
            "tenured_channels_total_current_level": float(self._phase1_tenured_channels_total_current_level),
            "rejected_channels_total_current_level": float(self._phase1_rejected_channels_total_current_level),
            "created_channels_total_current_level": float(self._created_channels_total_current_level),
            "channel_probe_threshold": float(threshold),
            "created_channels_total": float(self.created_channels_total),
            "promoted_probe_channels_total": float(self.tenured_channels_total),
            "tenured_channels_total": float(self.tenured_channels_total),
            "rejected_channels_total": float(self.rejected_channels_total),
            "local_reinforce_channels_total": float(self.local_reinforce_channels_total),
            "local_reinforce_edges_total": float(self.local_reinforce_edges_total),
            "probe_promotion_rate_total": (
                float(self.tenured_channels_total / max(1, self.tenured_channels_total + self.rejected_channels_total))
                if (self.tenured_channels_total + self.rejected_channels_total) > 0
                else 0.0
            ),
            "tenure_rate_total": (
                float(self.tenured_channels_total / max(1, self.tenured_channels_total + self.rejected_channels_total))
                if (self.tenured_channels_total + self.rejected_channels_total) > 0
                else 0.0
            ),
            "avg_channel_activity_promoted": (
                float(self._channel_activity_tenured_sum / max(1, self._channel_activity_tenured_count))
                if self._channel_activity_tenured_count > 0
                else 0.0
            ),
            "avg_channel_activity_tenured": (
                float(self._channel_activity_tenured_sum / max(1, self._channel_activity_tenured_count))
                if self._channel_activity_tenured_count > 0
                else 0.0
            ),
            "avg_channel_activity_rejected": (
                float(self._channel_activity_rejected_sum / max(1, self._channel_activity_rejected_count))
                if self._channel_activity_rejected_count > 0
                else 0.0
            ),
        }
        return dict(self.last_channel_probe_stats)

    def gradient_guided_synaptogenesis(
        self,
        max_new_synapses: int = 10,
        score_temperature: float = 1.0,
        min_score: float = 0.0,
        per_connection_quota: Optional[int] = None,
        per_layer_quota: Optional[int] = None,
        init_scale: float = 0.01,
        epoch_created: int = -1,
    ) -> int:
        """
        Create synapses from creation score buffers.
        Score buffers are updated from weight gradients through hooks.
        """
        all_keys = list(self.weights.keys())

        def _blank_int_map() -> Dict[str, int]:
            return {k: 0 for k in all_keys}

        def _blank_float_map() -> Dict[str, float]:
            return {k: 0.0 for k in all_keys}

        def _set_creation_stats(
            budget_requested: int,
            budget_used: int = 0,
            candidate_total: int = 0,
            ordered_keys: Optional[List[str]] = None,
            candidate_count_by_key: Optional[Dict[str, int]] = None,
            eligible_count_before_quality_by_key: Optional[Dict[str, int]] = None,
            eligible_count_by_key: Optional[Dict[str, int]] = None,
            score_mean_by_key: Optional[Dict[str, float]] = None,
            score_p95_by_key: Optional[Dict[str, float]] = None,
            score_max_by_key: Optional[Dict[str, float]] = None,
            active_ref_by_key: Optional[Dict[str, float]] = None,
            threshold_by_key: Optional[Dict[str, float]] = None,
            quality_threshold_by_key: Optional[Dict[str, float]] = None,
            gap_mass_by_key: Optional[Dict[str, float]] = None,
            relative_gap_mass_by_key: Optional[Dict[str, float]] = None,
            priority_mass_by_key: Optional[Dict[str, float]] = None,
            frustration_mass_by_key: Optional[Dict[str, float]] = None,
            relative_frustration_mass_by_key: Optional[Dict[str, float]] = None,
            coherence_mean_by_key: Optional[Dict[str, float]] = None,
            coherence_p50_by_key: Optional[Dict[str, float]] = None,
            coherence_p75_by_key: Optional[Dict[str, float]] = None,
            coherence_p90_by_key: Optional[Dict[str, float]] = None,
            interference_mean_by_key: Optional[Dict[str, float]] = None,
            interference_p95_by_key: Optional[Dict[str, float]] = None,
            interference_max_by_key: Optional[Dict[str, float]] = None,
            support_mean_by_key: Optional[Dict[str, float]] = None,
            support_p75_by_key: Optional[Dict[str, float]] = None,
            support_p90_by_key: Optional[Dict[str, float]] = None,
            support_scale_by_key: Optional[Dict[str, float]] = None,
            scarcity_mean_by_key: Optional[Dict[str, float]] = None,
            scarcity_p95_by_key: Optional[Dict[str, float]] = None,
            scarcity_max_by_key: Optional[Dict[str, float]] = None,
            need_score_mean_by_key: Optional[Dict[str, float]] = None,
            need_score_p95_by_key: Optional[Dict[str, float]] = None,
            need_score_max_by_key: Optional[Dict[str, float]] = None,
            need_interference_mean_by_key: Optional[Dict[str, float]] = None,
            need_interference_p95_by_key: Optional[Dict[str, float]] = None,
            need_by_key: Optional[Dict[str, float]] = None,
            need_ema_by_key: Optional[Dict[str, float]] = None,
            priority_positive_count_by_key: Optional[Dict[str, int]] = None,
            selected_priority_mean_by_key: Optional[Dict[str, float]] = None,
            selected_priority_max_by_key: Optional[Dict[str, float]] = None,
            created_by_key: Optional[Dict[str, int]] = None,
            created_floor_by_key: Optional[Dict[str, int]] = None,
            created_global_by_key: Optional[Dict[str, int]] = None,
            group_budget_by_key: Optional[Dict[str, int]] = None,
            group_budget_by_role_group: Optional[Dict[str, int]] = None,
            candidate_count_by_role_group: Optional[Dict[str, int]] = None,
            eligible_count_by_role_group: Optional[Dict[str, int]] = None,
            gap_mass_by_role_group: Optional[Dict[str, float]] = None,
            relative_gap_mass_by_role_group: Optional[Dict[str, float]] = None,
            priority_mass_by_role_group: Optional[Dict[str, float]] = None,
            frustration_mass_by_role_group: Optional[Dict[str, float]] = None,
            created_by_role_group: Optional[Dict[str, int]] = None,
            group_share_target_by_role_group: Optional[Dict[str, float]] = None,
            group_share_ema_by_role_group: Optional[Dict[str, float]] = None,
            group_share_applied_by_role_group: Optional[Dict[str, float]] = None,
            role_need_by_role_group: Optional[Dict[str, float]] = None,
            role_need_ema_by_role_group: Optional[Dict[str, float]] = None,
            created_channels_epoch: int = 0,
            created_channel_edges_epoch: int = 0,
            created_channels_by_prefix: Optional[Dict[str, int]] = None,
            selected_hidden_mean_degree_by_prefix: Optional[Dict[str, float]] = None,
            selected_hidden_p75_degree_by_prefix: Optional[Dict[str, float]] = None,
            selected_hidden_mean_fire_rate_by_prefix: Optional[Dict[str, float]] = None,
            selected_hidden_mean_usage_by_prefix: Optional[Dict[str, float]] = None,
            created_channels_total: Optional[int] = None,
            created_channel_edges_total: Optional[int] = None,
            created_channels_total_by_prefix: Optional[Dict[str, int]] = None,
            skipped_by_headroom_by_prefix: Optional[Dict[str, int]] = None,
            skipped_by_incomplete_bundle_by_prefix: Optional[Dict[str, int]] = None,
            skipped_by_density_cap_by_prefix: Optional[Dict[str, int]] = None,
            block_density_by_prefix: Optional[Dict[str, float]] = None,
            mean_edges_per_channel: Optional[float] = None,
            budget_scale: float = 1.0,
            budget_effective: Optional[int] = None,
            floor_per_connection: Optional[int] = None,
            eligible_hidden_count_by_prefix: Optional[Dict[str, int]] = None,
            explored_hidden_count_by_prefix: Optional[Dict[str, int]] = None,
            coverage_by_prefix: Optional[Dict[str, float]] = None,
            coverage_global: Optional[float] = None,
            new_hidden_attempts_epoch: Optional[int] = None,
            retested_hidden_attempts_epoch: Optional[int] = None,
        ) -> None:
            counts = _blank_int_map()
            before_quality_counts = _blank_int_map()
            eligible_counts = _blank_int_map()
            means = _blank_float_map()
            p95s = _blank_float_map()
            maxs = _blank_float_map()
            refs = _blank_float_map()
            thresholds = _blank_float_map()
            quality_thresholds = _blank_float_map()
            gap_masses = _blank_float_map()
            relative_gap_masses = _blank_float_map()
            masses = _blank_float_map()
            frustration_masses = _blank_float_map()
            relative_frustration_masses = _blank_float_map()
            coherence_means = _blank_float_map()
            coherence_p50s = _blank_float_map()
            coherence_p75s = _blank_float_map()
            coherence_p90s = _blank_float_map()
            interference_means = _blank_float_map()
            interference_p95s = _blank_float_map()
            interference_maxs = _blank_float_map()
            support_means = _blank_float_map()
            support_p75s = _blank_float_map()
            support_p90s = _blank_float_map()
            support_scales = _blank_float_map()
            scarcity_means = _blank_float_map()
            scarcity_p95s = _blank_float_map()
            scarcity_maxs = _blank_float_map()
            need_score_means = _blank_float_map()
            need_score_p95s = _blank_float_map()
            need_score_maxs = _blank_float_map()
            need_interference_means = _blank_float_map()
            need_interference_p95s = _blank_float_map()
            needs = _blank_float_map()
            need_emas = _blank_float_map()
            selected_means = _blank_float_map()
            selected_maxs = _blank_float_map()
            positive_counts = _blank_int_map()
            created = _blank_int_map()
            created_floor = _blank_int_map()
            created_global = _blank_int_map()
            group_budget = _blank_int_map()
            role_candidate_counts = {"enc": 0, "dec": 0, "other": 0}
            role_eligible_counts = {"enc": 0, "dec": 0, "other": 0}
            role_gap_masses = {"enc": 0.0, "dec": 0.0, "other": 0.0}
            role_relative_gap_masses = {"enc": 0.0, "dec": 0.0, "other": 0.0}
            role_priority_masses = {"enc": 0.0, "dec": 0.0, "other": 0.0}
            role_frustration_masses = {"enc": 0.0, "dec": 0.0, "other": 0.0}
            role_created = {"enc": 0, "dec": 0, "other": 0}
            role_group_budget = {"enc": 0, "dec": 0, "other": 0}
            role_group_share_target = {"enc": 0.0, "dec": 0.0, "other": 0.0}
            role_group_share_ema = {"enc": 0.0, "dec": 0.0, "other": 0.0}
            role_group_share_applied = {"enc": 0.0, "dec": 0.0, "other": 0.0}
            role_need = {"enc": 0.0, "dec": 0.0, "other": 0.0}
            role_need_ema = {"enc": 0.0, "dec": 0.0, "other": 0.0}
            skipped_headroom = {}
            skipped_incomplete = {}
            skipped_density = {}
            block_density = {}

            for mapping, target in (
                (candidate_count_by_key or {}, counts),
                (eligible_count_before_quality_by_key or {}, before_quality_counts),
                (eligible_count_by_key or {}, eligible_counts),
                (priority_positive_count_by_key or {}, positive_counts),
                (created_by_key or {}, created),
                (created_floor_by_key or {}, created_floor),
                (created_global_by_key or {}, created_global),
                (group_budget_by_key or {}, group_budget),
            ):
                for key, value in mapping.items():
                    target[key] = int(value)
            for mapping, target in (
                (score_mean_by_key or {}, means),
                (score_p95_by_key or {}, p95s),
                (score_max_by_key or {}, maxs),
                (active_ref_by_key or {}, refs),
                (threshold_by_key or {}, thresholds),
                (quality_threshold_by_key or {}, quality_thresholds),
                (gap_mass_by_key or {}, gap_masses),
                (relative_gap_mass_by_key or {}, relative_gap_masses),
                (priority_mass_by_key or {}, masses),
                (frustration_mass_by_key or {}, frustration_masses),
                (relative_frustration_mass_by_key or {}, relative_frustration_masses),
                (coherence_mean_by_key or {}, coherence_means),
                (coherence_p50_by_key or {}, coherence_p50s),
                (coherence_p75_by_key or {}, coherence_p75s),
                (coherence_p90_by_key or {}, coherence_p90s),
                (interference_mean_by_key or {}, interference_means),
                (interference_p95_by_key or {}, interference_p95s),
                (interference_max_by_key or {}, interference_maxs),
                (support_mean_by_key or {}, support_means),
                (support_p75_by_key or {}, support_p75s),
                (support_p90_by_key or {}, support_p90s),
                (support_scale_by_key or {}, support_scales),
                (scarcity_mean_by_key or {}, scarcity_means),
                (scarcity_p95_by_key or {}, scarcity_p95s),
                (scarcity_max_by_key or {}, scarcity_maxs),
                (need_score_mean_by_key or {}, need_score_means),
                (need_score_p95_by_key or {}, need_score_p95s),
                (need_score_max_by_key or {}, need_score_maxs),
                (need_interference_mean_by_key or {}, need_interference_means),
                (need_interference_p95_by_key or {}, need_interference_p95s),
                (need_by_key or {}, needs),
                (need_ema_by_key or {}, need_emas),
                (selected_priority_mean_by_key or {}, selected_means),
                (selected_priority_max_by_key or {}, selected_maxs),
            ):
                for key, value in mapping.items():
                    target[key] = float(value)
            for mapping, target in (
                (candidate_count_by_role_group or {}, role_candidate_counts),
                (eligible_count_by_role_group or {}, role_eligible_counts),
                (created_by_role_group or {}, role_created),
                (group_budget_by_role_group or {}, role_group_budget),
            ):
                for key, value in mapping.items():
                    target[str(key)] = int(value)
            for mapping, target in (
                (gap_mass_by_role_group or {}, role_gap_masses),
                (relative_gap_mass_by_role_group or {}, role_relative_gap_masses),
                (priority_mass_by_role_group or {}, role_priority_masses),
                (frustration_mass_by_role_group or {}, role_frustration_masses),
                (group_share_target_by_role_group or {}, role_group_share_target),
                (group_share_ema_by_role_group or {}, role_group_share_ema),
                (group_share_applied_by_role_group or {}, role_group_share_applied),
                (role_need_by_role_group or {}, role_need),
                (role_need_ema_by_role_group or {}, role_need_ema),
            ):
                for key, value in mapping.items():
                    target[str(key)] = float(value)
            for mapping, target in (
                (skipped_by_headroom_by_prefix or {}, skipped_headroom),
                (skipped_by_incomplete_bundle_by_prefix or {}, skipped_incomplete),
                (skipped_by_density_cap_by_prefix or {}, skipped_density),
                (block_density_by_prefix or {}, block_density),
            ):
                for key, value in mapping.items():
                    target[str(key)] = float(value) if target is block_density else int(value)

            (
                reg_eligible_hidden_count_by_prefix,
                reg_explored_hidden_count_by_prefix,
                reg_coverage_by_prefix,
                reg_coverage_global,
            ) = self._phase1_registry_coverage()
            used_eligible_hidden_count_by_prefix = (
                dict(eligible_hidden_count_by_prefix)
                if eligible_hidden_count_by_prefix is not None
                else dict(reg_eligible_hidden_count_by_prefix)
            )
            used_explored_hidden_count_by_prefix = (
                dict(explored_hidden_count_by_prefix)
                if explored_hidden_count_by_prefix is not None
                else dict(reg_explored_hidden_count_by_prefix)
            )
            used_coverage_by_prefix = (
                dict(coverage_by_prefix)
                if coverage_by_prefix is not None
                else dict(reg_coverage_by_prefix)
            )
            used_coverage_global = (
                float(coverage_global) if coverage_global is not None else float(reg_coverage_global)
            )
            used_new_hidden_attempts_epoch = int(
                self._phase1_new_hidden_attempts_epoch
                if new_hidden_attempts_epoch is None
                else new_hidden_attempts_epoch
            )
            used_retested_hidden_attempts_epoch = int(
                self._phase1_retested_hidden_attempts_epoch
                if retested_hidden_attempts_epoch is None
                else retested_hidden_attempts_epoch
            )

            self.last_creation_stats = {
                "policy": self.creation_policy,
                "budget_requested": float(budget_requested),
                "budget_used": float(budget_used),
                "candidate_total": float(candidate_total),
                "floor_per_connection": float(
                    self.creation_floor_per_connection if floor_per_connection is None else floor_per_connection
                ),
                "ordered_keys": list(ordered_keys or []),
                "candidate_count_by_key": counts,
                "eligible_count_before_quality_by_key": before_quality_counts,
                "eligible_count_by_key": eligible_counts,
                "score_mean_by_key": means,
                "score_p95_by_key": p95s,
                "score_max_by_key": maxs,
                "active_ref_by_key": refs,
                "threshold_by_key": thresholds,
                "quality_threshold_by_key": quality_thresholds,
                "gap_mass_by_key": gap_masses,
                "relative_gap_mass_by_key": relative_gap_masses,
                "priority_mass_by_key": masses,
                "frustration_mass_by_key": frustration_masses,
                "relative_frustration_mass_by_key": relative_frustration_masses,
                "coherence_mean_by_key": coherence_means,
                "coherence_p50_by_key": coherence_p50s,
                "coherence_p75_by_key": coherence_p75s,
                "coherence_p90_by_key": coherence_p90s,
                "interference_mean_by_key": interference_means,
                "interference_p95_by_key": interference_p95s,
                "interference_max_by_key": interference_maxs,
                "support_mean_by_key": support_means,
                "support_p75_by_key": support_p75s,
                "support_p90_by_key": support_p90s,
                "support_scale_by_key": support_scales,
                "scarcity_mean_by_key": scarcity_means,
                "scarcity_p95_by_key": scarcity_p95s,
                "scarcity_max_by_key": scarcity_maxs,
                "need_score_mean_by_key": need_score_means,
                "need_score_p95_by_key": need_score_p95s,
                "need_score_max_by_key": need_score_maxs,
                "need_interference_mean_by_key": need_interference_means,
                "need_interference_p95_by_key": need_interference_p95s,
                "need_by_key": needs,
                "need_ema_by_key": need_emas,
                "priority_positive_count_by_key": positive_counts,
                "selected_priority_mean_by_key": selected_means,
                "selected_priority_max_by_key": selected_maxs,
                "created_by_key": created,
                "created_floor_by_key": created_floor,
                "created_global_by_key": created_global,
                "group_budget_by_key": group_budget,
                "group_budget_by_role_group": role_group_budget,
                "candidate_count_by_role_group": role_candidate_counts,
                "eligible_count_by_role_group": role_eligible_counts,
                "gap_mass_by_role_group": role_gap_masses,
                "relative_gap_mass_by_role_group": role_relative_gap_masses,
                "priority_mass_by_role_group": role_priority_masses,
                "frustration_mass_by_role_group": role_frustration_masses,
                "created_by_role_group": role_created,
                "group_share_target_by_role_group": role_group_share_target,
                "group_share_ema_by_role_group": role_group_share_ema,
                "group_share_applied_by_role_group": role_group_share_applied,
                "role_need_by_role_group": role_need,
                "role_need_ema_by_role_group": role_need_ema,
                "created_channels_epoch": float(created_channels_epoch),
                "created_channel_edges_epoch": float(created_channel_edges_epoch),
                "created_channels_by_prefix": {
                    str(key): int(value) for key, value in (created_channels_by_prefix or {}).items()
                },
                "selected_hidden_mean_degree_by_prefix": {
                    str(key): float(value)
                    for key, value in (selected_hidden_mean_degree_by_prefix or {}).items()
                },
                "selected_hidden_p75_degree_by_prefix": {
                    str(key): float(value)
                    for key, value in (selected_hidden_p75_degree_by_prefix or {}).items()
                },
                "selected_hidden_mean_fire_rate_by_prefix": {
                    str(key): float(value)
                    for key, value in (selected_hidden_mean_fire_rate_by_prefix or {}).items()
                },
                "selected_hidden_mean_usage_by_prefix": {
                    str(key): float(value)
                    for key, value in (selected_hidden_mean_usage_by_prefix or {}).items()
                },
                "created_channels_total": float(
                    self.created_channels_total if created_channels_total is None else created_channels_total
                ),
                "created_channel_edges_total": float(
                    self.created_channel_edges_total
                    if created_channel_edges_total is None else created_channel_edges_total
                ),
                "created_channels_total_by_prefix": {
                    str(key): int(value)
                    for key, value in (
                        self.created_channels_total_by_prefix
                        if created_channels_total_by_prefix is None
                        else created_channels_total_by_prefix
                    ).items()
                },
                "mean_edges_per_channel": float(
                    (
                        self.created_channel_edges_total / max(1, self.created_channels_total)
                        if mean_edges_per_channel is None
                        else mean_edges_per_channel
                    )
                ),
                "avg_edges_per_channel_total": float(
                    self.created_channel_edges_total / max(1, self.created_channels_total)
                ) if self.created_channels_total > 0 else 0.0,
                "skipped_by_headroom_by_prefix": skipped_headroom,
                "skipped_by_incomplete_bundle_by_prefix": skipped_incomplete,
                "skipped_by_density_cap_by_prefix": skipped_density,
                "block_density_by_prefix": block_density,
                "final_block_density_by_prefix": dict(block_density),
                "budget_scale": float(budget_scale),
                "budget_effective": float(
                    budget_requested if budget_effective is None else int(budget_effective)
                ),
                "new_hidden_attempts_epoch": float(used_new_hidden_attempts_epoch),
                "retested_hidden_attempts_epoch": float(used_retested_hidden_attempts_epoch),
                "eligible_hidden_count_by_prefix": {
                    str(key): int(value) for key, value in used_eligible_hidden_count_by_prefix.items()
                },
                "explored_hidden_count_by_prefix": {
                    str(key): int(value) for key, value in used_explored_hidden_count_by_prefix.items()
                },
                "coverage_by_prefix": {
                    str(key): float(value) for key, value in used_coverage_by_prefix.items()
                },
                "coverage_global": float(used_coverage_global),
            }

        if max_new_synapses <= 0:
            _set_creation_stats(budget_requested=int(max_new_synapses))
            return 0

        remaining_budget = int(max_new_synapses)
        if math.isfinite(float(self.energy_budget)):
            current_energy = int(round(float(self.compute_energy())))
            headroom = max(0, int(math.floor(float(self.energy_budget) - float(current_energy))))
            remaining_budget = min(remaining_budget, headroom)
        if remaining_budget <= 0:
            _set_creation_stats(budget_requested=int(max_new_synapses))
            return 0

        mask_shadow: Dict[str, torch.Tensor] = {}
        age_shadow: Dict[str, torch.Tensor] = {}
        for key in self.weights.keys():
            mask_shadow[key] = (getattr(self, self.masks[key]).detach().cpu() > 0)
            age_shadow[key] = self._get_age(key).detach().cpu().float()

        candidate_meta: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        raw_scores_by_key: Dict[str, torch.Tensor] = {}
        gap_scores_by_key: Dict[str, torch.Tensor] = {}
        priority_scores_by_key: Dict[str, torch.Tensor] = {}
        candidate_count_by_key: Dict[str, int] = {}
        eligible_count_before_quality_by_key: Dict[str, int] = {}
        eligible_count_by_key: Dict[str, int] = {}
        score_mean_by_key: Dict[str, float] = {}
        score_p95_by_key: Dict[str, float] = {}
        score_max_by_key: Dict[str, float] = {}
        active_ref_by_key: Dict[str, float] = {}
        threshold_by_key: Dict[str, float] = {}
        quality_threshold_by_key: Dict[str, float] = {}
        gap_mass_by_key: Dict[str, float] = {}
        relative_gap_mass_by_key: Dict[str, float] = {}
        priority_mass_by_key: Dict[str, float] = {}
        frustration_mass_by_key: Dict[str, float] = {}
        relative_frustration_mass_by_key: Dict[str, float] = {}
        coherence_mean_by_key: Dict[str, float] = {}
        coherence_p50_by_key: Dict[str, float] = {}
        coherence_p75_by_key: Dict[str, float] = {}
        coherence_p90_by_key: Dict[str, float] = {}
        interference_mean_by_key: Dict[str, float] = {}
        interference_p95_by_key: Dict[str, float] = {}
        interference_max_by_key: Dict[str, float] = {}
        support_mean_by_key: Dict[str, float] = {}
        support_p75_by_key: Dict[str, float] = {}
        support_p90_by_key: Dict[str, float] = {}
        support_scale_by_key: Dict[str, float] = {}
        scarcity_mean_by_key: Dict[str, float] = {}
        scarcity_p95_by_key: Dict[str, float] = {}
        scarcity_max_by_key: Dict[str, float] = {}
        need_score_mean_by_key: Dict[str, float] = {}
        need_score_p95_by_key: Dict[str, float] = {}
        need_score_max_by_key: Dict[str, float] = {}
        need_interference_mean_by_key: Dict[str, float] = {}
        need_interference_p95_by_key: Dict[str, float] = {}
        need_by_key: Dict[str, float] = {}
        need_ema_by_key: Dict[str, float] = {}
        priority_positive_count_by_key: Dict[str, int] = {}
        fallback_mass_by_key: Dict[str, float] = {}
        eligible_idx_by_key: Dict[str, torch.Tensor] = {}
        eligible_raw_by_key: Dict[str, torch.Tensor] = {}
        eligible_dir_by_key: Dict[str, torch.Tensor] = {}
        eligible_interference_by_key: Dict[str, torch.Tensor] = {}
        candidate_count_by_role_group: Dict[str, int] = {"enc": 0, "dec": 0, "other": 0}
        eligible_count_by_role_group: Dict[str, int] = {"enc": 0, "dec": 0, "other": 0}
        gap_mass_by_role_group: Dict[str, float] = {"enc": 0.0, "dec": 0.0, "other": 0.0}
        relative_gap_mass_by_role_group: Dict[str, float] = {"enc": 0.0, "dec": 0.0, "other": 0.0}
        priority_mass_by_role_group: Dict[str, float] = {"enc": 0.0, "dec": 0.0, "other": 0.0}
        frustration_mass_by_role_group: Dict[str, float] = {"enc": 0.0, "dec": 0.0, "other": 0.0}
        group_share_target_by_role_group: Dict[str, float] = {"enc": 0.0, "dec": 0.0, "other": 0.0}
        group_share_ema_by_role_group: Dict[str, float] = {"enc": 0.0, "dec": 0.0, "other": 0.0}
        group_share_applied_by_role_group: Dict[str, float] = {"enc": 0.0, "dec": 0.0, "other": 0.0}
        role_need_by_role_group: Dict[str, float] = {"enc": 0.0, "dec": 0.0, "other": 0.0}
        role_need_ema_by_role_group: Dict[str, float] = {"enc": 0.0, "dec": 0.0, "other": 0.0}
        budget_scale = 1.0
        budget_effective = int(max_new_synapses)
        temp = max(1e-6, float(score_temperature))
        frustration_policy = self.creation_policy == "group_frustration_weighted"
        inverse_coverage_policy = self.creation_policy in {
            "need_quality_inverse_coverage",
            "paired_ffn_channel_random",
            "paired_ffn_channel_probe",
        }
        need_quality_policy = self.creation_policy in {
            "need_quality_filtered",
            "need_quality_inverse_coverage",
            "paired_ffn_channel_random",
            "paired_ffn_channel_probe",
        }

        for key in self.weights.keys():
            scores = self._get_creation_score(key).detach().cpu().float()
            dirs = self._get_creation_direction(key).detach().cpu().float()
            candidates = (~mask_shadow[key]) & (scores > float(min_score))
            if int(candidates.sum().item()) == 0:
                continue

            idx = candidates.nonzero(as_tuple=False)
            raw_scores = scores[candidates]
            cand_dirs = dirs[candidates]
            candidate_count_by_key[key] = int(idx.size(0))
            score_mean_by_key[key] = float(raw_scores.mean().item())
            score_p95_by_key[key] = float(torch.quantile(raw_scores, 0.95).item())
            score_max_by_key[key] = float(raw_scores.max().item())
            raw_scores_by_key[key] = raw_scores
            role_group = self._creation_role_group(key)
            candidate_count_by_role_group[role_group] = candidate_count_by_role_group.get(role_group, 0) + int(idx.size(0))

            if need_quality_policy:
                align_scores = torch.minimum(raw_scores, cand_dirs.abs())
                interference_scores = (raw_scores - align_scores).clamp_min(0.0)
                coherence_scores = align_scores / (raw_scores + 1e-12)
                active = mask_shadow[key]
                mature_active = active & (age_shadow[key] >= float(self.protection_period))
                support_mask = mature_active if int(mature_active.sum().item()) > 0 else active
                usage = self._get_usage(key).detach().cpu().float()
                support_by_col = (usage * support_mask.float()).sum(dim=0)
                support_positive = support_by_col[support_by_col > 0]
                support_scale = (
                    float(torch.quantile(support_positive, 0.75).item())
                    if int(support_positive.numel()) > 0 else 1.0
                )
                scarcity_by_col = (support_scale / (1e-8 + support_by_col)).clamp(0.25, 4.0)
                candidate_scarcity = scarcity_by_col[idx[:, 1]]
                need_scores = raw_scores * candidate_scarcity
                need_interference = interference_scores * candidate_scarcity
                coherence_mean_by_key[key] = float(coherence_scores.mean().item()) if int(coherence_scores.numel()) > 0 else 0.0
                coherence_p50_by_key[key] = (
                    float(torch.quantile(coherence_scores, 0.50).item())
                    if int(coherence_scores.numel()) > 0 else 0.0
                )
                coherence_p75_by_key[key] = (
                    float(torch.quantile(coherence_scores, 0.75).item())
                    if int(coherence_scores.numel()) > 0 else 0.0
                )
                coherence_p90_by_key[key] = (
                    float(torch.quantile(coherence_scores, 0.90).item())
                    if int(coherence_scores.numel()) > 0 else 0.0
                )
                interference_mean_by_key[key] = (
                    float(interference_scores.mean().item()) if int(interference_scores.numel()) > 0 else 0.0
                )
                interference_p95_by_key[key] = (
                    float(torch.quantile(interference_scores, 0.95).item())
                    if int(interference_scores.numel()) > 0 else 0.0
                )
                interference_max_by_key[key] = (
                    float(interference_scores.max().item()) if int(interference_scores.numel()) > 0 else 0.0
                )
                support_mean_by_key[key] = (
                    float(support_positive.mean().item()) if int(support_positive.numel()) > 0 else 0.0
                )
                support_p75_by_key[key] = (
                    float(torch.quantile(support_positive, 0.75).item())
                    if int(support_positive.numel()) > 0 else 0.0
                )
                support_p90_by_key[key] = (
                    float(torch.quantile(support_positive, 0.90).item())
                    if int(support_positive.numel()) > 0 else 0.0
                )
                support_scale_by_key[key] = float(support_scale)
                scarcity_mean_by_key[key] = (
                    float(scarcity_by_col.mean().item()) if int(scarcity_by_col.numel()) > 0 else 0.0
                )
                scarcity_p95_by_key[key] = (
                    float(torch.quantile(scarcity_by_col, 0.95).item())
                    if int(scarcity_by_col.numel()) > 0 else 0.0
                )
                scarcity_max_by_key[key] = (
                    float(scarcity_by_col.max().item()) if int(scarcity_by_col.numel()) > 0 else 0.0
                )
                need_score_mean_by_key[key] = (
                    float(need_scores.mean().item()) if int(need_scores.numel()) > 0 else 0.0
                )
                need_score_p95_by_key[key] = (
                    float(torch.quantile(need_scores, 0.95).item())
                    if int(need_scores.numel()) > 0 else 0.0
                )
                need_score_max_by_key[key] = (
                    float(need_scores.max().item()) if int(need_scores.numel()) > 0 else 0.0
                )
                need_interference_mean_by_key[key] = (
                    float(need_interference.mean().item()) if int(need_interference.numel()) > 0 else 0.0
                )
                need_interference_p95_by_key[key] = (
                    float(torch.quantile(need_interference, 0.95).item())
                    if int(need_interference.numel()) > 0 else 0.0
                )
                take_k = min(int(interference_scores.numel()), int(self.creation_priority_topk))
                if take_k > 0:
                    try:
                        top_idx = torch.topk(interference_scores, k=take_k, largest=True, sorted=False).indices
                    except RuntimeError:
                        top_idx = torch.argsort(interference_scores, descending=True)[:take_k]
                    frustration_mass = float(interference_scores[top_idx].sum().item())
                else:
                    frustration_mass = 0.0
                scale = max(1e-12, float(score_p95_by_key[key]))
                frustration_mass_by_key[key] = frustration_mass
                relative_frustration_mass_by_key[key] = float(frustration_mass / scale)
                gap_scores_by_key[key] = torch.zeros_like(raw_scores)
                priority_scores_by_key[key] = torch.zeros_like(raw_scores)
                gap_mass_by_key[key] = 0.0
                relative_gap_mass_by_key[key] = 0.0
                priority_mass_by_key[key] = 0.0
                threshold_by_key[key] = 0.0
                quality_threshold_by_key[key] = 0.0
                eligible_count_before_quality_by_key[key] = int(raw_scores.numel())
                active_ref_mask = mature_active if int(mature_active.sum().item()) > 0 else active
                active_ref_scores = scores[active_ref_mask]
                if inverse_coverage_policy and int(active_ref_scores.numel()) > 0:
                    active_ref_cols = active_ref_mask.nonzero(as_tuple=False)[:, 1]
                    active_ref_scores = active_ref_scores * scarcity_by_col[active_ref_cols]
                active_ref_med = float(active_ref_scores.median().item()) if int(active_ref_scores.numel()) > 0 else 0.0
                active_ref_by_key[key] = active_ref_med

                quality_scores = need_scores if inverse_coverage_policy else raw_scores
                selection_interference = need_interference if inverse_coverage_policy else interference_scores
                eligible = torch.zeros_like(raw_scores, dtype=torch.bool)
                if int(quality_scores.numel()) >= 8:
                    med_t = quality_scores.median()
                    mad_t = (quality_scores - med_t).abs().median()
                    quality_threshold = float((med_t + float(self.creation_quality_mad_factor) * mad_t).item())
                    quality_threshold_by_key[key] = quality_threshold
                    threshold_by_key[key] = quality_threshold
                    eligible = (quality_scores >= quality_threshold) & (
                        coherence_scores < float(self.creation_coherence_threshold)
                    )

                eligible_count = int(eligible.sum().item())
                eligible_count_by_key[key] = eligible_count
                priority_positive_count_by_key[key] = eligible_count
                eligible_count_by_role_group[role_group] = eligible_count_by_role_group.get(role_group, 0) + eligible_count

                need = 0.0
                if eligible_count > 0:
                    eligible_scores = quality_scores[eligible]
                    demand_p75 = (
                        float(torch.quantile(eligible_scores, 0.75).item())
                        if int(eligible_scores.numel()) >= 4 else 0.0
                    )
                    density = float(mask_shadow[key].float().mean().item())
                    headroom = max(0.0, 1.0 - density)
                    active_usage = self._get_usage(key).detach().cpu().float()[active]
                    if int(active_usage.numel()) > 0:
                        usage_p75 = torch.quantile(active_usage, 0.75)
                        active_usage_sat = float((active_usage > usage_p75).float().mean().item())
                    else:
                        active_usage_sat = 0.0
                    raw_pressure = max(0.0, demand_p75 - active_ref_med) / max(active_ref_med, 1e-8)
                    need = float(raw_pressure * headroom * (0.5 + 0.5 * active_usage_sat))
                    eligible_idx_by_key[key] = idx[eligible]
                    eligible_raw_by_key[key] = quality_scores[eligible]
                    eligible_dir_by_key[key] = cand_dirs[eligible]
                    eligible_interference_by_key[key] = selection_interference[eligible]

                prev_need = self._creation_need_ema.get(key)
                if prev_need is None:
                    need_ema = float(need)
                else:
                    alpha = float(self.creation_need_ema_alpha)
                    need_ema = alpha * float(need) + (1.0 - alpha) * float(prev_need)
                self._creation_need_ema[key] = float(need_ema)
                need_by_key[key] = float(need)
                need_ema_by_key[key] = float(need_ema)
                role_need_by_role_group[role_group] = role_need_by_role_group.get(role_group, 0.0) + float(need)
                role_need_ema_by_role_group[role_group] = (
                    role_need_ema_by_role_group.get(role_group, 0.0) + float(need_ema)
                )
                continue

            if frustration_policy:
                zero_gap = torch.zeros_like(raw_scores)
                align_scores = torch.minimum(raw_scores, cand_dirs.abs())
                interference_scores = (raw_scores - align_scores).clamp_min(0.0)
                coherence_scores = align_scores / (raw_scores + 1e-12)
                gap_scores_by_key[key] = zero_gap
                priority_scores_by_key[key] = zero_gap
                active_ref_by_key[key] = 0.0
                threshold_by_key[key] = 0.0
                positive_count = int(raw_scores.numel())
                priority_positive_count_by_key[key] = positive_count
                eligible_count_by_role_group[role_group] = eligible_count_by_role_group.get(role_group, 0) + positive_count
                take_k = min(int(interference_scores.numel()), int(self.creation_priority_topk))
                if take_k > 0:
                    try:
                        top_idx = torch.topk(interference_scores, k=take_k, largest=True, sorted=False).indices
                        raw_top_idx = torch.topk(raw_scores, k=take_k, largest=True, sorted=False).indices
                    except RuntimeError:
                        top_idx = torch.argsort(interference_scores, descending=True)[:take_k]
                        raw_top_idx = torch.argsort(raw_scores, descending=True)[:take_k]
                    frustration_mass = float(interference_scores[top_idx].sum().item())
                    raw_mass = float(raw_scores[raw_top_idx].sum().item())
                else:
                    frustration_mass = 0.0
                    raw_mass = 0.0
                scale = max(1e-12, float(score_p95_by_key[key]))
                relative_frustration_mass = float(frustration_mass / scale)
                frustration_mass_by_key[key] = frustration_mass
                relative_frustration_mass_by_key[key] = relative_frustration_mass
                fallback_mass_by_key[key] = raw_mass
                gap_mass_by_key[key] = 0.0
                relative_gap_mass_by_key[key] = 0.0
                priority_mass_by_key[key] = 0.0
                frustration_mass_by_role_group[role_group] = (
                    frustration_mass_by_role_group.get(role_group, 0.0) + frustration_mass
                )
                coherence_mean_by_key[key] = float(coherence_scores.mean().item()) if int(coherence_scores.numel()) > 0 else 0.0
                interference_mean_by_key[key] = (
                    float(interference_scores.mean().item()) if int(interference_scores.numel()) > 0 else 0.0
                )
                interference_p95_by_key[key] = (
                    float(torch.quantile(interference_scores, 0.95).item())
                    if int(interference_scores.numel()) > 0 else 0.0
                )
                interference_max_by_key[key] = (
                    float(interference_scores.max().item()) if int(interference_scores.numel()) > 0 else 0.0
                )
                cand_scores = raw_scores
                if abs(temp - 1.0) > 1e-6:
                    cand_scores = cand_scores.pow(1.0 / temp)
                candidate_meta[key] = (idx, cand_scores, cand_dirs)
                continue

            active = mask_shadow[key]
            active_scores = scores[active]
            ref_mask = active
            if self.creation_ref_mature_only:
                mature_active = active & (age_shadow[key] >= float(self.protection_period))
                if int(mature_active.sum().item()) > 0:
                    ref_mask = mature_active
            ref_scores = scores[ref_mask]
            if int(ref_scores.numel()) > 0:
                ref_val = float(torch.quantile(ref_scores, self.creation_active_ref_quantile).item())
            else:
                ref_val = 0.0
            threshold = ref_val + max(
                float(self.creation_gap_abs_margin),
                float(self.creation_gap_rel_margin) * max(0.0, ref_val),
            )
            gap_scores = (raw_scores - threshold).clamp_min(0.0)
            priority_scores = raw_scores * gap_scores
            gap_scores_by_key[key] = gap_scores
            priority_scores_by_key[key] = priority_scores
            active_ref_by_key[key] = float(ref_val)
            threshold_by_key[key] = float(threshold)
            positive_count = int((gap_scores > 0).sum().item())
            priority_positive_count_by_key[key] = positive_count
            eligible_count_by_role_group[role_group] = eligible_count_by_role_group.get(role_group, 0) + positive_count
            if int(priority_scores.numel()) > 0:
                take_k = min(int(priority_scores.numel()), int(self.creation_priority_topk))
                if take_k > 0:
                    try:
                        gap_top_idx = torch.topk(gap_scores, k=take_k, largest=True, sorted=False).indices
                        top_idx = torch.topk(priority_scores, k=take_k, largest=True, sorted=False).indices
                        gap_mass = float(gap_scores[gap_top_idx].sum().item())
                        priority_mass = float(priority_scores[top_idx].sum().item())
                        raw_mass = float(raw_scores[top_idx].sum().item())
                    except RuntimeError:
                        gap_sorted_idx = torch.argsort(gap_scores, descending=True)[:take_k]
                        sorted_idx = torch.argsort(priority_scores, descending=True)[:take_k]
                        gap_mass = float(gap_scores[gap_sorted_idx].sum().item())
                        priority_mass = float(priority_scores[sorted_idx].sum().item())
                        raw_mass = float(raw_scores[sorted_idx].sum().item())
                else:
                    gap_mass = 0.0
                    priority_mass = 0.0
                    raw_mass = 0.0
            else:
                gap_mass = 0.0
                priority_mass = 0.0
                raw_mass = 0.0
            gap_mass_by_key[key] = gap_mass
            scale = max(1e-12, float(score_p95_by_key[key]))
            relative_gap_mass = float(gap_mass / scale)
            relative_gap_mass_by_key[key] = relative_gap_mass
            priority_mass_by_key[key] = priority_mass
            fallback_mass_by_key[key] = raw_mass
            gap_mass_by_role_group[role_group] = gap_mass_by_role_group.get(role_group, 0.0) + gap_mass
            relative_gap_mass_by_role_group[role_group] = (
                relative_gap_mass_by_role_group.get(role_group, 0.0) + relative_gap_mass
            )
            priority_mass_by_role_group[role_group] = priority_mass_by_role_group.get(role_group, 0.0) + priority_mass
            cand_scores = raw_scores
            if abs(temp - 1.0) > 1e-6:
                cand_scores = cand_scores.pow(1.0 / temp)
            candidate_meta[key] = (idx, cand_scores, cand_dirs)

        tracked_creation_keys = list(candidate_count_by_key.keys())
        if not tracked_creation_keys:
            _set_creation_stats(
                budget_requested=int(max_new_synapses),
                candidate_count_by_key=candidate_count_by_key,
                eligible_count_before_quality_by_key=eligible_count_before_quality_by_key,
                eligible_count_by_key=eligible_count_by_key,
                score_mean_by_key=score_mean_by_key,
                score_p95_by_key=score_p95_by_key,
                score_max_by_key=score_max_by_key,
                active_ref_by_key=active_ref_by_key,
                threshold_by_key=threshold_by_key,
                quality_threshold_by_key=quality_threshold_by_key,
                gap_mass_by_key=gap_mass_by_key,
                relative_gap_mass_by_key=relative_gap_mass_by_key,
                priority_mass_by_key=priority_mass_by_key,
                frustration_mass_by_key=frustration_mass_by_key,
                relative_frustration_mass_by_key=relative_frustration_mass_by_key,
                coherence_mean_by_key=coherence_mean_by_key,
                coherence_p50_by_key=coherence_p50_by_key,
                coherence_p75_by_key=coherence_p75_by_key,
                coherence_p90_by_key=coherence_p90_by_key,
                interference_mean_by_key=interference_mean_by_key,
                interference_p95_by_key=interference_p95_by_key,
                interference_max_by_key=interference_max_by_key,
                need_by_key=need_by_key,
                need_ema_by_key=need_ema_by_key,
                priority_positive_count_by_key=priority_positive_count_by_key,
                candidate_count_by_role_group=candidate_count_by_role_group,
                eligible_count_by_role_group=eligible_count_by_role_group,
                gap_mass_by_role_group=gap_mass_by_role_group,
                relative_gap_mass_by_role_group=relative_gap_mass_by_role_group,
                priority_mass_by_role_group=priority_mass_by_role_group,
                frustration_mass_by_role_group=frustration_mass_by_role_group,
            )
            return 0

        n_conn = max(1, len(tracked_creation_keys))
        if per_connection_quota is None or per_connection_quota <= 0:
            per_connection_quota = max(1, int(math.ceil(max_new_synapses / n_conn)))

        dst_groups = sorted(set(int(self.connection_meta[k]["dst_group"]) for k in tracked_creation_keys))
        if per_layer_quota is None or per_layer_quota <= 0:
            per_layer_quota = max(1, int(math.ceil(max_new_synapses / max(1, len(dst_groups)))))

        layer_remaining: Dict[int, int] = {g: int(per_layer_quota) for g in dst_groups}

        pending_rows: Dict[str, List[int]] = {k: [] for k in tracked_creation_keys}
        pending_cols: Dict[str, List[int]] = {k: [] for k in tracked_creation_keys}
        pending_vals: Dict[str, List[float]] = {k: [] for k in tracked_creation_keys}
        created_by_key: Dict[str, int] = {k: 0 for k in tracked_creation_keys}
        created_floor_by_key: Dict[str, int] = {k: 0 for k in tracked_creation_keys}
        created_global_by_key: Dict[str, int] = {k: 0 for k in tracked_creation_keys}
        selected_priority_by_key: Dict[str, List[float]] = {k: [] for k in tracked_creation_keys}
        group_budget_by_key: Dict[str, int] = {k: 0 for k in tracked_creation_keys}
        group_budget_by_role_group: Dict[str, int] = {"enc": 0, "dec": 0, "other": 0}
        created_by_role_group: Dict[str, int] = {"enc": 0, "dec": 0, "other": 0}

        if self.creation_policy == "effect_gap_global":
            ordered_keys = sorted(
                candidate_meta.keys(),
                key=lambda k: (float(priority_mass_by_key.get(k, 0.0)), float(score_mean_by_key.get(k, 0.0))),
                reverse=True,
            )
        elif self.creation_policy == "group_frustration_weighted":
            ordered_keys = sorted(
                candidate_meta.keys(),
                key=lambda k: (float(frustration_mass_by_key.get(k, 0.0)), float(score_mean_by_key.get(k, 0.0))),
                reverse=True,
            )
        elif self.creation_policy in {
            "need_quality_filtered",
            "need_quality_inverse_coverage",
            "paired_ffn_channel_random",
            "paired_ffn_channel_probe",
        }:
            ordered_keys = sorted(
                tracked_creation_keys,
                key=lambda k: (
                    float(need_ema_by_key.get(k, 0.0)),
                    float(need_by_key.get(k, 0.0)),
                    float(
                        need_interference_p95_by_key.get(k, 0.0)
                        if self.creation_policy in {
                            "need_quality_inverse_coverage",
                            "paired_ffn_channel_random",
                            "paired_ffn_channel_probe",
                        }
                        else interference_p95_by_key.get(k, 0.0)
                    ),
                    float(score_mean_by_key.get(k, 0.0)),
                ),
                reverse=True,
            )
        elif self.creation_policy in {"group_balanced_effect_gap", "group_calibrated_weighted"}:
            ordered_keys = sorted(
                candidate_meta.keys(),
                key=lambda k: (float(gap_mass_by_key.get(k, 0.0)), float(score_mean_by_key.get(k, 0.0))),
                reverse=True,
            )
        else:
            ordered_keys = sorted(
                candidate_meta.keys(),
                key=lambda k: float(candidate_meta[k][1].mean().item()),
                reverse=True,
            )

        def _init_weight(sig: float) -> float:
            if abs(sig) < 1e-8:
                return float(torch.randn(1).item()) * float(init_scale)
            return max(-3.0, min(3.0, sig)) * float(init_scale)

        def _buffer_create(key: str, src_n: int, dst_n: int, val: float) -> bool:
            shadow = mask_shadow[key]
            if bool(shadow[src_n, dst_n].item()):
                return False
            shadow[src_n, dst_n] = True
            pending_rows[key].append(int(src_n))
            pending_cols[key].append(int(dst_n))
            pending_vals[key].append(float(val))
            return True

        def _record_created(key: str, *, is_floor: bool = False, is_group_budget: bool = False) -> None:
            created_by_key[key] += 1
            if is_floor:
                created_floor_by_key[key] += 1
            else:
                created_global_by_key[key] += 1
            if is_group_budget:
                group_budget_by_key[key] += 1
                role_group = self._creation_role_group(key)
                group_budget_by_role_group[role_group] = group_budget_by_role_group.get(role_group, 0) + 1
            role_group = self._creation_role_group(key)
            created_by_role_group[role_group] = created_by_role_group.get(role_group, 0) + 1

        def _topk_indices(scores: torch.Tensor, num_samples: int) -> torch.Tensor:
            if num_samples <= 0 or int(scores.numel()) == 0:
                return torch.empty(0, dtype=torch.long)
            k = min(int(num_samples), int(scores.numel()))
            try:
                return torch.topk(scores, k=k, largest=True, sorted=True).indices.detach().cpu()
            except RuntimeError:
                return torch.argsort(scores, descending=True)[:k].detach().cpu()

        def _allocate_proportional_budget(
            labels: List[str],
            mass_by_label: Dict[str, float],
            total_budget: int,
            capacity_by_label: Optional[Dict[str, int]] = None,
        ) -> Dict[str, int]:
            alloc = {label: 0 for label in labels}
            if total_budget <= 0 or not labels:
                return alloc
            positive = [label for label in labels if float(mass_by_label.get(label, 0.0)) > 0.0]
            if not positive:
                return alloc

            total_mass = float(sum(float(mass_by_label[label]) for label in positive))
            residuals: List[Tuple[float, str]] = []
            assigned = 0
            for label in positive:
                exact = float(total_budget) * float(mass_by_label[label]) / max(1e-12, total_mass)
                base = int(math.floor(exact))
                if capacity_by_label is not None:
                    base = min(base, int(capacity_by_label.get(label, 0)))
                alloc[label] = max(0, int(base))
                assigned += alloc[label]
                residuals.append((exact - float(base), label))

            leftover = max(0, int(total_budget) - int(assigned))
            for _frac, label in sorted(residuals, reverse=True):
                if leftover <= 0:
                    break
                if capacity_by_label is not None and alloc[label] >= int(capacity_by_label.get(label, 0)):
                    continue
                alloc[label] += 1
                leftover -= 1

            if leftover > 0:
                made_progress = True
                while leftover > 0 and made_progress:
                    made_progress = False
                    for label in positive:
                        if leftover <= 0:
                            break
                        if capacity_by_label is not None and alloc[label] >= int(capacity_by_label.get(label, 0)):
                            continue
                        alloc[label] += 1
                        leftover -= 1
                        made_progress = True
            return alloc

        def _selection_weights(raw_scores: torch.Tensor, gap_scores: torch.Tensor) -> torch.Tensor:
            mode = self.creation_selection_weight
            raw = raw_scores.detach().float().clamp_min(0.0)
            gap = gap_scores.detach().float().clamp_min(0.0)
            if mode == "score_gap":
                weights = raw * gap
            elif mode == "sqrt_score_gap":
                weights = torch.sqrt((raw * gap).clamp_min(0.0))
            else:
                weights = raw
            if float(weights.sum().item()) <= 0.0:
                if float(raw.sum().item()) > 0.0:
                    return raw
                return gap
            return weights

        def _available_quality_candidates(
            key: str,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            idx = eligible_idx_by_key.get(key)
            raw = eligible_raw_by_key.get(key)
            dirs = eligible_dir_by_key.get(key)
            interference = eligible_interference_by_key.get(key)
            if idx is None or raw is None or dirs is None or interference is None or int(idx.size(0)) == 0:
                empty_idx = torch.empty((0, 2), dtype=torch.long)
                empty_vals = torch.empty(0, dtype=torch.float32)
                return empty_idx, empty_vals, empty_vals, empty_vals
            shadow = mask_shadow[key]
            keep = ~shadow[idx[:, 0], idx[:, 1]]
            if int(keep.sum().item()) == 0:
                empty_idx = torch.empty((0, 2), dtype=torch.long)
                empty_vals = torch.empty(0, dtype=torch.float32)
                return empty_idx, empty_vals, empty_vals, empty_vals
            return idx[keep], raw[keep], dirs[keep], interference[keep]

        created_channels_by_prefix_epoch: Dict[str, int] = {}
        selected_hidden_degree_samples_by_prefix: Dict[str, List[float]] = {}
        selected_hidden_fire_rate_samples_by_prefix: Dict[str, List[float]] = {}
        selected_hidden_usage_samples_by_prefix: Dict[str, List[float]] = {}
        skipped_by_headroom_by_prefix_epoch: Dict[str, int] = {}
        skipped_by_incomplete_bundle_by_prefix_epoch: Dict[str, int] = {}
        skipped_by_density_cap_by_prefix_epoch: Dict[str, int] = {}
        block_density_by_prefix_epoch: Dict[str, float] = {}
        eligible_hidden_count_by_prefix_epoch: Dict[str, int] = {}

        floor_used = int(per_connection_quota)
        if self.creation_policy in {"paired_ffn_channel_random", "paired_ffn_channel_probe"}:
            floor_used = 0
            if not self.channel_creation_enabled:
                remaining_budget = 0
            else:
                channel_width = max(1, int(self.channel_fan_in + self.channel_fan_out))
                channel_budget = int(
                    math.floor(float(max_new_synapses) * float(self.channel_budget_fraction)) // channel_width
                )
                remaining_budget = min(int(remaining_budget), max(0, int(channel_budget * channel_width)))
                max_new_channels = max(0, int(remaining_budget // channel_width))
                block_capacity: Dict[str, int] = {}
                block_need: Dict[str, float] = {}
                eligible_channels_by_prefix: Dict[str, torch.Tensor] = {}
                block_prefixes: List[str] = []
                current_probationary_global = int(len(self._probationary_channels))
                probationary_counts_by_prefix = self._probationary_counts_by_prefix()
                probationary_global_remaining = max(
                    0,
                    int(self.max_probationary_channels_global) - current_probationary_global,
                )
                admission_budget = min(
                    int(max_new_channels),
                    int(probationary_global_remaining),
                    int(self.max_new_probationary_channels_per_epoch)
                    if int(self.max_new_probationary_channels_per_epoch) > 0
                    else int(probationary_global_remaining),
                )

                for up_key in [str(k) for k in tracked_creation_keys if str(k).endswith("_FFN_up")]:
                    prefix = up_key.rsplit("_FFN_up", 1)[0]
                    down_key = f"{prefix}_FFN_down"
                    if down_key not in tracked_creation_keys:
                        continue
                    up_shadow = mask_shadow[up_key]
                    down_shadow = mask_shadow[down_key]
                    if up_shadow.dim() != 2 or down_shadow.dim() != 2:
                        continue
                    block_density = 0.5 * (
                        float(up_shadow.detach().cpu().float().mean().item())
                        + float(down_shadow.detach().cpu().float().mean().item())
                    )
                    block_density_by_prefix_epoch[prefix] = block_density
                    if block_density >= float(self.block_density_cap):
                        skipped_by_density_cap_by_prefix_epoch[prefix] = (
                            skipped_by_density_cap_by_prefix_epoch.get(prefix, 0) + 1
                        )
                        continue
                    up_free = (~up_shadow).sum(dim=0).to(dtype=torch.long)
                    down_free = (~down_shadow).sum(dim=1).to(dtype=torch.long)
                    has_both_sides = (up_free > 0) & (down_free > 0)
                    if int(has_both_sides.sum().item()) <= 0:
                        continue
                    if self.channel_require_full_bundle:
                        has_bundle = (up_free >= int(self.channel_fan_in)) & (down_free >= int(self.channel_fan_out))
                    else:
                        has_bundle = has_both_sides
                    skipped_by_incomplete_bundle_by_prefix_epoch[prefix] = int(
                        (has_both_sides & (~has_bundle)).sum().item()
                    )
                    if int(has_bundle.sum().item()) <= 0:
                        continue
                    up_deg = up_shadow.sum(dim=0).detach().cpu().float()
                    down_deg = down_shadow.sum(dim=1).detach().cpu().float()
                    total_deg = up_deg + down_deg
                    positive_deg = total_deg[has_bundle & (total_deg > 0)]
                    if int(positive_deg.numel()) > 0:
                        deg_thr = float(torch.quantile(positive_deg, self.channel_underused_max_quantile).item())
                        eligible = has_bundle & (total_deg <= deg_thr)
                    else:
                        eligible = has_bundle
                    skipped_by_headroom_by_prefix_epoch[prefix] = int((has_bundle & (~eligible)).sum().item())
                    if int(eligible.sum().item()) <= 0:
                        continue
                    eligible_idx = eligible.nonzero(as_tuple=False).flatten().detach().cpu()
                    eligible_hidden_count_by_prefix_epoch[prefix] = int(eligible_idx.numel())
                    eligible_channels_by_prefix[prefix] = eligible_idx
                    prefix_probationary_remaining = max(
                        0,
                        int(self.max_probationary_channels_per_prefix)
                        - int(probationary_counts_by_prefix.get(prefix, 0)),
                    )
                    block_capacity[prefix] = min(
                        int(eligible_idx.numel()),
                        int(prefix_probationary_remaining),
                    )
                    if block_capacity[prefix] <= 0:
                        continue
                    block_need[prefix] = math.sqrt(
                        max(0.0, float(need_by_key.get(up_key, 0.0)))
                        * max(0.0, float(need_by_key.get(down_key, 0.0)))
                    )
                    block_prefixes.append(prefix)

                channel_alloc_by_prefix: Dict[str, int] = {}
                if admission_budget > 0 and block_prefixes:
                    positive_blocks = [p for p in block_prefixes if float(block_need.get(p, 0.0)) > 0.0]
                    mass_by_prefix = (
                        {p: float(block_need[p]) for p in positive_blocks}
                        if positive_blocks
                        else {p: 1.0 for p in block_prefixes}
                    )
                    channel_alloc_by_prefix = _allocate_proportional_budget(
                        labels=list(mass_by_prefix.keys()),
                        mass_by_label=mass_by_prefix,
                        total_budget=int(admission_budget),
                        capacity_by_label=block_capacity,
                    )

                for prefix in sorted(block_prefixes, key=lambda p: float(block_need.get(p, 0.0)), reverse=True):
                    channels_to_create = min(
                        int(channel_alloc_by_prefix.get(prefix, 0)),
                        int(block_capacity.get(prefix, 0)),
                        int(remaining_budget // channel_width),
                    )
                    if channels_to_create <= 0:
                        continue
                    up_key = f"{prefix}_FFN_up"
                    down_key = f"{prefix}_FFN_down"
                    available_hidden = eligible_channels_by_prefix[prefix]
                    if int(available_hidden.numel()) <= 0:
                        continue
                    if self.phase1_registry_enabled and self.creation_policy == "paired_ffn_channel_probe":
                        selection_weights = torch.ones(int(available_hidden.numel()), dtype=torch.float32)
                        for local_idx in range(int(available_hidden.numel())):
                            h_local = int(available_hidden[local_idx].item())
                            registry_entry = self._phase1_registry_entry(prefix, h_local, create=False)
                            rejected_count = int(registry_entry.get("n_rejected_current_level", 0)) if registry_entry else 0
                            attempts_count = int(registry_entry.get("n_attempts_current_level", 0)) if registry_entry else 0
                            bonus = (
                                float(self.phase1_unexplored_bonus)
                                if attempts_count <= 0
                                else float(self.phase1_reexplored_bonus)
                            )
                            selection_weights[local_idx] = float(
                                (float(self.phase1_reselection_decay) ** float(rejected_count))
                                * max(0.0, bonus)
                            )
                        chosen_order = self._weighted_sample_indices(selection_weights, channels_to_create).detach().cpu()
                        if int(chosen_order.numel()) <= 0:
                            chosen_order = torch.randperm(int(available_hidden.numel()))[:channels_to_create]
                    else:
                        chosen_order = torch.randperm(int(available_hidden.numel()))[:channels_to_create]
                    hidden_buf_mean = self._get_hidden_usage_mean(prefix).detach().cpu().float()
                    hidden_buf_fire = self._get_hidden_fire_rate(prefix).detach().cpu().float()
                    up_shadow = mask_shadow[up_key]
                    down_shadow = mask_shadow[down_key]
                    total_deg = up_shadow.sum(dim=0).detach().cpu().float() + down_shadow.sum(dim=1).detach().cpu().float()
                    for local_pos in chosen_order.tolist():
                        h = int(available_hidden[int(local_pos)].item())
                        up_inactive = (~up_shadow[:, h]).nonzero(as_tuple=False).flatten().detach().cpu()
                        down_inactive = (~down_shadow[h, :]).nonzero(as_tuple=False).flatten().detach().cpu()
                        if int(up_inactive.numel()) <= 0 or int(down_inactive.numel()) <= 0:
                            continue
                        if self.channel_require_full_bundle and (
                            int(up_inactive.numel()) < int(self.channel_fan_in)
                            or int(down_inactive.numel()) < int(self.channel_fan_out)
                        ):
                            continue
                        k_in = min(int(self.channel_fan_in), int(up_inactive.numel()))
                        k_out = min(int(self.channel_fan_out), int(down_inactive.numel()))
                        if k_in <= 0 or k_out <= 0:
                            continue
                        up_pick = torch.randperm(int(up_inactive.numel()))[:k_in]
                        down_pick = torch.randperm(int(down_inactive.numel()))[:k_out]
                        selected_up_rows: List[int] = []
                        selected_down_cols: List[int] = []
                        created_up = 0
                        created_down = 0
                        for idx_in in up_pick.tolist():
                            src_n = int(up_inactive[int(idx_in)].item())
                            val = float(torch.randn(1).item()) * float(self.channel_bundle_init_scale)
                            if _buffer_create(up_key, src_n, h, val):
                                remaining_budget -= 1
                                created_up += 1
                                selected_up_rows.append(src_n)
                                _record_created(up_key, is_group_budget=True)
                                if remaining_budget <= 0:
                                    break
                        if remaining_budget > 0:
                            for idx_out in down_pick.tolist():
                                dst_n = int(down_inactive[int(idx_out)].item())
                                val = float(torch.randn(1).item()) * float(self.channel_bundle_init_scale)
                                if _buffer_create(down_key, h, dst_n, val):
                                    remaining_budget -= 1
                                    created_down += 1
                                    selected_down_cols.append(dst_n)
                                    _record_created(down_key, is_group_budget=True)
                                    if remaining_budget <= 0:
                                        break
                        full_bundle_created = created_up == k_in and created_down == k_out
                        if created_up > 0 and created_down > 0:
                            created_channels_by_prefix_epoch[prefix] = (
                                created_channels_by_prefix_epoch.get(prefix, 0) + 1
                            )
                            selected_hidden_degree_samples_by_prefix.setdefault(prefix, []).append(
                                float(total_deg[h].item())
                            )
                            selected_hidden_fire_rate_samples_by_prefix.setdefault(prefix, []).append(
                                float(hidden_buf_fire[h].item()) if h < int(hidden_buf_fire.numel()) else 0.0
                            )
                            selected_hidden_usage_samples_by_prefix.setdefault(prefix, []).append(
                                float(hidden_buf_mean[h].item()) if h < int(hidden_buf_mean.numel()) else 0.0
                            )
                            if self.creation_policy == "paired_ffn_channel_probe" and full_bundle_created:
                                self._register_probationary_channel(
                                    prefix=prefix,
                                    hidden_idx=h,
                                    up_key=up_key,
                                    down_key=down_key,
                                    up_rows=selected_up_rows,
                                    down_cols=selected_down_cols,
                                    epoch_created=int(epoch_created),
                                )
                        if remaining_budget < channel_width:
                            break
                if self.phase1_registry_enabled:
                    self._phase1_eligible_hidden_ids_by_prefix = {
                        str(key): [int(v) for v in torch.as_tensor(values, dtype=torch.long).tolist()]
                        for key, values in eligible_channels_by_prefix.items()
                    }
        elif self.creation_policy == "uniform_quota":
            for key in ordered_keys:
                if remaining_budget <= 0:
                    break
                grp = int(self.connection_meta[key]["dst_group"])
                if layer_remaining.get(grp, 0) <= 0:
                    continue

                idx, scores, dirs = candidate_meta[key]
                local_k = min(per_connection_quota, int(idx.size(0)), remaining_budget, int(layer_remaining[grp]))
                if local_k <= 0:
                    continue

                sampled = self._weighted_sample_indices(scores, local_k)
                for local_idx in sampled.tolist():
                    src_n = int(idx[local_idx, 0].item())
                    dst_n = int(idx[local_idx, 1].item())
                    if _buffer_create(key, src_n, dst_n, _init_weight(float(dirs[local_idx].item()))):
                        remaining_budget -= 1
                        layer_remaining[grp] = max(0, layer_remaining[grp] - 1)
                        _record_created(key, is_floor=True)
                        if remaining_budget <= 0:
                            break

            if remaining_budget > 0:
                made_progress = True
                while remaining_budget > 0 and made_progress:
                    made_progress = False
                    for key in ordered_keys:
                        if remaining_budget <= 0:
                            break

                        grp = int(self.connection_meta[key]["dst_group"])
                        if layer_remaining.get(grp, 0) <= 0:
                            continue

                        scores = self._get_creation_score(key).detach().cpu().float()
                        dirs = self._get_creation_direction(key).detach().cpu().float()
                        candidates = (~mask_shadow[key]) & (scores > float(min_score))
                        if int(candidates.sum().item()) == 0:
                            continue

                        idx = candidates.nonzero(as_tuple=False)
                        cand_scores = scores[candidates]
                        if abs(temp - 1.0) > 1e-6:
                            cand_scores = cand_scores.pow(1.0 / temp)

                        sampled = self._weighted_sample_indices(cand_scores, 1)
                        if sampled.numel() == 0:
                            continue

                        local_idx = int(sampled.item())
                        src_n = int(idx[local_idx, 0].item())
                        dst_n = int(idx[local_idx, 1].item())
                        dir_vals = dirs[candidates]
                        if _buffer_create(key, src_n, dst_n, _init_weight(float(dir_vals[local_idx].item()))):
                            remaining_budget -= 1
                            layer_remaining[grp] = max(0, layer_remaining[grp] - 1)
                            _record_created(key)
                            made_progress = True
        elif self.creation_policy == "hybrid_global":
            floor_used = self.creation_floor_per_connection
            if floor_used <= 0:
                floor_used = max(1, int(math.ceil(per_connection_quota * 0.25)))

            for key in ordered_keys:
                if remaining_budget <= 0:
                    break
                idx, scores, dirs = candidate_meta[key]
                local_k = min(floor_used, int(idx.size(0)), remaining_budget)
                if local_k <= 0:
                    continue
                sampled = self._weighted_sample_indices(scores, local_k)
                for local_idx in sampled.tolist():
                    src_n = int(idx[local_idx, 0].item())
                    dst_n = int(idx[local_idx, 1].item())
                    if _buffer_create(key, src_n, dst_n, _init_weight(float(dirs[local_idx].item()))):
                        remaining_budget -= 1
                        _record_created(key, is_floor=True)
                        if remaining_budget <= 0:
                            break

            if remaining_budget > 0:
                global_scores: List[torch.Tensor] = []
                global_key_ids: List[torch.Tensor] = []
                global_rows: List[torch.Tensor] = []
                global_cols: List[torch.Tensor] = []
                global_dirs: List[torch.Tensor] = []

                for kid, key in enumerate(ordered_keys):
                    scores = self._get_creation_score(key).detach().cpu().float()
                    dirs = self._get_creation_direction(key).detach().cpu().float()
                    candidates = (~mask_shadow[key]) & (scores > float(min_score))
                    if int(candidates.sum().item()) == 0:
                        continue

                    idx = candidates.nonzero(as_tuple=False)
                    cand_scores = scores[candidates]
                    dir_vals = dirs[candidates]
                    if abs(temp - 1.0) > 1e-6:
                        cand_scores = cand_scores.pow(1.0 / temp)

                    p95 = float(torch.quantile(cand_scores, 0.95).item()) if int(cand_scores.numel()) > 0 else 0.0
                    denom = max(1e-12, p95)
                    norm_scores = (cand_scores / denom).clamp_min(0.0)
                    if int(norm_scores.numel()) <= 0:
                        continue

                    n = int(norm_scores.numel())
                    global_scores.append(norm_scores.reshape(-1))
                    global_key_ids.append(torch.full((n,), kid, dtype=torch.long))
                    global_rows.append(idx[:, 0].to(dtype=torch.long))
                    global_cols.append(idx[:, 1].to(dtype=torch.long))
                    global_dirs.append(dir_vals.reshape(-1))

                if global_scores:
                    all_scores = torch.cat(global_scores, dim=0)
                    all_key_ids = torch.cat(global_key_ids, dim=0)
                    all_rows = torch.cat(global_rows, dim=0)
                    all_cols = torch.cat(global_cols, dim=0)
                    all_dirs = torch.cat(global_dirs, dim=0)

                    take_k = min(int(remaining_budget), int(all_scores.numel()))
                    if take_k > 0:
                        try:
                            selected = torch.topk(all_scores, k=take_k, largest=True, sorted=True).indices
                        except RuntimeError:
                            selected = torch.argsort(all_scores, descending=True)[:take_k]

                        for pos in selected.tolist():
                            key = ordered_keys[int(all_key_ids[pos].item())]
                            src_n = int(all_rows[pos].item())
                            dst_n = int(all_cols[pos].item())
                            if _buffer_create(key, src_n, dst_n, _init_weight(float(all_dirs[pos].item()))):
                                remaining_budget -= 1
                                _record_created(key)
                                if remaining_budget <= 0:
                                    break
        elif self.creation_policy == "effect_gap_global":
            floor_used = self.creation_floor_per_connection
            if floor_used <= 0:
                floor_used = 16

            # Phase 1: small deterministic floor per matrix, chosen by absolute score.
            for key in ordered_keys:
                if remaining_budget <= 0:
                    break
                idx, _scores, dirs = candidate_meta[key]
                raw_scores = raw_scores_by_key[key]
                local_k = min(floor_used, int(idx.size(0)), remaining_budget)
                if local_k <= 0:
                    continue
                selected = _topk_indices(raw_scores, local_k)
                for local_idx in selected.tolist():
                    src_n = int(idx[local_idx, 0].item())
                    dst_n = int(idx[local_idx, 1].item())
                    if _buffer_create(key, src_n, dst_n, _init_weight(float(dirs[local_idx].item()))):
                        remaining_budget -= 1
                        _record_created(key, is_floor=True)
                        selected_priority_by_key[key].append(float(priority_scores_by_key[key][local_idx].item()))
                        if remaining_budget <= 0:
                            break

            # Phase 2: allocate the rest by matrix lack mass, then select top priority within each matrix.
            if remaining_budget > 0:
                available_after_floor: Dict[str, int] = {}
                matrix_mass: Dict[str, float] = {}
                for key in ordered_keys:
                    available = int((~mask_shadow[key]).sum().item())
                    available_after_floor[key] = available
                    if available <= 0:
                        matrix_mass[key] = 0.0
                        continue
                    mass = float(priority_mass_by_key.get(key, 0.0))
                    if mass <= 0.0:
                        mass = float(fallback_mass_by_key.get(key, 0.0))
                    matrix_mass[key] = mass

                positive_keys = [k for k in ordered_keys if available_after_floor.get(k, 0) > 0 and matrix_mass.get(k, 0.0) > 0.0]
                extra_budget_by_key: Dict[str, int] = {k: 0 for k in ordered_keys}
                if positive_keys:
                    total_mass = float(sum(matrix_mass[k] for k in positive_keys))
                    residuals: List[Tuple[float, str]] = []
                    allocated = 0
                    for key in positive_keys:
                        exact = float(remaining_budget) * float(matrix_mass[key]) / max(1e-12, total_mass)
                        base_alloc = min(available_after_floor[key], int(math.floor(exact)))
                        extra_budget_by_key[key] = base_alloc
                        allocated += base_alloc
                        residuals.append((exact - base_alloc, key))
                    leftover = max(0, remaining_budget - allocated)
                    for _frac, key in sorted(residuals, reverse=True):
                        if leftover <= 0:
                            break
                        if extra_budget_by_key[key] >= available_after_floor[key]:
                            continue
                        extra_budget_by_key[key] += 1
                        leftover -= 1

                    if leftover > 0:
                        made_progress = True
                        while leftover > 0 and made_progress:
                            made_progress = False
                            for key in positive_keys:
                                if leftover <= 0:
                                    break
                                if extra_budget_by_key[key] >= available_after_floor[key]:
                                    continue
                                extra_budget_by_key[key] += 1
                                leftover -= 1
                                made_progress = True

                for key in ordered_keys:
                    if remaining_budget <= 0:
                        break
                    local_k = min(extra_budget_by_key.get(key, 0), int((~mask_shadow[key]).sum().item()), remaining_budget)
                    if local_k <= 0:
                        continue

                    scores = self._get_creation_score(key).detach().cpu().float()
                    dirs = self._get_creation_direction(key).detach().cpu().float()
                    candidates = (~mask_shadow[key]) & (scores > float(min_score))
                    if int(candidates.sum().item()) == 0:
                        continue

                    idx = candidates.nonzero(as_tuple=False)
                    dir_vals = dirs[candidates]
                    # Recompute per current candidate set after floor-created updates.
                    raw_vals = scores[candidates]
                    threshold = float(threshold_by_key.get(key, 0.0))
                    priority_vals = raw_vals * (raw_vals - threshold).clamp_min(0.0)
                    selection_scores = priority_vals
                    if float(selection_scores.max().item()) <= 0.0:
                        selection_scores = raw_vals

                    selected = _topk_indices(selection_scores, local_k)
                    for local_idx in selected.tolist():
                        src_n = int(idx[local_idx, 0].item())
                        dst_n = int(idx[local_idx, 1].item())
                        if _buffer_create(key, src_n, dst_n, _init_weight(float(dir_vals[local_idx].item()))):
                            remaining_budget -= 1
                            _record_created(key)
                            selected_priority_by_key[key].append(float(priority_vals[local_idx].item()))
                            if remaining_budget <= 0:
                                break
        elif self.creation_policy == "group_balanced_effect_gap":
            floor_used = self.creation_floor_per_connection
            if floor_used <= 0:
                floor_used = 16

            # Phase 1: keep a small floor per matrix for exploration.
            for key in ordered_keys:
                if remaining_budget <= 0:
                    break
                idx, _scores, dirs = candidate_meta[key]
                raw_scores = raw_scores_by_key[key]
                local_k = min(floor_used, int(idx.size(0)), remaining_budget)
                if local_k <= 0:
                    continue
                selected = _topk_indices(raw_scores, local_k)
                for local_idx in selected.tolist():
                    src_n = int(idx[local_idx, 0].item())
                    dst_n = int(idx[local_idx, 1].item())
                    if _buffer_create(key, src_n, dst_n, _init_weight(float(dirs[local_idx].item()))):
                        remaining_budget -= 1
                        _record_created(key, is_floor=True)
                        selected_priority_by_key[key].append(float(priority_scores_by_key[key][local_idx].item()))
                        if remaining_budget <= 0:
                            break

            # Phase 2: balance budget between encoder and decoder groups,
            # then allocate inside each group by lack mass.
            if remaining_budget > 0:
                available_after_floor: Dict[str, int] = {}
                matrix_need: Dict[str, float] = {}
                keys_by_role: Dict[str, List[str]] = {"enc": [], "dec": [], "other": []}

                for key in ordered_keys:
                    available = int((~mask_shadow[key]).sum().item())
                    available_after_floor[key] = available
                    role_group = self._creation_role_group(key)
                    if available <= 0:
                        continue
                    need_mass = float(gap_mass_by_key.get(key, 0.0))
                    if need_mass <= 0.0:
                        need_mass = float(fallback_mass_by_key.get(key, 0.0))
                    matrix_need[key] = need_mass
                    if need_mass > 0.0:
                        keys_by_role.setdefault(role_group, []).append(key)

                positive_roles = [
                    role for role, keys in keys_by_role.items()
                    if keys and sum(float(matrix_need.get(key, 0.0)) for key in keys) > 0.0
                ]
                extra_budget_by_key: Dict[str, int] = {k: 0 for k in ordered_keys}

                if positive_roles:
                    role_budget_target: Dict[str, int] = {role: 0 for role in keys_by_role.keys()}
                    role_capacity = {
                        role: int(sum(available_after_floor.get(key, 0) for key in keys_by_role.get(role, [])))
                        for role in keys_by_role.keys()
                    }
                    role_mass = {
                        role: float(sum(float(matrix_need.get(key, 0.0)) for key in keys_by_role.get(role, [])))
                        for role in keys_by_role.keys()
                    }

                    if set(positive_roles) == {"enc", "dec"}:
                        total_role_mass = max(1e-12, float(role_mass["enc"] + role_mass["dec"]))
                        enc_raw_share = float(role_mass["enc"]) / total_role_mass
                        min_share = float(self.creation_group_min_share)
                        enc_share = min(max(enc_raw_share, min_share), 1.0 - min_share)
                        group_share_target_by_role_group["enc"] = float(enc_raw_share)
                        group_share_target_by_role_group["dec"] = float(1.0 - enc_raw_share)
                        group_share_applied_by_role_group["enc"] = float(enc_share)
                        group_share_applied_by_role_group["dec"] = float(1.0 - enc_share)
                        exact_role_budget = {
                            "enc": float(remaining_budget) * enc_share,
                            "dec": float(remaining_budget) * (1.0 - enc_share),
                        }
                        residuals: List[Tuple[float, str]] = []
                        assigned = 0
                        for role in ("enc", "dec"):
                            base_alloc = min(role_capacity.get(role, 0), int(math.floor(exact_role_budget[role])))
                            role_budget_target[role] = max(0, int(base_alloc))
                            assigned += role_budget_target[role]
                            residuals.append((exact_role_budget[role] - float(base_alloc), role))
                        leftover = max(0, int(remaining_budget) - int(assigned))
                        for _frac, role in sorted(residuals, reverse=True):
                            if leftover <= 0:
                                break
                            if role_budget_target[role] >= role_capacity.get(role, 0):
                                continue
                            role_budget_target[role] += 1
                            leftover -= 1
                        if leftover > 0:
                            role_alloc = _allocate_proportional_budget(
                                labels=["enc", "dec"],
                                mass_by_label={"enc": role_mass["enc"], "dec": role_mass["dec"]},
                                total_budget=int(leftover),
                                capacity_by_label={
                                    "enc": max(0, role_capacity.get("enc", 0) - role_budget_target["enc"]),
                                    "dec": max(0, role_capacity.get("dec", 0) - role_budget_target["dec"]),
                                },
                            )
                            for role, extra in role_alloc.items():
                                role_budget_target[role] += int(extra)
                    else:
                        role_budget_target = _allocate_proportional_budget(
                            labels=positive_roles,
                            mass_by_label=role_mass,
                            total_budget=int(remaining_budget),
                            capacity_by_label=role_capacity,
                        )

                    for role in positive_roles:
                        role_keys = keys_by_role.get(role, [])
                        role_budget = int(role_budget_target.get(role, 0))
                        if role_budget <= 0 or not role_keys:
                            continue
                        matrix_capacity = {key: int(available_after_floor.get(key, 0)) for key in role_keys}
                        matrix_budget = _allocate_proportional_budget(
                            labels=role_keys,
                            mass_by_label=matrix_need,
                            total_budget=role_budget,
                            capacity_by_label=matrix_capacity,
                        )
                        for key, alloc in matrix_budget.items():
                            extra_budget_by_key[key] += int(alloc)

                for key in ordered_keys:
                    if remaining_budget <= 0:
                        break
                    local_k = min(extra_budget_by_key.get(key, 0), int((~mask_shadow[key]).sum().item()), remaining_budget)
                    if local_k <= 0:
                        continue

                    scores = self._get_creation_score(key).detach().cpu().float()
                    dirs = self._get_creation_direction(key).detach().cpu().float()
                    candidates = (~mask_shadow[key]) & (scores > float(min_score))
                    if int(candidates.sum().item()) == 0:
                        continue

                    idx = candidates.nonzero(as_tuple=False)
                    dir_vals = dirs[candidates]
                    raw_vals = scores[candidates]
                    gap_vals = gap_scores_by_key.get(key)
                    if gap_vals is None or int(gap_vals.numel()) != int(raw_vals.numel()):
                        threshold = float(threshold_by_key.get(key, 0.0))
                        gap_vals = (raw_vals - threshold).clamp_min(0.0)
                    eligible = gap_vals > 0.0
                    selection_idx = idx
                    selection_dirs = dir_vals
                    selection_raw = raw_vals
                    selection_gap = gap_vals
                    if int(eligible.sum().item()) > 0:
                        selection_idx = idx[eligible]
                        selection_dirs = dir_vals[eligible]
                        selection_raw = raw_vals[eligible]
                        selection_gap = gap_vals[eligible]

                    selected = _topk_indices(selection_raw, local_k)
                    for local_idx in selected.tolist():
                        src_n = int(selection_idx[local_idx, 0].item())
                        dst_n = int(selection_idx[local_idx, 1].item())
                        if _buffer_create(
                            key,
                            src_n,
                            dst_n,
                            _init_weight(float(selection_dirs[local_idx].item())),
                        ):
                            remaining_budget -= 1
                            _record_created(key, is_group_budget=True)
                            selected_priority_by_key[key].append(
                                float((selection_raw[local_idx] * selection_gap[local_idx]).item())
                            )
                            if remaining_budget <= 0:
                                break
        elif self.creation_policy == "group_calibrated_weighted":
            if self.creation_auto_budget_from_role_need:
                tracked_roles = [role for role in ("enc", "dec") if float(relative_gap_mass_by_role_group.get(role, 0.0)) > 0.0]
                current_total_need = 0.0
                ema_total_need = 0.0
                ema_alpha = float(self.creation_role_need_ema_alpha)
                for role in ("enc", "dec", "other"):
                    current_need = float(relative_gap_mass_by_role_group.get(role, 0.0))
                    role_need_by_role_group[role] = current_need
                    prev_need = self._creation_role_need_ema.get(role)
                    if prev_need is None:
                        role_ema = current_need
                    else:
                        role_ema = float(ema_alpha) * current_need + (1.0 - float(ema_alpha)) * float(prev_need)
                    self._creation_role_need_ema[role] = float(role_ema)
                    role_need_ema_by_role_group[role] = float(role_ema)
                    if role in tracked_roles:
                        current_total_need += current_need
                        ema_total_need += float(role_ema)
                if tracked_roles and ema_total_need > 1e-12:
                    budget_scale = float(current_total_need / ema_total_need)
                    budget_scale = min(1.0, max(float(self.creation_auto_budget_min_ratio), budget_scale))
                    budget_effective = max(1, int(round(float(max_new_synapses) * budget_scale)))
                    remaining_budget = min(int(remaining_budget), int(budget_effective))
            floor_used = self.creation_floor_per_connection
            if floor_used <= 0:
                floor_used = 16

            # Phase 1: preserve mild exploration within each matrix.
            for key in ordered_keys:
                if remaining_budget <= 0:
                    break
                idx, scores, dirs = candidate_meta[key]
                local_k = min(floor_used, int(idx.size(0)), remaining_budget)
                if local_k <= 0:
                    continue
                sampled = self._weighted_sample_indices(scores, local_k)
                for local_idx in sampled.tolist():
                    src_n = int(idx[local_idx, 0].item())
                    dst_n = int(idx[local_idx, 1].item())
                    if _buffer_create(key, src_n, dst_n, _init_weight(float(dirs[local_idx].item()))):
                        remaining_budget -= 1
                        _record_created(key, is_floor=True)
                        selected_priority_by_key[key].append(float(priority_scores_by_key[key][local_idx].item()))
                        if remaining_budget <= 0:
                            break

            if remaining_budget > 0:
                available_after_floor: Dict[str, int] = {}
                matrix_need: Dict[str, float] = {}
                keys_by_role: Dict[str, List[str]] = {"enc": [], "dec": [], "other": []}

                for key in ordered_keys:
                    available = int((~mask_shadow[key]).sum().item())
                    available_after_floor[key] = available
                    role_group = self._creation_role_group(key)
                    if available <= 0:
                        continue
                    need_mass = float(relative_gap_mass_by_key.get(key, 0.0))
                    if need_mass <= 0.0:
                        scale = max(1e-12, float(score_p95_by_key.get(key, 0.0)))
                        fallback = float(fallback_mass_by_key.get(key, 0.0))
                        need_mass = float(fallback / scale)
                    matrix_need[key] = need_mass
                    if need_mass > 0.0:
                        keys_by_role.setdefault(role_group, []).append(key)

                positive_roles = [
                    role for role, keys in keys_by_role.items()
                    if keys and sum(float(matrix_need.get(key, 0.0)) for key in keys) > 0.0
                ]
                extra_budget_by_key: Dict[str, int] = {k: 0 for k in ordered_keys}

                if positive_roles:
                    role_budget_target: Dict[str, int] = {role: 0 for role in keys_by_role.keys()}
                    role_capacity = {
                        role: int(sum(available_after_floor.get(key, 0) for key in keys_by_role.get(role, [])))
                        for role in keys_by_role.keys()
                    }
                    role_mass = {
                        role: float(sum(float(matrix_need.get(key, 0.0)) for key in keys_by_role.get(role, [])))
                        for role in keys_by_role.keys()
                    }

                    if set(positive_roles) == {"enc", "dec"}:
                        total_role_mass = max(1e-12, float(role_mass["enc"] + role_mass["dec"]))
                        enc_raw_share = float(role_mass["enc"]) / total_role_mass
                        ema_alpha = float(self.creation_group_share_ema_alpha)
                        prev_ema = self._creation_group_share_ema.get("enc")
                        if prev_ema is None:
                            enc_share_ema = float(enc_raw_share)
                        else:
                            enc_share_ema = float(ema_alpha) * float(enc_raw_share) + (1.0 - float(ema_alpha)) * float(prev_ema)
                        self._creation_group_share_ema["enc"] = float(enc_share_ema)
                        self._creation_group_share_ema["dec"] = float(1.0 - enc_share_ema)
                        mix = float(self.creation_group_mix)
                        prior_enc_share = float(self.creation_group_prior_enc_share)
                        enc_share = (1.0 - mix) * prior_enc_share + mix * enc_share_ema
                        min_share = float(self.creation_group_min_share)
                        enc_share = min(max(enc_share, min_share), 1.0 - min_share)
                        group_share_target_by_role_group["enc"] = float(enc_raw_share)
                        group_share_target_by_role_group["dec"] = float(1.0 - enc_raw_share)
                        group_share_ema_by_role_group["enc"] = float(enc_share_ema)
                        group_share_ema_by_role_group["dec"] = float(1.0 - enc_share_ema)
                        group_share_applied_by_role_group["enc"] = float(enc_share)
                        group_share_applied_by_role_group["dec"] = float(1.0 - enc_share)
                        exact_role_budget = {
                            "enc": float(remaining_budget) * enc_share,
                            "dec": float(remaining_budget) * (1.0 - enc_share),
                        }
                        residuals: List[Tuple[float, str]] = []
                        assigned = 0
                        for role in ("enc", "dec"):
                            base_alloc = min(role_capacity.get(role, 0), int(math.floor(exact_role_budget[role])))
                            role_budget_target[role] = max(0, int(base_alloc))
                            assigned += role_budget_target[role]
                            residuals.append((exact_role_budget[role] - float(base_alloc), role))
                        leftover = max(0, int(remaining_budget) - int(assigned))
                        for _frac, role in sorted(residuals, reverse=True):
                            if leftover <= 0:
                                break
                            if role_budget_target[role] >= role_capacity.get(role, 0):
                                continue
                            role_budget_target[role] += 1
                            leftover -= 1
                        if leftover > 0:
                            role_alloc = _allocate_proportional_budget(
                                labels=["enc", "dec"],
                                mass_by_label={"enc": role_mass["enc"], "dec": role_mass["dec"]},
                                total_budget=int(leftover),
                                capacity_by_label={
                                    "enc": max(0, role_capacity.get("enc", 0) - role_budget_target["enc"]),
                                    "dec": max(0, role_capacity.get("dec", 0) - role_budget_target["dec"]),
                                },
                            )
                            for role, extra in role_alloc.items():
                                role_budget_target[role] += int(extra)
                    else:
                        role_budget_target = _allocate_proportional_budget(
                            labels=positive_roles,
                            mass_by_label=role_mass,
                            total_budget=int(remaining_budget),
                            capacity_by_label=role_capacity,
                        )

                    for role in positive_roles:
                        role_keys = keys_by_role.get(role, [])
                        role_budget = int(role_budget_target.get(role, 0))
                        if role_budget <= 0 or not role_keys:
                            continue
                        matrix_capacity = {key: int(available_after_floor.get(key, 0)) for key in role_keys}
                        matrix_budget = _allocate_proportional_budget(
                            labels=role_keys,
                            mass_by_label=matrix_need,
                            total_budget=role_budget,
                            capacity_by_label=matrix_capacity,
                        )
                        for key, alloc in matrix_budget.items():
                            extra_budget_by_key[key] += int(alloc)

                for key in ordered_keys:
                    if remaining_budget <= 0:
                        break
                    local_k = min(extra_budget_by_key.get(key, 0), int((~mask_shadow[key]).sum().item()), remaining_budget)
                    if local_k <= 0:
                        continue

                    scores = self._get_creation_score(key).detach().cpu().float()
                    dirs = self._get_creation_direction(key).detach().cpu().float()
                    candidates = (~mask_shadow[key]) & (scores > float(min_score))
                    if int(candidates.sum().item()) == 0:
                        continue

                    idx = candidates.nonzero(as_tuple=False)
                    dir_vals = dirs[candidates]
                    raw_vals = scores[candidates]
                    threshold = float(threshold_by_key.get(key, 0.0))
                    gap_vals = (raw_vals - threshold).clamp_min(0.0)
                    eligible = gap_vals > 0.0
                    selection_idx = idx
                    selection_dirs = dir_vals
                    selection_raw = raw_vals
                    selection_gap = gap_vals
                    if int(eligible.sum().item()) > 0:
                        selection_idx = idx[eligible]
                        selection_dirs = dir_vals[eligible]
                        selection_raw = raw_vals[eligible]
                        selection_gap = gap_vals[eligible]

                    selection_weights = _selection_weights(selection_raw, selection_gap)
                    sampled = self._weighted_sample_indices(selection_weights, local_k)
                    for local_idx in sampled.tolist():
                        src_n = int(selection_idx[local_idx, 0].item())
                        dst_n = int(selection_idx[local_idx, 1].item())
                        if _buffer_create(
                            key,
                            src_n,
                            dst_n,
                            _init_weight(float(selection_dirs[local_idx].item())),
                        ):
                            remaining_budget -= 1
                            _record_created(key, is_group_budget=True)
                            selected_priority_by_key[key].append(
                                float((selection_raw[local_idx] * selection_gap[local_idx]).item())
                            )
                            if remaining_budget <= 0:
                                break
        elif self.creation_policy == "group_frustration_weighted":
            if self.creation_auto_budget_from_role_need:
                tracked_roles = [
                    role
                    for role in ("enc", "dec")
                    if float(relative_frustration_mass_by_role_group.get(role, 0.0)) > 0.0
                ]
                current_total_need = 0.0
                ema_total_need = 0.0
                ema_alpha = float(self.creation_role_need_ema_alpha)
                for role in ("enc", "dec", "other"):
                    current_need = float(relative_frustration_mass_by_role_group.get(role, 0.0))
                    role_need_by_role_group[role] = current_need
                    prev_need = self._creation_role_need_ema.get(role)
                    if prev_need is None:
                        role_ema = current_need
                    else:
                        role_ema = float(ema_alpha) * current_need + (1.0 - float(ema_alpha)) * float(prev_need)
                    self._creation_role_need_ema[role] = float(role_ema)
                    role_need_ema_by_role_group[role] = float(role_ema)
                    if role in tracked_roles:
                        current_total_need += current_need
                        ema_total_need += float(role_ema)
                if tracked_roles and ema_total_need > 1e-12:
                    budget_scale = float(current_total_need / ema_total_need)
                    budget_scale = min(1.0, max(float(self.creation_auto_budget_min_ratio), budget_scale))
                    budget_effective = max(1, int(round(float(max_new_synapses) * budget_scale)))
                    remaining_budget = min(int(remaining_budget), int(budget_effective))
            floor_used = self.creation_floor_per_connection
            if floor_used <= 0:
                floor_used = 16

            for key in ordered_keys:
                if remaining_budget <= 0:
                    break
                idx, scores, dirs = candidate_meta[key]
                local_k = min(floor_used, int(idx.size(0)), remaining_budget)
                if local_k <= 0:
                    continue
                sampled = self._weighted_sample_indices(scores, local_k)
                for local_idx in sampled.tolist():
                    src_n = int(idx[local_idx, 0].item())
                    dst_n = int(idx[local_idx, 1].item())
                    if _buffer_create(key, src_n, dst_n, _init_weight(float(dirs[local_idx].item()))):
                        remaining_budget -= 1
                        _record_created(key, is_floor=True)
                        selected_priority_by_key[key].append(float(raw_scores_by_key[key][local_idx].item()))
                        if remaining_budget <= 0:
                            break

            if remaining_budget > 0:
                available_after_floor: Dict[str, int] = {}
                matrix_need: Dict[str, float] = {}
                keys_by_role: Dict[str, List[str]] = {"enc": [], "dec": [], "other": []}

                for key in ordered_keys:
                    available = int((~mask_shadow[key]).sum().item())
                    available_after_floor[key] = available
                    role_group = self._creation_role_group(key)
                    if available <= 0:
                        continue
                    need_mass = float(relative_frustration_mass_by_key.get(key, 0.0))
                    if need_mass <= 0.0:
                        scale = max(1e-12, float(score_p95_by_key.get(key, 0.0)))
                        fallback = float(fallback_mass_by_key.get(key, 0.0))
                        need_mass = float(fallback / scale)
                    matrix_need[key] = need_mass
                    if need_mass > 0.0:
                        keys_by_role.setdefault(role_group, []).append(key)

                positive_roles = [
                    role for role, keys in keys_by_role.items()
                    if keys and sum(float(matrix_need.get(key, 0.0)) for key in keys) > 0.0
                ]
                extra_budget_by_key: Dict[str, int] = {k: 0 for k in ordered_keys}

                if positive_roles:
                    role_budget_target: Dict[str, int] = {role: 0 for role in keys_by_role.keys()}
                    role_capacity = {
                        role: int(sum(available_after_floor.get(key, 0) for key in keys_by_role.get(role, [])))
                        for role in keys_by_role.keys()
                    }
                    role_mass = {
                        role: float(sum(float(matrix_need.get(key, 0.0)) for key in keys_by_role.get(role, [])))
                        for role in keys_by_role.keys()
                    }

                    if set(positive_roles) == {"enc", "dec"}:
                        total_role_mass = max(1e-12, float(role_mass["enc"] + role_mass["dec"]))
                        enc_raw_share = float(role_mass["enc"]) / total_role_mass
                        ema_alpha = float(self.creation_group_share_ema_alpha)
                        prev_ema = self._creation_group_share_ema.get("enc")
                        if prev_ema is None:
                            enc_share_ema = float(enc_raw_share)
                        else:
                            enc_share_ema = float(ema_alpha) * float(enc_raw_share) + (1.0 - float(ema_alpha)) * float(prev_ema)
                        self._creation_group_share_ema["enc"] = float(enc_share_ema)
                        self._creation_group_share_ema["dec"] = float(1.0 - enc_share_ema)
                        mix = float(self.creation_group_mix)
                        prior_enc_share = float(self.creation_group_prior_enc_share)
                        enc_share = (1.0 - mix) * prior_enc_share + mix * enc_share_ema
                        min_share = float(self.creation_group_min_share)
                        enc_share = min(max(enc_share, min_share), 1.0 - min_share)
                        group_share_target_by_role_group["enc"] = float(enc_raw_share)
                        group_share_target_by_role_group["dec"] = float(1.0 - enc_raw_share)
                        group_share_ema_by_role_group["enc"] = float(enc_share_ema)
                        group_share_ema_by_role_group["dec"] = float(1.0 - enc_share_ema)
                        group_share_applied_by_role_group["enc"] = float(enc_share)
                        group_share_applied_by_role_group["dec"] = float(1.0 - enc_share)
                        exact_role_budget = {
                            "enc": float(remaining_budget) * enc_share,
                            "dec": float(remaining_budget) * (1.0 - enc_share),
                        }
                        residuals: List[Tuple[float, str]] = []
                        assigned = 0
                        for role in ("enc", "dec"):
                            base_alloc = min(role_capacity.get(role, 0), int(math.floor(exact_role_budget[role])))
                            role_budget_target[role] = max(0, int(base_alloc))
                            assigned += role_budget_target[role]
                            residuals.append((exact_role_budget[role] - float(base_alloc), role))
                        leftover = max(0, int(remaining_budget) - int(assigned))
                        for _frac, role in sorted(residuals, reverse=True):
                            if leftover <= 0:
                                break
                            if role_budget_target[role] >= role_capacity.get(role, 0):
                                continue
                            role_budget_target[role] += 1
                            leftover -= 1
                        if leftover > 0:
                            role_alloc = _allocate_proportional_budget(
                                labels=["enc", "dec"],
                                mass_by_label={"enc": role_mass["enc"], "dec": role_mass["dec"]},
                                total_budget=int(leftover),
                                capacity_by_label={
                                    "enc": max(0, role_capacity.get("enc", 0) - role_budget_target["enc"]),
                                    "dec": max(0, role_capacity.get("dec", 0) - role_budget_target["dec"]),
                                },
                            )
                            for role, extra in role_alloc.items():
                                role_budget_target[role] += int(extra)
                    else:
                        role_budget_target = _allocate_proportional_budget(
                            labels=positive_roles,
                            mass_by_label=role_mass,
                            total_budget=int(remaining_budget),
                            capacity_by_label=role_capacity,
                        )

                    for role in positive_roles:
                        role_keys = keys_by_role.get(role, [])
                        role_budget = int(role_budget_target.get(role, 0))
                        if role_budget <= 0 or not role_keys:
                            continue
                        matrix_capacity = {key: int(available_after_floor.get(key, 0)) for key in role_keys}
                        matrix_budget = _allocate_proportional_budget(
                            labels=role_keys,
                            mass_by_label=matrix_need,
                            total_budget=role_budget,
                            capacity_by_label=matrix_capacity,
                        )
                        for key, alloc in matrix_budget.items():
                            extra_budget_by_key[key] += int(alloc)

                for key in ordered_keys:
                    if remaining_budget <= 0:
                        break
                    local_k = min(extra_budget_by_key.get(key, 0), int((~mask_shadow[key]).sum().item()), remaining_budget)
                    if local_k <= 0:
                        continue

                    scores = self._get_creation_score(key).detach().cpu().float()
                    dirs = self._get_creation_direction(key).detach().cpu().float()
                    candidates = (~mask_shadow[key]) & (scores > float(min_score))
                    if int(candidates.sum().item()) == 0:
                        continue

                    idx = candidates.nonzero(as_tuple=False)
                    dir_vals = dirs[candidates]
                    raw_vals = scores[candidates]
                    interference_vals = (raw_vals - torch.minimum(raw_vals, dir_vals.abs())).clamp_min(0.0)
                    selection_scores = interference_vals
                    if float(selection_scores.max().item()) <= 0.0:
                        selection_scores = raw_vals

                    selected = _topk_indices(selection_scores, local_k)
                    for local_idx in selected.tolist():
                        src_n = int(idx[local_idx, 0].item())
                        dst_n = int(idx[local_idx, 1].item())
                        if _buffer_create(
                            key,
                            src_n,
                            dst_n,
                            _init_weight(float(dir_vals[local_idx].item())),
                        ):
                            remaining_budget -= 1
                            _record_created(key, is_group_budget=True)
                            selected_priority_by_key[key].append(float(selection_scores[local_idx].item()))
                            if remaining_budget <= 0:
                                break
        elif self.creation_policy in {"need_quality_filtered", "need_quality_inverse_coverage"}:
            floor_used = int(self.creation_floor_per_connection)
            if floor_used <= 0:
                floor_used = 4

            for key in ordered_keys:
                if remaining_budget <= 0:
                    break
                if int(eligible_count_by_key.get(key, 0)) <= 0 or float(need_by_key.get(key, 0.0)) <= 0.0:
                    continue
                idx, raw_vals, dir_vals, interference_vals = _available_quality_candidates(key)
                local_k = min(int(floor_used), int(idx.size(0)), int(remaining_budget))
                if local_k <= 0:
                    continue
                selection_scores = interference_vals
                if int(selection_scores.numel()) > 0 and float(selection_scores.max().item()) <= 0.0:
                    selection_scores = raw_vals
                selected = _topk_indices(selection_scores, local_k)
                for local_idx in selected.tolist():
                    src_n = int(idx[local_idx, 0].item())
                    dst_n = int(idx[local_idx, 1].item())
                    if _buffer_create(
                        key,
                        src_n,
                        dst_n,
                        _init_weight(float(dir_vals[local_idx].item())),
                    ):
                        remaining_budget -= 1
                        _record_created(key, is_floor=True)
                        selected_priority_by_key[key].append(float(selection_scores[local_idx].item()))
                        if remaining_budget <= 0:
                            break

            if remaining_budget > 0:
                available_after_floor: Dict[str, int] = {}
                matrix_need: Dict[str, float] = {}
                extra_budget_by_key: Dict[str, int] = {k: 0 for k in ordered_keys}
                positive_keys: List[str] = []

                for key in ordered_keys:
                    idx, _raw_vals, _dir_vals, _interference_vals = _available_quality_candidates(key)
                    available = int(idx.size(0))
                    available_after_floor[key] = available
                    if available <= 0:
                        continue
                    need_mass = float(need_ema_by_key.get(key, 0.0))
                    if need_mass <= 0.0:
                        continue
                    matrix_need[key] = need_mass
                    positive_keys.append(key)

                if positive_keys:
                    extra_budget_by_key = _allocate_proportional_budget(
                        labels=positive_keys,
                        mass_by_label=matrix_need,
                        total_budget=int(remaining_budget),
                        capacity_by_label=available_after_floor,
                    )

                for key in ordered_keys:
                    if remaining_budget <= 0:
                        break
                    local_k = min(
                        int(extra_budget_by_key.get(key, 0)),
                        int(available_after_floor.get(key, 0)),
                        int(remaining_budget),
                    )
                    if local_k <= 0:
                        continue
                    idx, raw_vals, dir_vals, interference_vals = _available_quality_candidates(key)
                    if int(idx.size(0)) == 0:
                        continue
                    selection_scores = interference_vals
                    if int(selection_scores.numel()) > 0 and float(selection_scores.max().item()) <= 0.0:
                        selection_scores = raw_vals
                    selected = _topk_indices(selection_scores, local_k)
                    for local_idx in selected.tolist():
                        src_n = int(idx[local_idx, 0].item())
                        dst_n = int(idx[local_idx, 1].item())
                        if _buffer_create(
                            key,
                            src_n,
                            dst_n,
                            _init_weight(float(dir_vals[local_idx].item())),
                        ):
                            remaining_budget -= 1
                            _record_created(key, is_group_budget=True)
                            selected_priority_by_key[key].append(float(selection_scores[local_idx].item()))
                            if remaining_budget <= 0:
                                break
        else:
            raise ValueError(f"Unknown creation policy: {self.creation_policy}")

        created = 0
        with torch.no_grad():
            for key in ordered_keys:
                rows = pending_rows.get(key, [])
                if not rows:
                    continue
                cols = pending_cols[key]
                vals = pending_vals[key]

                w = self.weights[key]
                dev = w.device
                rows_t = torch.tensor(rows, device=dev, dtype=torch.long)
                cols_t = torch.tensor(cols, device=dev, dtype=torch.long)
                vals_t = torch.tensor(vals, device=dev, dtype=w.dtype)

                mask = getattr(self, self.masks[key])
                mask[rows_t, cols_t] = 1.0
                w.data[rows_t, cols_t] = vals_t

                self._get_usage(key)[rows_t, cols_t] = 0.0
                self._get_usage_peak(key)[rows_t, cols_t] = 0.0
                self._get_age(key)[rows_t, cols_t] = 0.0
                self._get_replay_tag(key)[rows_t, cols_t] = 0.0
                self._get_tenure_mask(key)[rows_t, cols_t] = 0.0
                self._get_tenure_score_ema(key)[rows_t, cols_t] = 0.0
                self._get_tenure_streak(key)[rows_t, cols_t] = 0
                created += len(rows)

        if created > 0:
            self.creation_count += int(created)
        created_channels_epoch = int(sum(int(v) for v in created_channels_by_prefix_epoch.values()))
        created_channel_edges_epoch = int(created)
        if created_channels_epoch > 0:
            self.created_channels_total += int(created_channels_epoch)
            self._created_channels_total_current_level += int(created_channels_epoch)
            self.created_channel_edges_total += int(created_channel_edges_epoch)
            for prefix, value in created_channels_by_prefix_epoch.items():
                self.created_channels_total_by_prefix[prefix] = (
                    int(self.created_channels_total_by_prefix.get(prefix, 0)) + int(value)
                )
        selected_hidden_mean_degree_by_prefix = {
            prefix: (float(sum(vals) / len(vals)) if vals else 0.0)
            for prefix, vals in selected_hidden_degree_samples_by_prefix.items()
        }
        selected_hidden_p75_degree_by_prefix = {
            prefix: (
                float(torch.quantile(torch.tensor(vals, dtype=torch.float32), 0.75).item())
                if vals else 0.0
            )
            for prefix, vals in selected_hidden_degree_samples_by_prefix.items()
        }
        selected_hidden_mean_fire_rate_by_prefix = {
            prefix: (float(sum(vals) / len(vals)) if vals else 0.0)
            for prefix, vals in selected_hidden_fire_rate_samples_by_prefix.items()
        }
        selected_hidden_mean_usage_by_prefix = {
            prefix: (float(sum(vals) / len(vals)) if vals else 0.0)
            for prefix, vals in selected_hidden_usage_samples_by_prefix.items()
        }
        selected_priority_mean_by_key = {
            k: (float(sum(v) / len(v)) if v else 0.0)
            for k, v in selected_priority_by_key.items()
        }
        selected_priority_max_by_key = {
            k: (float(max(v)) if v else 0.0)
            for k, v in selected_priority_by_key.items()
        }
        _set_creation_stats(
            budget_requested=int(max_new_synapses),
            budget_used=int(created),
            candidate_total=sum(candidate_count_by_key.values()),
            ordered_keys=ordered_keys,
            candidate_count_by_key=candidate_count_by_key,
            eligible_count_before_quality_by_key=eligible_count_before_quality_by_key,
            eligible_count_by_key=eligible_count_by_key,
            score_mean_by_key=score_mean_by_key,
            score_p95_by_key=score_p95_by_key,
            score_max_by_key=score_max_by_key,
            active_ref_by_key=active_ref_by_key,
            threshold_by_key=threshold_by_key,
            quality_threshold_by_key=quality_threshold_by_key,
            gap_mass_by_key=gap_mass_by_key,
            relative_gap_mass_by_key=relative_gap_mass_by_key,
            priority_mass_by_key=priority_mass_by_key,
            frustration_mass_by_key=frustration_mass_by_key,
            relative_frustration_mass_by_key=relative_frustration_mass_by_key,
            coherence_mean_by_key=coherence_mean_by_key,
            coherence_p50_by_key=coherence_p50_by_key,
            coherence_p75_by_key=coherence_p75_by_key,
            coherence_p90_by_key=coherence_p90_by_key,
            interference_mean_by_key=interference_mean_by_key,
            interference_p95_by_key=interference_p95_by_key,
            interference_max_by_key=interference_max_by_key,
            support_mean_by_key=support_mean_by_key,
            support_p75_by_key=support_p75_by_key,
            support_p90_by_key=support_p90_by_key,
            support_scale_by_key=support_scale_by_key,
            scarcity_mean_by_key=scarcity_mean_by_key,
            scarcity_p95_by_key=scarcity_p95_by_key,
            scarcity_max_by_key=scarcity_max_by_key,
            need_score_mean_by_key=need_score_mean_by_key,
            need_score_p95_by_key=need_score_p95_by_key,
            need_score_max_by_key=need_score_max_by_key,
            need_interference_mean_by_key=need_interference_mean_by_key,
            need_interference_p95_by_key=need_interference_p95_by_key,
            need_by_key=need_by_key,
            need_ema_by_key=need_ema_by_key,
            priority_positive_count_by_key=priority_positive_count_by_key,
            selected_priority_mean_by_key=selected_priority_mean_by_key,
            selected_priority_max_by_key=selected_priority_max_by_key,
            created_by_key=created_by_key,
            created_floor_by_key=created_floor_by_key,
            created_global_by_key=created_global_by_key,
            group_budget_by_key=group_budget_by_key,
            group_budget_by_role_group=group_budget_by_role_group,
            candidate_count_by_role_group=candidate_count_by_role_group,
            eligible_count_by_role_group=eligible_count_by_role_group,
            gap_mass_by_role_group=gap_mass_by_role_group,
            relative_gap_mass_by_role_group=relative_gap_mass_by_role_group,
            priority_mass_by_role_group=priority_mass_by_role_group,
            frustration_mass_by_role_group=frustration_mass_by_role_group,
            created_by_role_group=created_by_role_group,
            group_share_target_by_role_group=group_share_target_by_role_group,
            group_share_ema_by_role_group=group_share_ema_by_role_group,
            group_share_applied_by_role_group=group_share_applied_by_role_group,
            role_need_by_role_group=role_need_by_role_group,
            role_need_ema_by_role_group=role_need_ema_by_role_group,
            created_channels_epoch=created_channels_epoch,
            created_channel_edges_epoch=created_channel_edges_epoch,
            created_channels_by_prefix=created_channels_by_prefix_epoch,
            selected_hidden_mean_degree_by_prefix=selected_hidden_mean_degree_by_prefix,
            selected_hidden_p75_degree_by_prefix=selected_hidden_p75_degree_by_prefix,
            selected_hidden_mean_fire_rate_by_prefix=selected_hidden_mean_fire_rate_by_prefix,
            selected_hidden_mean_usage_by_prefix=selected_hidden_mean_usage_by_prefix,
            created_channels_total=int(self.created_channels_total),
            created_channel_edges_total=int(self.created_channel_edges_total),
            created_channels_total_by_prefix=dict(self.created_channels_total_by_prefix),
            skipped_by_headroom_by_prefix=skipped_by_headroom_by_prefix_epoch,
            skipped_by_incomplete_bundle_by_prefix=skipped_by_incomplete_bundle_by_prefix_epoch,
            skipped_by_density_cap_by_prefix=skipped_by_density_cap_by_prefix_epoch,
            block_density_by_prefix=block_density_by_prefix_epoch,
            mean_edges_per_channel=(
                float(created_channel_edges_epoch / max(1, created_channels_epoch))
                if created_channels_epoch > 0 else 0.0
            ),
            budget_scale=budget_scale,
            budget_effective=budget_effective,
            floor_per_connection=int(floor_used),
            eligible_hidden_count_by_prefix=eligible_hidden_count_by_prefix_epoch,
            new_hidden_attempts_epoch=int(self._phase1_new_hidden_attempts_epoch),
            retested_hidden_attempts_epoch=int(self._phase1_retested_hidden_attempts_epoch),
        )
        return int(created)

    def utility_based_pruning_with_protection(
        self,
        keep_ratio: float = 0.9,
        max_prune_count: Optional[int] = None,
        enable_replay_tag_protection: bool = False,
        replay_tag_quantile: float = 0.90,
        replay_tag_min: float = 0.0,
        tagged_prune_pressure: float = 0.0,
        tagged_prune_pressure_slope: float = 0.0,
        max_tagged_prune_ratio: float = 0.0,
    ) -> int:
        """
        Global pruning:
        - protect young synapses
        - optional replay-tag protection with controlled release pressure
        """
        if str(getattr(self, "pruning_policy", "global_usage")).strip().lower() == "matrix_survival_mix":
            return self._matrix_survival_pruning_with_protection(
                keep_ratio=keep_ratio,
                max_prune_count=max_prune_count,
                enable_replay_tag_protection=enable_replay_tag_protection,
                replay_tag_quantile=replay_tag_quantile,
                replay_tag_min=replay_tag_min,
                tagged_prune_pressure=tagged_prune_pressure,
                tagged_prune_pressure_slope=tagged_prune_pressure_slope,
                max_tagged_prune_ratio=max_tagged_prune_ratio,
            )

        protected_count = 0
        protected_young_total = 0
        protected_tagged_total = 0
        protected_tenured_total = 0
        tagged_active_total = 0
        released_tagged_total = 0
        tag_thresholds: List[float] = []

        candidate_utils: List[torch.Tensor] = []
        candidate_key_ids: List[torch.Tensor] = []
        candidate_rows: List[torch.Tensor] = []
        candidate_cols: List[torch.Tensor] = []
        candidate_tagged: List[torch.Tensor] = []
        tenure_excluded_by_key: Dict[str, int] = {}
        key_names: List[str] = []
        utility_by_key = self.compute_utility_scores()

        p = float(max(0.0, min(1.0, tagged_prune_pressure)))
        slope = float(max(0.0, tagged_prune_pressure_slope))
        max_ratio = float(max(0.0, min(1.0, max_tagged_prune_ratio)))
        tagged_release_ratio = float(max(0.0, min(max_ratio, slope * p)))

        q = float(max(0.0, min(1.0, replay_tag_quantile)))
        tmin = float(max(0.0, replay_tag_min))

        for key in self.weights.keys():
            active = (getattr(self, self.masks[key]).detach().cpu() > 0)
            if int(active.sum().item()) == 0:
                continue

            age = self._get_age(key).detach().cpu()
            utility = utility_by_key[key].detach().cpu().float()

            young_protected = (age < self.protection_period) & active
            protected_young_total += int(young_protected.sum().item())

            tagged_mask = torch.zeros_like(active)
            released_tagged = torch.zeros_like(active)

            if enable_replay_tag_protection:
                tag = self._get_replay_tag(key).detach().cpu().float()
                vals = tag[active]
                if int(vals.numel()) > 0:
                    thr = max(tmin, float(torch.quantile(vals, q).item()))
                    tag_thresholds.append(thr)
                    tagged_mask = (tag >= thr) & active & ~young_protected
                    tagged_active_total += int(tagged_mask.sum().item())

                    if tagged_release_ratio > 0.0 and int(tagged_mask.sum().item()) > 0:
                        n_release = int(math.ceil(tagged_release_ratio * int(tagged_mask.sum().item())))
                        n_release = max(0, min(int(tagged_mask.sum().item()), n_release))
                        if n_release > 0:
                            idx_tagged = tagged_mask.nonzero(as_tuple=False)
                            vals_tagged = tag[tagged_mask]
                            order = torch.argsort(vals_tagged, descending=False)
                            chosen = idx_tagged[order[:n_release]]
                            released_tagged[chosen[:, 0], chosen[:, 1]] = True
                            released_tagged_total += int(n_release)

            protected_tagged = tagged_mask & ~released_tagged
            protected_tagged_total += int(protected_tagged.sum().item())

            probationary_protected = self._probationary_edge_mask(key, active)
            tenured_grace_protected = self._tenured_grace_edge_mask(key, active)
            tenured_mask = (self._get_tenure_mask(key).detach().cpu() > 0) & active
            protected_tenured = tenured_mask & active & ~young_protected & ~protected_tagged
            protected_tenured_total += int(protected_tenured.sum().item())
            tenure_excluded_by_key[key] = int(protected_tenured.sum().item())
            protected = (
                young_protected
                | protected_tagged
                | probationary_protected
                | tenured_grace_protected
                | protected_tenured
            )
            prunable = active & ~protected
            protected_count += int(protected.sum().item())
            if int(prunable.sum().item()) == 0:
                continue

            idx = prunable.nonzero(as_tuple=False)
            vals = utility[prunable]
            tag_flags = tagged_mask[prunable]
            n = int(idx.size(0))
            if n <= 0:
                continue
            kid = len(key_names)
            key_names.append(key)
            candidate_utils.append(vals.reshape(-1))
            candidate_key_ids.append(torch.full((n,), kid, dtype=torch.long))
            candidate_rows.append(idx[:, 0].to(dtype=torch.long))
            candidate_cols.append(idx[:, 1].to(dtype=torch.long))
            candidate_tagged.append(tag_flags.reshape(-1).to(dtype=torch.bool))

        self.protected_from_pruning = protected_count
        if not candidate_utils:
            self.last_pruning_stats = {
                "policy": self.pruning_policy,
                "survival_score_mode": "usage_mean",
                "pruned_tagged": 0.0,
                "pruned_untagged": 0.0,
                "protected_tagged": float(protected_tagged_total),
                "protected_young": float(protected_young_total),
                "protected_tenured": float(protected_tenured_total),
                "released_tagged": float(released_tagged_total),
                "tag_threshold_mean": float(sum(tag_thresholds) / max(1, len(tag_thresholds)))
                if tag_thresholds
                else 0.0,
                "tagged_active_total": float(tagged_active_total),
                "total_prunable": 0.0,
                "tenured_total": float((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_total", 0.0)),
                "tenured_count_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_count_by_key", {})),
                "tenured_share_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_share_by_key", {})),
                "tenure_excluded_by_key": tenure_excluded_by_key,
                "prunable_count_by_key": {},
                "usage_mean_by_key": {},
                "usage_peak_by_key": {},
                "survival_mean_by_key": {},
                "survival_p95_by_key": {},
                "survival_max_by_key": {},
                "threshold_by_key": {},
                "prune_gap_mass_by_key": {},
                "positive_gap_count_by_key": {},
                "pruned_by_key": {},
                **self._empty_pruning_persistence_stats(),
            }
            return 0

        utilities = torch.cat(candidate_utils, dim=0)
        key_ids = torch.cat(candidate_key_ids, dim=0)
        rows_all = torch.cat(candidate_rows, dim=0)
        cols_all = torch.cat(candidate_cols, dim=0)
        tagged_all = torch.cat(candidate_tagged, dim=0)

        total_prunable = int(utilities.numel())
        keep_count = int(round(float(keep_ratio) * total_prunable))
        keep_count = max(0, min(total_prunable, keep_count))
        base_prune_count = max(0, total_prunable - keep_count)
        effective_max_prune_count = None
        prune_count = int(base_prune_count)
        if max_prune_count is not None:
            effective_max_prune_count = max(0, min(total_prunable, int(max_prune_count)))
            prune_count = min(prune_count, effective_max_prune_count)
        if prune_count <= 0:
            self.last_pruning_stats = {
                "policy": self.pruning_policy,
                "survival_score_mode": "usage_mean",
                "base_prune_count": float(base_prune_count),
                "effective_prune_count": float(prune_count),
                "max_prune_count": (
                    float(effective_max_prune_count) if effective_max_prune_count is not None else None
                ),
                "pruned_tagged": 0.0,
                "pruned_untagged": 0.0,
                "protected_tagged": float(protected_tagged_total),
                "protected_young": float(protected_young_total),
                "protected_tenured": float(protected_tenured_total),
                "released_tagged": float(released_tagged_total),
                "tag_threshold_mean": float(sum(tag_thresholds) / max(1, len(tag_thresholds)))
                if tag_thresholds
                else 0.0,
                "tagged_active_total": float(tagged_active_total),
                "total_prunable": float(total_prunable),
                "tenured_total": float((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_total", 0.0)),
                "tenured_count_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_count_by_key", {})),
                "tenured_share_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_share_by_key", {})),
                "tenure_excluded_by_key": tenure_excluded_by_key,
                "prunable_count_by_key": {},
                "usage_mean_by_key": {},
                "usage_peak_by_key": {},
                "survival_mean_by_key": {},
                "survival_p95_by_key": {},
                "survival_max_by_key": {},
                "threshold_by_key": {},
                "prune_gap_mass_by_key": {},
                "positive_gap_count_by_key": {},
                "pruned_by_key": {},
                **self._empty_pruning_persistence_stats(),
            }
            return 0

        # Stable order preserves legacy tie-breaks:
        # insertion order of keys + row-major order from nonzero().
        try:
            prune_order = torch.argsort(utilities, descending=False, stable=True)
        except TypeError:
            prune_order = torch.argsort(utilities, descending=False)
        prune_sel = prune_order[:prune_count]
        pr_key_ids = key_ids[prune_sel]
        pr_rows = rows_all[prune_sel]
        pr_cols = cols_all[prune_sel]
        pr_tagged = tagged_all[prune_sel]

        pending_rows: Dict[str, List[int]] = {}
        pending_cols: Dict[str, List[int]] = {}
        pruned_tagged = int(pr_tagged.sum().item())
        pruned_untagged = int(prune_count - pruned_tagged)
        for kid, key in enumerate(key_names):
            mask_k = (pr_key_ids == kid)
            if not bool(mask_k.any()):
                continue
            pending_rows[key] = pr_rows[mask_k].tolist()
            pending_cols[key] = pr_cols[mask_k].tolist()

        pruning_persistence_stats = self._compute_pruning_persistence_stats(
            pending_rows=pending_rows,
            pending_cols=pending_cols,
        )
        with torch.no_grad():
            for key, rows in pending_rows.items():
                if not rows:
                    continue
                cols = pending_cols[key]
                dev = self.weights[key].device
                rows_t = torch.tensor(rows, device=dev, dtype=torch.long)
                cols_t = torch.tensor(cols, device=dev, dtype=torch.long)

                mask = getattr(self, self.masks[key])
                mask[rows_t, cols_t] = 0.0
                self.weights[key].data[rows_t, cols_t] = 0.0

                self._get_replay_tag(key)[rows_t, cols_t] = 0.0
                self._get_usage(key)[rows_t, cols_t] = 0.0
                self._get_usage_peak(key)[rows_t, cols_t] = 0.0
                self._get_age(key)[rows_t, cols_t] = 0.0
                self._get_tenure_mask(key)[rows_t, cols_t] = 0.0
                self._get_tenure_score_ema(key)[rows_t, cols_t] = 0.0
                self._get_tenure_streak(key)[rows_t, cols_t] = 0

        self.pruning_count += int(prune_count)
        self.last_pruning_stats = {
            "policy": self.pruning_policy,
            "survival_score_mode": "usage_mean",
            "base_prune_count": float(base_prune_count),
            "effective_prune_count": float(prune_count),
            "max_prune_count": (
                float(effective_max_prune_count) if effective_max_prune_count is not None else None
            ),
            "pruned_tagged": float(pruned_tagged),
            "pruned_untagged": float(pruned_untagged),
            "protected_tagged": float(protected_tagged_total),
            "protected_young": float(protected_young_total),
            "protected_tenured": float(protected_tenured_total),
            "released_tagged": float(released_tagged_total),
            "tag_threshold_mean": float(sum(tag_thresholds) / max(1, len(tag_thresholds)))
            if tag_thresholds
            else 0.0,
            "tagged_active_total": float(tagged_active_total),
            "total_prunable": float(total_prunable),
            "tenured_total": float((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_total", 0.0)),
            "tenured_count_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_count_by_key", {})),
            "tenured_share_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_share_by_key", {})),
            "tenure_excluded_by_key": tenure_excluded_by_key,
            "prunable_count_by_key": {},
            "usage_mean_by_key": {},
            "usage_peak_by_key": {},
            "survival_mean_by_key": {},
            "survival_p95_by_key": {},
            "survival_max_by_key": {},
            "threshold_by_key": {},
            "prune_gap_mass_by_key": {},
            "positive_gap_count_by_key": {},
            "pruned_by_key": {},
            **pruning_persistence_stats,
        }
        return int(prune_count)

    def relative_median_pruning_with_protection(
        self,
        k: float = 0.01,
        safety_cap_fraction: float = 0.01,
        max_prune_count: Optional[int] = None,
        enable_replay_tag_protection: bool = False,
        replay_tag_quantile: float = 0.90,
        replay_tag_min: float = 0.0,
        tagged_prune_pressure: float = 0.0,
        tagged_prune_pressure_slope: float = 0.0,
        max_tagged_prune_ratio: float = 0.0,
    ) -> int:
        """
        Pruning by per-matrix relative median threshold:
          threshold_key = k * median(usage_ema on mature/prunable active)
        with per-matrix safety cap (fraction of prunable synapses).
        """
        k = float(max(0.0, k))
        safety_cap_fraction = float(max(0.0, min(1.0, safety_cap_fraction)))

        protected_count = 0
        protected_young_total = 0
        protected_tagged_total = 0
        protected_tenured_total = 0
        tagged_active_total = 0
        released_tagged_total = 0
        tag_thresholds: List[float] = []

        p = float(max(0.0, min(1.0, tagged_prune_pressure)))
        slope = float(max(0.0, tagged_prune_pressure_slope))
        max_ratio = float(max(0.0, min(1.0, max_tagged_prune_ratio)))
        tagged_release_ratio = float(max(0.0, min(max_ratio, slope * p)))

        q = float(max(0.0, min(1.0, replay_tag_quantile)))
        tmin = float(max(0.0, replay_tag_min))

        utility_by_key = self.compute_utility_scores()
        total_prunable = 0
        median_usage_by_key: Dict[str, float] = {}
        median_prune_threshold_by_key: Dict[str, float] = {}
        median_prune_candidate_count_by_key: Dict[str, int] = {}
        median_prune_effective_count_by_key: Dict[str, int] = {}
        prunable_count_by_key: Dict[str, int] = {}
        pending_rows: Dict[str, List[int]] = {}
        pending_cols: Dict[str, List[int]] = {}
        pending_utils: Dict[str, List[float]] = {}
        pending_tagged: Dict[str, List[bool]] = {}
        tenure_excluded_by_key: Dict[str, int] = {}

        for key in self.weights.keys():
            active = (getattr(self, self.masks[key]).detach().cpu() > 0)
            if int(active.sum().item()) == 0:
                continue

            age = self._get_age(key).detach().cpu()
            utility = utility_by_key[key].detach().cpu().float().clamp_min(0.0)

            young_protected = (age < self.protection_period) & active
            protected_young_total += int(young_protected.sum().item())

            tagged_mask = torch.zeros_like(active)
            released_tagged = torch.zeros_like(active)
            if enable_replay_tag_protection:
                tag = self._get_replay_tag(key).detach().cpu().float()
                vals = tag[active]
                if int(vals.numel()) > 0:
                    thr = max(tmin, float(torch.quantile(vals, q).item()))
                    tag_thresholds.append(thr)
                    tagged_mask = (tag >= thr) & active & ~young_protected
                    tagged_active_total += int(tagged_mask.sum().item())
                    if tagged_release_ratio > 0.0 and int(tagged_mask.sum().item()) > 0:
                        n_release = int(math.ceil(tagged_release_ratio * int(tagged_mask.sum().item())))
                        n_release = max(0, min(int(tagged_mask.sum().item()), n_release))
                        if n_release > 0:
                            idx_tagged = tagged_mask.nonzero(as_tuple=False)
                            vals_tagged = tag[tagged_mask]
                            order = torch.argsort(vals_tagged, descending=False)
                            chosen = idx_tagged[order[:n_release]]
                            released_tagged[chosen[:, 0], chosen[:, 1]] = True
                            released_tagged_total += int(n_release)

            protected_tagged = tagged_mask & ~released_tagged
            protected_tagged_total += int(protected_tagged.sum().item())

            probationary_protected = self._probationary_edge_mask(key, active)
            tenured_grace_protected = self._tenured_grace_edge_mask(key, active)
            tenured_mask = (self._get_tenure_mask(key).detach().cpu() > 0) & active
            protected_tenured = tenured_mask & active & ~young_protected & ~protected_tagged
            protected_tenured_total += int(protected_tenured.sum().item())
            tenure_excluded_by_key[key] = int(protected_tenured.sum().item())
            protected = (
                young_protected
                | protected_tagged
                | probationary_protected
                | tenured_grace_protected
                | protected_tenured
            )
            prunable = active & ~protected
            protected_count += int(protected.sum().item())
            prunable_count = int(prunable.sum().item())
            prunable_count_by_key[key] = int(prunable_count)
            total_prunable += int(prunable_count)
            if prunable_count <= 0:
                median_usage_by_key[key] = 0.0
                median_prune_threshold_by_key[key] = 0.0
                median_prune_candidate_count_by_key[key] = 0
                median_prune_effective_count_by_key[key] = 0
                continue

            prunable_usage = utility[prunable]
            median_usage = float(torch.median(prunable_usage).item()) if int(prunable_usage.numel()) > 0 else 0.0
            threshold = float(k * median_usage)
            candidate_mask = prunable & (utility < threshold)
            candidate_count = int(candidate_mask.sum().item())
            safety_cap = int(math.floor(float(safety_cap_fraction) * float(prunable_count)))
            effective_count = int(min(candidate_count, max(0, safety_cap)))

            median_usage_by_key[key] = float(median_usage)
            median_prune_threshold_by_key[key] = float(threshold)
            median_prune_candidate_count_by_key[key] = int(candidate_count)
            median_prune_effective_count_by_key[key] = int(effective_count)
            if effective_count <= 0:
                continue

            idx = candidate_mask.nonzero(as_tuple=False)
            vals = utility[candidate_mask]
            try:
                order = torch.argsort(vals, descending=False, stable=True)
            except TypeError:
                order = torch.argsort(vals, descending=False)
            chosen = idx[order[:effective_count]]
            chosen_vals = vals[order[:effective_count]]
            chosen_tags = tagged_mask[chosen[:, 0], chosen[:, 1]].to(dtype=torch.bool)

            pending_rows[key] = chosen[:, 0].tolist()
            pending_cols[key] = chosen[:, 1].tolist()
            pending_utils[key] = [float(v) for v in chosen_vals.tolist()]
            pending_tagged[key] = [bool(v) for v in chosen_tags.tolist()]

        self.protected_from_pruning = int(protected_count)
        candidate_total_epoch = int(sum(int(v) for v in median_prune_candidate_count_by_key.values()))

        selected_records: List[Tuple[float, str, int, int, bool]] = []
        for key, rows in pending_rows.items():
            cols = pending_cols.get(key, [])
            vals = pending_utils.get(key, [])
            tags = pending_tagged.get(key, [])
            n = min(len(rows), len(cols), len(vals), len(tags))
            for i in range(n):
                selected_records.append((float(vals[i]), str(key), int(rows[i]), int(cols[i]), bool(tags[i])))

        base_prune_count = int(len(selected_records))
        effective_max_prune_count = None
        if max_prune_count is not None:
            effective_max_prune_count = max(0, min(int(base_prune_count), int(max_prune_count)))
            if base_prune_count > effective_max_prune_count:
                selected_records.sort(key=lambda x: x[0])
                selected_records = selected_records[:effective_max_prune_count]

        prune_count = int(len(selected_records))
        if prune_count <= 0:
            self.last_pruning_stats = {
                "policy": "relative_median_usage",
                "base_prune_count": float(base_prune_count),
                "effective_prune_count": 0.0,
                "max_prune_count": (
                    float(effective_max_prune_count) if effective_max_prune_count is not None else None
                ),
                "pruned_tagged": 0.0,
                "pruned_untagged": 0.0,
                "protected_tagged": float(protected_tagged_total),
                "protected_young": float(protected_young_total),
                "protected_tenured": float(protected_tenured_total),
                "released_tagged": float(released_tagged_total),
                "tag_threshold_mean": float(sum(tag_thresholds) / max(1, len(tag_thresholds)))
                if tag_thresholds else 0.0,
                "tagged_active_total": float(tagged_active_total),
                "total_prunable": float(total_prunable),
                "tenured_total": float((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_total", 0.0)),
                "tenured_count_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_count_by_key", {})),
                "tenured_share_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_share_by_key", {})),
                "tenure_excluded_by_key": tenure_excluded_by_key,
                "prunable_count_by_key": {str(k): int(v) for k, v in prunable_count_by_key.items()},
                "median_usage_by_key": {str(k): float(v) for k, v in median_usage_by_key.items()},
                "median_prune_threshold_by_key": {str(k): float(v) for k, v in median_prune_threshold_by_key.items()},
                "median_prune_candidate_count_by_key": {str(k): int(v) for k, v in median_prune_candidate_count_by_key.items()},
                "median_prune_effective_count_by_key": {str(k): int(v) for k, v in median_prune_effective_count_by_key.items()},
                "median_prune_candidate_count_epoch": float(candidate_total_epoch),
                "median_prune_effective_count_epoch": 0.0,
                **self._empty_pruning_persistence_stats(),
            }
            return 0

        pending_rows_final: Dict[str, List[int]] = {}
        pending_cols_final: Dict[str, List[int]] = {}
        pruned_tagged = 0
        for _, key, row_i, col_i, tagged in selected_records:
            pending_rows_final.setdefault(key, []).append(int(row_i))
            pending_cols_final.setdefault(key, []).append(int(col_i))
            if tagged:
                pruned_tagged += 1
        pruned_untagged = int(prune_count - pruned_tagged)

        pruning_persistence_stats = self._compute_pruning_persistence_stats(
            pending_rows=pending_rows_final,
            pending_cols=pending_cols_final,
        )

        with torch.no_grad():
            for key, rows in pending_rows_final.items():
                if not rows:
                    continue
                cols = pending_cols_final[key]
                dev = self.weights[key].device
                rows_t = torch.tensor(rows, device=dev, dtype=torch.long)
                cols_t = torch.tensor(cols, device=dev, dtype=torch.long)
                mask = getattr(self, self.masks[key])
                mask[rows_t, cols_t] = 0.0
                self.weights[key].data[rows_t, cols_t] = 0.0
                self._get_replay_tag(key)[rows_t, cols_t] = 0.0
                self._get_usage(key)[rows_t, cols_t] = 0.0
                self._get_usage_peak(key)[rows_t, cols_t] = 0.0
                self._get_age(key)[rows_t, cols_t] = 0.0
                self._get_tenure_mask(key)[rows_t, cols_t] = 0.0
                self._get_tenure_score_ema(key)[rows_t, cols_t] = 0.0
                self._get_tenure_streak(key)[rows_t, cols_t] = 0

        self.pruning_count += int(prune_count)
        self.last_pruning_stats = {
            "policy": "relative_median_usage",
            "base_prune_count": float(base_prune_count),
            "effective_prune_count": float(prune_count),
            "max_prune_count": (
                float(effective_max_prune_count) if effective_max_prune_count is not None else None
            ),
            "pruned_tagged": float(pruned_tagged),
            "pruned_untagged": float(pruned_untagged),
            "protected_tagged": float(protected_tagged_total),
            "protected_young": float(protected_young_total),
            "protected_tenured": float(protected_tenured_total),
            "released_tagged": float(released_tagged_total),
            "tag_threshold_mean": float(sum(tag_thresholds) / max(1, len(tag_thresholds)))
            if tag_thresholds else 0.0,
            "tagged_active_total": float(tagged_active_total),
            "total_prunable": float(total_prunable),
            "tenured_total": float((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_total", 0.0)),
            "tenured_count_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_count_by_key", {})),
            "tenured_share_by_key": dict((getattr(self, "last_tenure_stats", {}) or {}).get("tenured_share_by_key", {})),
            "tenure_excluded_by_key": tenure_excluded_by_key,
            "prunable_count_by_key": {str(k): int(v) for k, v in prunable_count_by_key.items()},
            "median_usage_by_key": {str(k): float(v) for k, v in median_usage_by_key.items()},
            "median_prune_threshold_by_key": {str(k): float(v) for k, v in median_prune_threshold_by_key.items()},
            "median_prune_candidate_count_by_key": {str(k): int(v) for k, v in median_prune_candidate_count_by_key.items()},
            "median_prune_effective_count_by_key": {str(k): int(v) for k, v in median_prune_effective_count_by_key.items()},
            "median_prune_candidate_count_epoch": float(candidate_total_epoch),
            "median_prune_effective_count_epoch": float(prune_count),
            **pruning_persistence_stats,
        }
        return int(prune_count)

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    def increment_synapse_ages(self) -> None:
        for key in self.weights.keys():
            mask = getattr(self, self.masks[key])
            age = self._get_age(key)
            age.add_(mask)

    def _empty_creation_stats(self) -> Dict[str, Any]:
        return {
            "policy": self.creation_policy,
            "score_normalization": self.creation_score_normalization,
            "budget_requested": 0.0,
            "budget_used": 0.0,
            "candidate_total": 0.0,
            "floor_per_connection": float(self.creation_floor_per_connection),
            "ordered_keys": [],
            "candidate_count_by_key": {},
            "eligible_count_before_quality_by_key": {},
            "eligible_count_by_key": {},
            "score_mean_by_key": {},
            "score_p95_by_key": {},
            "score_max_by_key": {},
            "active_ref_by_key": {},
            "threshold_by_key": {},
            "quality_threshold_by_key": {},
            "gap_mass_by_key": {},
            "relative_gap_mass_by_key": {},
            "priority_mass_by_key": {},
            "frustration_mass_by_key": {},
            "relative_frustration_mass_by_key": {},
            "coherence_mean_by_key": {},
            "coherence_p50_by_key": {},
            "coherence_p75_by_key": {},
            "coherence_p90_by_key": {},
            "interference_mean_by_key": {},
            "interference_p95_by_key": {},
            "interference_max_by_key": {},
            "support_mean_by_key": {},
            "support_p75_by_key": {},
            "support_p90_by_key": {},
            "support_scale_by_key": {},
            "scarcity_mean_by_key": {},
            "scarcity_p95_by_key": {},
            "scarcity_max_by_key": {},
            "need_score_mean_by_key": {},
            "need_score_p95_by_key": {},
            "need_score_max_by_key": {},
            "need_interference_mean_by_key": {},
            "need_interference_p95_by_key": {},
            "need_by_key": {},
            "need_ema_by_key": {},
            "priority_positive_count_by_key": {},
            "selected_priority_mean_by_key": {},
            "selected_priority_max_by_key": {},
            "created_by_key": {},
            "created_floor_by_key": {},
            "created_global_by_key": {},
            "group_budget_by_key": {},
            "group_budget_by_role_group": {},
            "candidate_count_by_role_group": {},
            "eligible_count_by_role_group": {},
            "gap_mass_by_role_group": {},
            "relative_gap_mass_by_role_group": {},
            "priority_mass_by_role_group": {},
            "frustration_mass_by_role_group": {},
            "created_by_role_group": {},
            "group_share_target_by_role_group": {},
            "group_share_ema_by_role_group": {},
            "group_share_applied_by_role_group": {},
            "role_need_by_group": {},
            "role_need_ema_by_group": {},
            "created_channels_epoch": 0.0,
            "created_channel_edges_epoch": 0.0,
            "created_channels_by_prefix": {},
            "selected_hidden_mean_degree_by_prefix": {},
            "selected_hidden_p75_degree_by_prefix": {},
            "selected_hidden_mean_fire_rate_by_prefix": {},
            "selected_hidden_mean_usage_by_prefix": {},
            "created_channels_total": float(self.created_channels_total),
            "created_channel_edges_total": float(self.created_channel_edges_total),
            "created_channels_total_by_prefix": {
                str(key): int(value) for key, value in self.created_channels_total_by_prefix.items()
            },
            "tenured_channels_total": float(self.tenured_channels_total),
            "rejected_channels_total": float(self.rejected_channels_total),
            "mean_edges_per_channel": float(
                self.created_channel_edges_total / max(1, self.created_channels_total)
            ) if self.created_channels_total > 0 else 0.0,
            "avg_edges_per_channel_total": float(
                self.created_channel_edges_total / max(1, self.created_channels_total)
            ) if self.created_channels_total > 0 else 0.0,
            "skipped_by_headroom_by_prefix": {},
            "skipped_by_incomplete_bundle_by_prefix": {},
            "skipped_by_density_cap_by_prefix": {},
            "block_density_by_prefix": {},
            "final_block_density_by_prefix": {},
            "budget_scale": 1.0,
            "budget_effective": 0.0,
            "new_hidden_attempts_epoch": 0.0,
            "retested_hidden_attempts_epoch": 0.0,
            "eligible_hidden_count_by_prefix": {},
            "explored_hidden_count_by_prefix": {},
            "coverage_by_prefix": {},
            "coverage_global": 0.0,
        }

    def reset_creation_tracking(self) -> None:
        for key in self.weights.keys():
            self._get_creation_score(key).zero_()
            self._get_creation_direction(key).zero_()
        self._creation_group_share_ema = {"enc": None, "dec": None}
        self._creation_role_need_ema = {"enc": None, "dec": None, "other": None}
        self._creation_need_ema = {key: None for key in self.weights.keys()}
        self._pruning_creation_need_ema = {key: None for key in self.weights.keys()}
        self.pruned_persistent_low_total = 0
        self.pruned_short_only_low_total = 0
        self._dead_count_totals_by_label = {
            dead_label: 0 for dead_label, _ in self._pruning_dead_thresholds
        }
        self._dead_pruned_observed_total = 0
        self._median_dead_count_k001_total = 0
        self._median_dead_count_k002_total = 0
        self._median_dead_pruned_observed_total = 0
        self.created_channels_total = 0
        self.created_channel_edges_total = 0
        self.created_channels_total_by_prefix = {}
        self.tenured_channels_total = 0
        self.rejected_channels_total = 0
        self.local_reinforce_channels_total = 0
        self.local_reinforce_edges_total = 0
        self._channel_activity_tenured_sum = 0.0
        self._channel_activity_tenured_count = 0
        self._channel_activity_rejected_sum = 0.0
        self._channel_activity_rejected_count = 0
        self._next_probationary_channel_id = 1
        self._probationary_channels = {}
        self._channel_probe_threshold_ema = None
        self._phase1_exploration_registry = {}
        self._phase1_eligible_hidden_ids_by_prefix = {}
        self._phase1_new_hidden_attempts_epoch = 0
        self._phase1_retested_hidden_attempts_epoch = 0
        self._phase1_evaluated_channels_total_current_level = 0
        self._phase1_tenured_channels_total_current_level = 0
        self._phase1_rejected_channels_total_current_level = 0
        self._created_channels_total_current_level = 0
        self._tenured_grace_channels = {}
        self._local_reinforce_queue = {}
        self._epoch_runtime_tag = -1
        self._epoch_pathway_dropout_active = False
        self._epoch_pathway_dropout_rate_effective = 0.0
        self._epoch_pathway_dropout_candidate_channels_by_prefix = {}
        self._epoch_pathway_dropout_dropped_channels_by_prefix = {}
        self._epoch_pathway_dropout_protected_channels_by_prefix = {}
        self._epoch_local_reinforce_channels = {}
        self._epoch_local_reinforce_edges_up_by_prefix = {}
        self._epoch_local_reinforce_edges_down_by_prefix = {}
        self._epoch_local_reinforce_edges = 0
        self._epoch_local_reinforce_budget_used = 0
        self.last_channel_probe_stats = self._empty_channel_probe_stats()
        self.last_creation_stats = self._empty_creation_stats()
        self.last_tenure_stats = {
            "enabled": bool(self.tenure_enabled),
            "score_source": "survival_sqrt_mean_peak",
            "new_tenures_total": 0.0,
            "tenured_total": 0.0,
            "tenured_share_global": 0.0,
            "active_count_by_key": {},
            "mature_count_by_key": {},
            "tenured_count_by_key": {},
            "tenured_share_by_key": {},
            "tenure_threshold_by_key": {},
            "tenure_norm_anchor_by_key": {},
            "eligible_to_tenure_by_key": {},
            "new_tenures_by_key": {},
            "max_tenured_by_key": {},
        }

    def reset_tracking(self) -> None:
        # Keep replay tags across levels by design, as in 3B1.
        for key in self.weights.keys():
            self._get_usage(key).zero_()
            self._get_usage_long(key).zero_()
            self._get_usage_peak(key).zero_()
            self._get_contribution(key).zero_()
            self._get_tenure_mask(key).zero_()
            self._get_tenure_score_ema(key).zero_()
            self._get_tenure_streak(key).zero_()
            if str(key).endswith("_FFN_up"):
                prefix = str(key).rsplit("_FFN_up", 1)[0]
                self._get_hidden_usage_mean(prefix).zero_()
                self._get_hidden_usage_peak(prefix).zero_()
                self._get_hidden_fire_rate(prefix).zero_()
        self.reset_creation_tracking()
        self.last_ffn_hidden_stats = {"by_prefix": {}, "by_role_group": {}}

    def get_connectivity_stats(self) -> SynapseStats:
        total_possible = 0
        total_active = 0
        by_conn: Dict[str, Dict[str, float]] = {}
        for key in self.weights.keys():
            mask = getattr(self, self.masks[key])
            n_possible = int(mask.numel())
            n_active = int(mask.sum().item())
            total_possible += n_possible
            total_active += n_active
            by_conn[key] = {
                "possible": float(n_possible),
                "active": float(n_active),
                "density": float(n_active / max(1, n_possible)),
            }
        return SynapseStats(
            total_possible=total_possible,
            total_active=total_active,
            overall_density=float(total_active / max(1, total_possible)),
            energy_consumed=float(self.compute_energy()),
            energy_budget=float(self.energy_budget),
            by_connection=by_conn,
        )

    def count_parameters(self) -> int:
        """
        Count effective parameters:
        - sparse matrices: active synapses only
        - dense modules (embeddings/norms/output): full parameter count
        """
        sparse_active = 0
        for key in self.weights.keys():
            sparse_active += int(getattr(self, self.masks[key]).sum().item())

        dense_params = 0
        for name, p in self.named_parameters():
            if name.startswith("weights."):
                continue
            dense_params += int(p.numel())
        return int(sparse_active + dense_params)


__all__ = [
    "SynapseStats",
    "SparseTransformerSeq2Seq",
]
