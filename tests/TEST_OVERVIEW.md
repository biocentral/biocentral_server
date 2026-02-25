# Test Overview

## Summary

| Metric | Count |
|--------|-------|
| **Total Test Files** | 18 |
| **Total Test Cases** | ~95 |
| **Unit Tests** | 43 |
| **Integration Tests** | 28 |
| **Property-Based Tests** | 16 |
| **Performance Tests** | 18 |

### Notable Coverage Areas

- **Embeddings**: Complete endpoint coverage, FixedEmbedder validation, throughput benchmarks
- **Predictions**: Model metadata, request validation, task completion flows
- **Projections**: PCA/UMAP/t-SNE configuration and execution
- **Custom Models**: Training and inference lifecycle, config validation
- **Oracles**: Batch invariance, masking robustness, determinism, output validity

---

## Unit Tests

Unit tests verify isolated components with mocked dependencies.

### test_embeddings.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_common_embedders_returns_list` | Common embedders endpoint returns list | None | `status_code == 200`, list not empty |
| `test_common_embedders_contains_expected_models` | All CommonEmbedder enum values returned | None | Response matches enum values |
| `test_common_embedders_includes_prot_t5` | ProtT5 embedder available | None | `"Rostlab/prot_t5_xl_uniref50"` in response |
| `test_common_embedders_includes_esm2` | ESM2 embedders available | None | At least one ESM2 model present |
| `test_common_embedders_includes_baseline_models` | Baseline embedders available | None | `"one_hot_encoding"` and `"blosum62"` present |
| `test_embed_valid_request` | Valid embed request returns task_id | TaskManager, UserManager, MetricsCollector, RateLimiter | `status_code == 200`, `task_id` in response |
| `test_embed_empty_sequences` | Empty sequences rejected | RateLimiter | `status_code == 422` |
| `test_embed_missing_embedder_name` | Missing embedder name rejected | RateLimiter | `status_code == 422` |
| `test_get_missing_embeddings_valid` | Valid missing embeddings request | EmbeddingDatabaseFactory, RateLimiter | `status_code == 200`, `missing` in response |
| `test_get_missing_embeddings_invalid_json` | Invalid JSON rejected | RateLimiter | `status_code == 422` |
| `test_get_missing_embeddings_sequences_not_dict` | Non-dict sequences rejected | RateLimiter | `status_code == 422` |
| `test_add_embeddings_valid` | Valid HDF5 embeddings accepted | EmbeddingDatabaseFactory, RateLimiter | `status_code in [200, 201]` |
| `test_add_embeddings_invalid_base64` | Invalid base64 rejected | RateLimiter | `status_code in [400, 422, 500]` |

### test_predict.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_model_metadata_returns_available_models` | Model metadata endpoint returns models | get_metadata_for_all_models, RateLimiter | `status_code == 200`, metadata list populated |
| `test_model_metadata_empty_when_no_models` | Empty models raises validation error | get_metadata_for_all_models, RateLimiter | `ValidationError` with `"too_short"` |
| `test_predict_valid_request` | Valid prediction request returns task_id | TaskManager, UserManager, filter_models, RateLimiter | `status_code == 200`, `task_id` in response |
| `test_predict_unknown_model` | Unknown model name rejected | get_metadata_for_all_models, RateLimiter | `status_code == 422` |
| `test_predict_empty_model_names` | Empty model names list rejected | RateLimiter | `status_code == 422` |
| `test_predict_empty_sequences` | Empty sequence data rejected | RateLimiter | `status_code == 422` |
| `test_predict_sequence_too_short` | Sequence below min length rejected | RateLimiter | `status_code == 422` |
| `test_predict_sequence_too_long` | Sequence above max length rejected | RateLimiter | `status_code == 422` |

### test_projection.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_projection_config_returns_methods` | Projection config endpoint works | REDUCERS, ProtSpaceDimensionReductionConfig, RateLimiter | `status_code == 200`, `projection_config` present |
| `test_projection_config_includes_all_methods` | All methods have parameters | REDUCERS, ProtSpaceDimensionReductionConfig, RateLimiter | `parameters_by_method` called for each method |
| `test_project_valid_request` | Valid projection request returns task_id | TaskManager, UserManager, REDUCERS, RateLimiter | `status_code == 200`, `task_id` in response |
| `test_project_unknown_method` | Unknown projection method rejected | REDUCERS, RateLimiter | `status_code == 400`, `"Unknown method"` in detail |
| `test_project_missing_required_fields` | Missing fields rejected | RateLimiter | `status_code == 422` |
| `test_project_with_custom_config` | Custom config parameters passed through | TaskManager, UserManager, REDUCERS, convert_config, RateLimiter | Config captured matches input |

### test_custom_models.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_config_options_valid_protocol` | Valid protocol returns config options | Configurator, BiotrainerTask, Protocol, RateLimiter | `status_code == 200`, options list has 2 items |
| `test_config_options_invalid_protocol` | Invalid protocol returns 400 | Protocol, RateLimiter | `status_code == 400`, `"Invalid protocol"` in detail |
| `test_verify_config_valid` | Valid config verification returns no error | verify_biotrainer_config, RateLimiter | `status_code == 200`, `error == ""` |
| `test_verify_config_invalid` | Invalid config returns error message | verify_biotrainer_config, RateLimiter | `status_code == 200`, error message present |
| `test_verify_config_empty_dict` | Empty config dict rejected | RateLimiter | `status_code == 422` |
| `test_start_training_valid` | Valid training request returns task_id | TaskManager, FileManager, UserManager, verify_biotrainer_config, RateLimiter | `status_code == 200`, `task_id` in response |
| `test_start_training_invalid_config` | Invalid config rejected | verify_biotrainer_config, RateLimiter | `status_code == 400` |
| `test_start_training_empty_training_data` | Empty training data rejected | RateLimiter | `status_code == 422` |
| `test_model_files_found` | Valid model hash returns files | FileManager, UserManager, RateLimiter | `status_code == 200`, file keys present |
| `test_model_files_not_found` | Invalid model hash returns 404 | FileManager, UserManager, RateLimiter | `status_code == 404` |
| `test_start_inference_valid` | Valid inference request returns task_id | TaskManager, FileManager, UserManager, RateLimiter | `status_code == 200`, `task_id` in response |
| `test_start_inference_model_not_found` | Invalid model hash returns 404 | FileManager, UserManager, RateLimiter | `status_code == 404` |
| `test_start_inference_empty_sequences` | Empty sequences rejected | RateLimiter | `status_code == 422` |

### test_fixed_embedder.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_same_sequence_same_embedding` | Same sequence produces identical embeddings | None | Arrays equal |
| `test_same_sequence_different_instances` | Different instances, same config, identical results | None | Arrays equal |
| `test_different_sequences_different_embeddings` | Different sequences produce different embeddings | None | Arrays not close |
| `test_different_seeds_different_embeddings` | Different seeds produce different embeddings | None | Arrays not close |
| `test_pooled_embedding_determinism` | Pooled embeddings are deterministic | None | Arrays equal |
| `test_determinism_across_batch_methods` | Batch matches individual embed calls | None | Each array pair equal |
| `test_model_dimensions` | Each model has correct embedding dimension | None | Shape matches expected |
| `test_custom_dimension` | Custom dimension overrides default | None | Shape matches custom dim |
| `test_pooled_dimension` | Pooled embeddings have 1D shape | None | Shape is `(dim,)` |
| `test_empty_sequence` | Empty sequence returns empty embedding | None | Shape `(0, dim)` |
| `test_single_residue` | Single residue works | None | Shape `(1, dim)`, no NaN |
| `test_very_long_sequence` | Long sequences work | None | Correct shape, no NaN |
| `test_unknown_amino_acid` | Unknown AA (X) handled | None | Shape correct, no NaN |
| `test_all_amino_acids` | All 20 standard AAs work | None | Shape `(20, dim)` |
| `test_lowercase_handling` | Lowercase handled same as uppercase | None | Arrays equal |
| `test_batch_embedding` | Batch embedding correct shapes | None | Length matches, shapes correct |
| `test_batch_pooled` | Batch pooled embeddings work | None | All shapes `(dim,)` |
| `test_dict_embedding` | Dict-based embedding works | None | Keys match, shapes correct |
| `test_get_embedder_creates_instance` | Registry creates new instance | Registry cleared | Not None, model_name correct |
| `test_get_embedder_reuses_instance` | Registry reuses instance | Registry cleared | Same object |
| `test_clear_registry` | Clear removes cached instances | Registry cleared | Different objects |
| `test_embedding_not_all_zeros` | Embeddings not all zeros | None | Not all close to 0 |
| `test_embedding_no_nan` | No NaN values | None | No NaN in embeddings |
| `test_embedding_no_inf` | No infinite values | None | No Inf in embeddings |
| `test_embedding_reasonable_magnitude` | Values within bounds | None | Max abs < 100 |
| `test_embedding_dtype` | Embeddings are float32 | None | dtype == float32 |
| `test_different_positions_have_different_embeddings` | Position embeddings differ | None | Adjacent positions not close |

---

## Integration Tests

Integration tests verify end-to-end flows against a running server.

### test_embed_flow.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_common_embedders_returns_list` | Embedders endpoint returns list via live server | Live server | `status_code == 200`, list not empty |
| `test_common_embedders_includes_baseline_models` | Baseline embedders available | Live server | `one_hot_encoding`, `blosum62` present |
| `test_common_embedders_response_is_consistent` | Multiple calls return same list | Live server | Responses equal |
| `test_embed_empty_sequences_rejected` | Empty sequences rejected | Live server | `status_code == 422` |
| `test_embed_missing_embedder_name_rejected` | Missing embedder name rejected | Live server | `status_code == 422` |
| `test_embed_and_wait_for_completion` | Full embed flow completes, cache populated | Live server, Redis | Task finishes, embeddings cached |
| `test_embed_with_different_embedders` | Multiple embedders work | Live server | Both `one_hot_encoding` and `blosum62` complete |

### test_project_flow.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_get_projection_config` | Projection config endpoint works | Live server | `status_code == 200`, config dict not empty |
| `test_project_task_completes` | PCA projection task completes | Live server, cached embeddings | Task status `FINISHED` |
| `test_project_invalid_method_rejected` | Invalid projection method rejected | Live server | `status_code == 400` |
| `test_project_empty_sequences_rejected` | Empty sequences rejected | Live server | `status_code in [400, 422]` |
| `test_complete_projection_flow` | Full projection flow completes | Live server | Task status `FINISHED` |
| `test_umap_projection_flow` | UMAP projection completes | Live server | Task completes (pass or fail acceptable) |
| `test_projection_with_real_world_collection` | Projection with diverse sequences | Live server | Task completes |

### test_predict_flow.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_get_model_metadata` | Model metadata endpoint works | Live server | `status_code == 200`, metadata list present |
| `test_model_metadata_structure` | Metadata has expected structure | Live server | `name` key present in each model |
| `test_model_metadata_consistent` | Metadata consistent across calls | Live server | Responses equal |
| `test_predict_invalid_model_rejected` | Invalid model name rejected | Live server | `status_code == 422` |
| `test_predict_empty_sequences_rejected` | Empty sequences rejected | Live server | `status_code == 422` |
| `test_predict_short_sequence_rejected` | Short sequences rejected | Live server | `status_code == 422` |
| `test_predict_empty_model_names_rejected` | Empty model names rejected | Live server | `status_code == 422` |
| `test_predict_task_completes` | Full predict flow completes | Live server, ONNX model, cached embeddings | Task status `FINISHED` |

### test_train_inference_flow.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_get_config_options_for_classification` | Classification config options returned | Live server | `status_code == 200`, options list |
| `test_get_config_options_for_regression` | Regression config options returned | Live server | `status_code == 200`, options present |
| `test_get_config_options_invalid_protocol` | Invalid protocol returns error | Live server | `status_code in [400, 500]` |
| `test_verify_valid_classification_config` | Valid classification config accepted | Live server | `status_code == 200`, error field present |
| `test_verify_valid_regression_config` | Valid regression config accepted | Live server | `status_code == 200`, error field present |
| `test_verify_invalid_protocol_config` | Invalid protocol config returns error | Live server | Error message not empty |
| `test_start_training_with_real_proteins` | Training with real sequences starts | Live server | Task submitted, not immediately failed |
| `test_start_regression_training` | Regression training starts | Live server | Task submitted, not immediately failed |
| `test_start_training_empty_data_rejected` | Empty training data rejected | Live server | `status_code == 422` |
| `test_start_training_invalid_config` | Invalid config rejected | Live server | `status_code in [400, 422]` |
| `test_start_inference_empty_sequences_rejected` | Empty sequences rejected | Live server | `status_code == 422` |
| `test_start_inference_with_standard_sequences` | Inference with sequences accepted or 404 | Live server | `status_code in [200, 404]` |
| `test_get_model_files_nonexistent` | Non-existent model returns 404 | Live server | `status_code in [404, 500]` |
| `test_train_then_inference_flow` | Full train → inference flow | Live server, Redis | Training completes, inference completes, predictions match input |
| `test_get_model_files_after_training` | Model files retrievable after training | Live server, Redis | All expected file keys present |
| `test_training_with_train_val_split` | Train/val split accepted | Live server | Task submitted successfully |

---

## Property-Based Tests

Property-based tests verify invariants and mathematical properties using oracles.

### test_embedding_oracles.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_embedding_matches_across_batch_sizes` (FixedEmbedder) | Batch invariance: single vs batched embedding identical | FixedEmbedder | Cosine distance ≤ threshold (1.0) |
| `test_embedding_stable_under_progressive_masking` (FixedEmbedder) | Masking robustness: embeddings stable at low (0-30%) masking | FixedEmbedder | Cosine distance ≤ threshold (1.0) |
| `test_embedding_matches_across_batch_sizes` (ESM2) | Batch invariance with real ESM2 model | ESM2 via biotrainer | Cosine distance ≤ 0.2 |
| `test_embedding_stable_under_progressive_masking` (ESM2) | Masking robustness with real ESM2 model | ESM2 via biotrainer | Cosine distance ≤ 0.2 |

### test_prediction_oracles.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_secondary_structure_determinism` | SS predictions deterministic across runs | MockPredictor | All runs identical |
| `test_binding_site_determinism` | Binding predictions deterministic | MockPredictor | All runs identical |
| `test_disorder_determinism` | Disorder predictions deterministic | MockPredictor | All runs identical |
| `test_secondary_structure_output_validity` | SS output probabilities valid | MockPredictor | Probs in [0,1], sum to 1 |
| `test_binding_site_output_validity` | Binding output valid | MockPredictor | Probs in [0,1], sum to 1 |
| `test_disorder_output_validity` | Disorder values in [0,1] | MockPredictor | Values within range |
| `test_secondary_structure_shape_invariance` | SS output shape matches sequence length | MockPredictor | Length matches, class count matches |
| `test_binding_site_shape_invariance` | Binding output shape correct | MockPredictor | Length matches, class count matches |

### test_projection_oracles.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_pca_determinism` | PCA projections deterministic | MockProjector | All runs identical |
| `test_umap_determinism` | UMAP projections deterministic (fixed seed) | MockProjector | All runs identical |
| `test_pca_dimensionality_2d` | PCA output has correct dimensions | MockProjector | Shape `(n_components,)` |
| `test_umap_dimensionality_2d` | UMAP output has correct dimensions | MockProjector | Shape `(n_components,)` |
| `test_pca_values_valid` | PCA values finite and bounded | MockProjector | No NaN/Inf, values < 1000 |
| `test_umap_values_valid` | UMAP values valid for diverse inputs | MockProjector | No NaN/Inf, values bounded |
| `test_pca_preserves_distances` | PCA preserves pairwise distances | MockProjector | Correlation ≥ 0.4 |
| `test_umap_preserves_local_structure` | UMAP preserves local structure | MockProjector | Correlation ≥ 0.3 |

---

## Performance Tests

Performance tests measure throughput, latency, memory usage, and scaling behavior.

### test_embedding_throughput.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_short_sequence_latency` | 10 aa embedding latency | FixedEmbedder | Output shape correct |
| `test_medium_sequence_latency` | ~79 aa embedding latency | FixedEmbedder | Output shape correct |
| `test_long_sequence_latency` | ~211 aa embedding latency | FixedEmbedder | Output shape correct |
| `test_very_long_sequence_latency` | 400 aa embedding latency | FixedEmbedder | Output shape correct |
| `test_small_batch_throughput` | 5 sequence batch throughput | FixedEmbedder | Result count matches |
| `test_medium_batch_throughput` | 15 sequence batch throughput | FixedEmbedder | Result count matches |
| `test_large_batch_throughput` | Full dataset batch throughput | FixedEmbedder | Result count matches |
| `test_pooled_single_sequence` | Single pooled embedding throughput | FixedEmbedder | Shape `(dim,)` |
| `test_pooled_batch` | Batch pooled embedding throughput | FixedEmbedder | All shapes `(dim,)` |
| `test_dict_embedding_throughput` | Dict format embedding throughput | FixedEmbedder | Keys preserved |

### test_esm2_throughput.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_short_sequence_latency` | 10 aa ESM2 latency | Real ESM2 model | Output shape `(len, 320)` |
| `test_medium_sequence_latency` | ~79 aa ESM2 latency | Real ESM2 model | Output shape correct |
| `test_long_sequence_latency` | ~211 aa ESM2 latency | Real ESM2 model | Output shape correct |
| `test_very_long_sequence_latency` | 400 aa ESM2 latency | Real ESM2 model | Output shape correct |
| `test_canonical_dataset_throughput` | Full dataset ESM2 throughput | Real ESM2 model | All sequences embedded |
| `test_small_batch_throughput` | Small batch ESM2 throughput | Real ESM2 model | Result count matches |
| `test_pooled_embedding_latency` | Pooled ESM2 embedding latency | Real ESM2 model | Shape `(320,)` |
| `test_esm2_vs_fixed_embedder` | ESM2 vs FixedEmbedder comparison | Real ESM2, FixedEmbedder | Both produce valid embeddings |
| `test_scaling_report` | ESM2 scaling with sequence length | Real ESM2 model | Mean latency > 0 |

### test_memory_usage.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_no_leak_repeated_single_embedding` | No memory leak on repeated single embeds | FixedEmbedder | Memory growth < 100 MB |
| `test_no_leak_repeated_batch_embedding` | No memory leak on repeated batch embeds | FixedEmbedder | Memory growth < 200 MB |
| `test_gc_releases_embeddings` | GC properly releases embedding memory | FixedEmbedder | Memory decreases after GC |
| `test_embedding_memory_size` | Embedding memory matches expected | FixedEmbedder | `nbytes == seq_len * dim * 4` |
| `test_batch_memory_size` | Batch total memory measured | FixedEmbedder | Total bytes calculated |
| `test_pooled_vs_per_residue_memory` | Pooled uses <10% of per-residue memory | FixedEmbedder | Pooled < per_residue / 10 |
| `test_memory_per_dimension` | Memory scales with embedding dimension | FixedEmbedder(various dims) | Sizes recorded |
| `test_estimate_batch_memory` | Memory estimates for batch configs | FixedEmbedder | Estimates printed |

### test_scaling.py

| Test Name | What It Verifies | Dependencies/Mocks | Key Assertions |
|-----------|------------------|-------------------|----------------|
| `test_linear_scaling_with_length` | Time scales O(n) with sequence length | FixedEmbedder | Time ratio < 3x length ratio |
| `test_collect_scaling_data` | Scaling data collected | FixedEmbedder | Data printed |
| `test_linear_scaling_with_batch_size` | Time scales O(n) with batch size | FixedEmbedder | Time ratio < 3x size ratio |
| `test_collect_batch_scaling_data` | Batch scaling data collected | FixedEmbedder | Data printed |
| `test_batch_vs_sequential` | Batch vs sequential comparison | FixedEmbedder | Results identical, times compared |

---

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.property` | Property-based oracle tests |
| `@pytest.mark.integration` | Integration tests requiring live server |
| `@pytest.mark.performance` | Performance benchmark tests |
| `@pytest.mark.slow` | Long-running tests (ESM2 model, full training) |

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run unit tests only
uv run pytest tests/unit/ -v

# Run integration tests (requires server)
CI_SERVER_URL=http://localhost:9540 uv run pytest tests/integration/ -v -m integration

# Run property tests
uv run pytest tests/property/ -v -m property

# Run performance benchmarks
uv run pytest tests/performance/ -v -m performance

# Skip slow tests
uv run pytest tests/ -m "not slow"
```
