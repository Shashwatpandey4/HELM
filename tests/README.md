# HELM Tests

Organized test suite for HELM compiler.

## Structure

### `unit/`
Unit tests for individual components:
- `test_quantization.py`: Quantization utilities
- `test_cost_model.py`: Cost modeling
- `test_dynamic_shapes.py`: Dynamic shape handling

### `integration/`
Integration tests for multi-component features:
- `test_tensor_parallelism.py`: TP pass integration
- `test_tp_extensions.py`: TP extensions (Embedding, LayerNorm)
- `test_pipeline_executor.py`: Pipeline execution
- `test_optimizer.py`: ParallelOptimizer
- `test_distributed_simple.py`: Distributed execution

### `e2e/`
End-to-end system tests:
- `test_helm_backend_sanity.py`: Basic compiler sanity check
- `test_device_mesh.py`: Device mesh functionality
- `test_dp_mapping.py`: Data parallelism mapping

## Running Tests

```bash
# All tests
pytest tests/

# Specific category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Specific test
pytest tests/unit/test_quantization.py
```
