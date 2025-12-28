# fao2hkb

FAOSTAT **Bulk Downloads** → **Hierarchical Knowledge Base (HKB)** generator.

## Quick start

```bash
pip install -e .
python -m fao2hkb run --config examples/config.example.yaml
```

Embeddings are optional:

```bash
pip install -e ".[embeddings]"
# then set embeddings.enabled: true in YAML
```
