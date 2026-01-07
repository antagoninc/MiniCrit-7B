# /gen-training-data

Generate MiniCrit adversarial reasoning training examples.

## Parameters

- `--domain` - Target domain (FIN, COMP, MED, OPS, STAT, GEN)
- `--flaw` - Specific flaw ID or category (e.g., L01, statistical_fallacies)
- `--count` - Number of examples to generate (default: 1000)
- `--output` - Output file path (default: data/generated/{timestamp}.csv)

## Process

1. Load flaw taxonomy from `src/taxonomy/`
2. Select input templates for specified domain/flaw
3. Generate diverse inputs using templates + variation
4. For each input:
   - Generate 3-5 different rebuttal approaches
   - Score each for quality
   - Keep best rebuttal
5. Validate all examples against schema
6. Save with checkpoint every 100 examples
7. Report statistics

## Output Schema

```csv
input_id,flaw_id,domain,flawed_reasoning,rebuttal,confidence,generated_at
```

## Quality Checks

- Rebuttal length: 50-500 words
- Must identify the specific flaw
- Must explain why the reasoning is flawed
- Must NOT just say "this is wrong" without explanation
- Diversity: no two rebuttals should be >80% similar

## Example Usage

```
/gen-training-data --domain FIN --count 5000
/gen-training-data --flaw L01 --count 1000 --output data/affirming_consequent.csv
/gen-training-data --domain COMP --flaw S05 --count 2000  # compliance + survivorship bias
```

## Resume

If interrupted, run with same parameters - will resume from last checkpoint.
