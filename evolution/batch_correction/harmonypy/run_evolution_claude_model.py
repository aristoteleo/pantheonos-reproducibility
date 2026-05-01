#!/usr/bin/env python
"""Run evolution with Claude Opus 4.6 as mutator and analyzer model."""

import asyncio
import os
import sys
from pathlib import Path

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Set API key and data dir
os.environ["ANTHROPIC_API_KEY"] = "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"
os.environ.setdefault("HARMONY_DATA_DIR", str(Path(__file__).parent.parent.resolve() / "data"))


async def main():
    from pantheon.evolution import EvolutionTeam, EvolutionConfig
    from pantheon.evolution.program import CodebaseSnapshot

    example_dir = Path(__file__).parent
    output_dir = str(example_dir / "results_claude_model")

    # Load initial code and evaluator
    initial_code = CodebaseSnapshot.from_single_file(
        "harmony.py",
        (example_dir / "harmony.py").read_text(),
    )
    evaluator_code = (example_dir / "evaluator.py").read_text()

    config = EvolutionConfig(
        max_iterations=300,
        num_workers=4,
        num_islands=3,
        num_inspirations=2,
        num_top_programs=3,
        max_parallel_evaluations=2,
        evaluation_timeout=180,
        analyzer_timeout=600,   # Opus needs more time for deep analysis with think tool
        mutation_timeout=600,   # Opus generates longer, more detailed diffs

        # Use Claude Opus 4.6 for mutation and analysis
        mutator_model="anthropic/claude-opus-4-6",
        analyzer_model="anthropic/claude-opus-4-6",

        feature_dimensions=["mixing_score", "speed_score", "bio_conservation_score"],
        early_stop_generations=200,
        function_weight=1.0,
        llm_weight=0.0,
        log_level="INFO",
        log_iterations=True,
        checkpoint_interval=5,
        db_path=output_dir,
    )

    objective = """Optimize the Harmony algorithm implementation for:

1. **Integration Quality** (40% weight): Improve batch mixing while preserving biological structure.
   - The algorithm should effectively remove batch effects
   - Biological clusters should remain distinct after correction

2. **Performance** (20% weight): Reduce execution time.
   - Optimize hot loops and matrix operations
   - Consider vectorization opportunities
   - Avoid redundant computations

3. **Convergence** (10% weight): Improve convergence behavior.
   - Reduce number of iterations needed
   - Ensure stable convergence

4. **Biological Conservation** (30% weight): Preserve biological variance.
   - Don't over-correct and remove biological signal
   - Maintain cluster separation

Constraints:
- Keep the public API (run_harmony function signature)
- Maintain numerical stability
- Don't remove essential functionality
"""

    print("=" * 60)
    print("Pantheon Evolution: Harmony (Claude Opus 4.6)")
    print("=" * 60)
    print(f"Model: anthropic/claude-opus-4-6")
    print(f"Iterations: {config.max_iterations}")
    print(f"Workers: {config.num_workers}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    team = EvolutionTeam(config=config)
    result = await team.evolve(
        initial_code=initial_code,
        evaluator_code=evaluator_code,
        objective=objective,
        resume_from=output_dir,
    )

    print("\n" + "=" * 60)
    print(result.get_summary())

    # Save best code
    output_path = Path(output_dir)
    best_code_path = output_path / "harmony_optimized.py"
    best_code_path.write_text(result.best_code)
    print(f"\nBest code saved to: {best_code_path}")

    result.save_report(str(output_path / "evolution_report.json"))
    config.to_yaml(str(output_path / "config.yaml"))

    return result


if __name__ == "__main__":
    result = asyncio.run(main())
    print(f"\nFinal best score: {result.best_score:.4f}")
