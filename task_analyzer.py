"""
Task Analyzer Module

Analyzes coverage and complexity of generated tasks.

Usage:
    python task_analyzer.py --generated-dir generated_teams

Output includes:
- Number of unique tool sequences
- Tool coverage (fraction of tools used at least once)
- Distribution of tool sequence lengths
- Distribution of task complexity
- Tool usage frequency
- Algorithm distribution
- And more stats
"""

import os
import json
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class TaskStats:
    """Container for task analysis statistics."""
    total_tasks: int = 0
    unique_tool_sequences: int = 0
    total_tools_in_spec: int = 0
    tools_used: Set[str] = field(default_factory=set)
    tool_coverage: float = 0.0
    
    # Sequence length distribution
    sequence_lengths: List[int] = field(default_factory=list)
    length_distribution: Dict[int, int] = field(default_factory=dict)
    
    # Complexity distribution
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Tool usage frequency
    tool_frequency: Dict[str, int] = field(default_factory=dict)
    
    # Algorithm distribution
    algorithm_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Verifier stats
    total_verifiers: int = 0
    avg_verifiers_per_task: float = 0.0
    verifier_type_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Task prompt stats
    avg_prompt_length: float = 0.0
    min_prompt_length: int = 0
    max_prompt_length: int = 0
    
    # Tools never used
    unused_tools: Set[str] = field(default_factory=set)
    
    # Tool pair co-occurrence
    tool_pairs: Dict[Tuple[str, str], int] = field(default_factory=dict)
    
    # Sequence patterns
    all_sequences: List[Tuple[str, ...]] = field(default_factory=list)
    duplicate_sequences: int = 0


class TaskAnalyzer:
    """Analyzes generated tasks for coverage and complexity metrics."""
    
    def __init__(self, generated_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            generated_dir: Directory containing spec.json and tasks/ subfolder
        """
        self.generated_dir = os.path.abspath(generated_dir)
        self.tasks_dir = os.path.join(generated_dir, "tasks")
        
        # Load spec
        spec_path = os.path.join(generated_dir, "spec.json")
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"spec.json not found in {generated_dir}")
        
        with open(spec_path, 'r', encoding='utf-8') as f:
            self.spec = json.load(f)
        
        self.all_tools = {func["name"] for func in self.spec.get("functions", [])}
        self.tool_descriptions = {
            func["name"]: func.get("description", "") 
            for func in self.spec.get("functions", [])
        }
    
    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load all task JSON files from the tasks directory."""
        tasks = []
        
        if not os.path.exists(self.tasks_dir):
            print(f"Warning: Tasks directory not found: {self.tasks_dir}")
            return tasks
        
        for filename in sorted(os.listdir(self.tasks_dir)):
            if filename.startswith("task_") and filename.endswith(".json"):
                task_path = os.path.join(self.tasks_dir, filename)
                try:
                    with open(task_path, 'r', encoding='utf-8') as f:
                        task_data = json.load(f)
                        task_data["_filename"] = filename
                        tasks.append(task_data)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not load {filename}: {e}")
        
        return tasks
    
    def extract_tool_sequences(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, ...]]:
        """Extract all tool sequences from tasks."""
        sequences = []
        
        for task in tasks:
            for scenario in task.get("scenarios", []):
                for prompt in scenario.get("prompts", []):
                    expected_tools = prompt.get("expected_tools", [])
                    if expected_tools:
                        sequences.append(tuple(expected_tools))
        
        return sequences
    
    def extract_metadata(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract metadata from tasks."""
        metadata_list = []
        
        for task in tasks:
            for scenario in task.get("scenarios", []):
                meta = scenario.get("metadata", {})
                meta["_task_name"] = task.get("name", "")
                meta["_scenario_id"] = scenario.get("scenario_id", "")
                metadata_list.append(meta)
        
        return metadata_list
    
    def extract_verifiers(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract all verifiers from tasks."""
        verifiers = []
        
        for task in tasks:
            for scenario in task.get("scenarios", []):
                for prompt in scenario.get("prompts", []):
                    for v in prompt.get("verifier", []):
                        verifiers.append(v)
        
        return verifiers
    
    def extract_prompts(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Extract all prompt texts from tasks."""
        prompts = []
        
        for task in tasks:
            for scenario in task.get("scenarios", []):
                for prompt in scenario.get("prompts", []):
                    text = prompt.get("prompt_text", "")
                    if text:
                        prompts.append(text)
        
        return prompts
    
    def analyze(self) -> TaskStats:
        """Perform comprehensive analysis of generated tasks."""
        stats = TaskStats()
        
        # Load tasks
        tasks = self.load_tasks()
        stats.total_tasks = len(tasks)
        
        if stats.total_tasks == 0:
            print("No tasks found to analyze.")
            return stats
        
        # Tool sequences
        sequences = self.extract_tool_sequences(tasks)
        stats.all_sequences = sequences
        
        unique_sequences = set(sequences)
        stats.unique_tool_sequences = len(unique_sequences)
        stats.duplicate_sequences = len(sequences) - len(unique_sequences)
        
        # Tool coverage
        stats.total_tools_in_spec = len(self.all_tools)
        for seq in sequences:
            stats.tools_used.update(seq)
        
        stats.tool_coverage = len(stats.tools_used) / len(self.all_tools) if self.all_tools else 0.0
        stats.unused_tools = self.all_tools - stats.tools_used
        
        # Sequence length distribution
        stats.sequence_lengths = [len(seq) for seq in sequences]
        stats.length_distribution = dict(Counter(stats.sequence_lengths))
        
        # Tool frequency
        tool_counts = Counter()
        for seq in sequences:
            tool_counts.update(seq)
        stats.tool_frequency = dict(tool_counts)
        
        # Tool pair co-occurrence (adjacent tools in sequences)
        pair_counts = Counter()
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += 1
        stats.tool_pairs = dict(pair_counts)
        
        # Metadata analysis (complexity, algorithm)
        metadata_list = self.extract_metadata(tasks)
        
        complexity_counts = Counter()
        algorithm_counts = Counter()
        
        for meta in metadata_list:
            difficulty = meta.get("difficulty", "unknown")
            complexity_counts[difficulty] += 1
            
            algorithm = meta.get("algorithm", "unknown")
            algorithm_counts[algorithm] += 1
        
        stats.complexity_distribution = dict(complexity_counts)
        stats.algorithm_distribution = dict(algorithm_counts)
        
        # Verifier analysis
        verifiers = self.extract_verifiers(tasks)
        stats.total_verifiers = len(verifiers)
        stats.avg_verifiers_per_task = len(verifiers) / stats.total_tasks if stats.total_tasks > 0 else 0
        
        verifier_type_counts = Counter()
        for v in verifiers:
            vtype = v.get("verifier_type", "unknown")
            verifier_type_counts[vtype] += 1
        stats.verifier_type_distribution = dict(verifier_type_counts)
        
        # Prompt analysis
        prompts = self.extract_prompts(tasks)
        if prompts:
            prompt_lengths = [len(p) for p in prompts]
            stats.avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths)
            stats.min_prompt_length = min(prompt_lengths)
            stats.max_prompt_length = max(prompt_lengths)
        
        return stats
    
    def print_report(self, stats: TaskStats) -> None:
        """Print a formatted analysis report."""
        print("\n" + "=" * 70)
        print("TASK ANALYSIS REPORT")
        print("=" * 70)
        
        # Overview
        print("\n## OVERVIEW")
        print(f"  Total tasks analyzed: {stats.total_tasks}")
        print(f"  Unique tool sequences: {stats.unique_tool_sequences}")
        print(f"  Duplicate sequences: {stats.duplicate_sequences}")
        
        # Tool Coverage
        print("\n## TOOL COVERAGE")
        print(f"  Total tools in spec: {stats.total_tools_in_spec}")
        print(f"  Tools used at least once: {len(stats.tools_used)}")
        print(f"  Coverage: {stats.tool_coverage:.1%}")
        
        if stats.unused_tools:
            print(f"\n  Unused tools ({len(stats.unused_tools)}):")
            for tool in sorted(stats.unused_tools):
                desc = self.tool_descriptions.get(tool, "")[:50]
                print(f"    - {tool}: {desc}...")
        
        # Sequence Length Distribution
        print("\n## SEQUENCE LENGTH DISTRIBUTION")
        if stats.length_distribution:
            avg_len = sum(stats.sequence_lengths) / len(stats.sequence_lengths)
            print(f"  Average sequence length: {avg_len:.2f}")
            print(f"  Min: {min(stats.sequence_lengths)}, Max: {max(stats.sequence_lengths)}")
            print("\n  Length -> Count:")
            for length in sorted(stats.length_distribution.keys()):
                count = stats.length_distribution[length]
                bar = "█" * min(count, 40)
                pct = count / len(stats.sequence_lengths) * 100
                print(f"    {length} tools: {count:3d} ({pct:5.1f}%) {bar}")
        
        # Complexity Distribution
        print("\n## COMPLEXITY DISTRIBUTION")
        if stats.complexity_distribution:
            for complexity, count in sorted(stats.complexity_distribution.items()):
                pct = count / stats.total_tasks * 100
                bar = "█" * min(count, 30)
                print(f"    {complexity:10s}: {count:3d} ({pct:5.1f}%) {bar}")
        else:
            print("  No complexity metadata available")
        
        # Algorithm Distribution
        print("\n## ALGORITHM DISTRIBUTION")
        if stats.algorithm_distribution:
            for algo, count in sorted(stats.algorithm_distribution.items()):
                pct = count / stats.total_tasks * 100
                bar = "█" * min(count, 30)
                print(f"    {algo:15s}: {count:3d} ({pct:5.1f}%) {bar}")
        else:
            print("  No algorithm metadata available")
        
        # Tool Usage Frequency
        print("\n## TOOL USAGE FREQUENCY (Top 15)")
        if stats.tool_frequency:
            sorted_tools = sorted(stats.tool_frequency.items(), key=lambda x: -x[1])
            max_count = max(stats.tool_frequency.values())
            for tool, count in sorted_tools[:15]:
                bar_len = int(count / max_count * 30)
                bar = "█" * bar_len
                print(f"    {tool:30s}: {count:3d} {bar}")
        
        # Verifier Stats
        print("\n## VERIFIER STATISTICS")
        print(f"  Total verifiers: {stats.total_verifiers}")
        print(f"  Average verifiers per task: {stats.avg_verifiers_per_task:.2f}")
        if stats.verifier_type_distribution:
            print("  Verifier types:")
            for vtype, count in stats.verifier_type_distribution.items():
                print(f"    - {vtype}: {count}")
        
        # Prompt Stats
        print("\n## PROMPT STATISTICS")
        print(f"  Average prompt length: {stats.avg_prompt_length:.0f} characters")
        print(f"  Min prompt length: {stats.min_prompt_length}")
        print(f"  Max prompt length: {stats.max_prompt_length}")
        
        # Common Tool Pairs
        print("\n## COMMON TOOL PAIRS (Top 10)")
        if stats.tool_pairs:
            sorted_pairs = sorted(stats.tool_pairs.items(), key=lambda x: -x[1])
            for (tool1, tool2), count in sorted_pairs[:10]:
                print(f"    {tool1} -> {tool2}: {count}")
        
        # Recommendations
        print("\n## RECOMMENDATIONS")
        self._print_recommendations(stats)
        
        print("\n" + "=" * 70)
    
    def _print_recommendations(self, stats: TaskStats) -> None:
        """Print recommendations based on analysis."""
        recommendations = []
        
        # Coverage recommendations
        if stats.tool_coverage < 0.5:
            recommendations.append(
                f"LOW COVERAGE: Only {stats.tool_coverage:.0%} of tools are used. "
                f"Consider generating more tasks to cover {len(stats.unused_tools)} unused tools."
            )
        elif stats.tool_coverage < 0.8:
            recommendations.append(
                f"MODERATE COVERAGE: {stats.tool_coverage:.0%} of tools used. "
                f"{len(stats.unused_tools)} tools still unused."
            )
        else:
            recommendations.append(
                f"GOOD COVERAGE: {stats.tool_coverage:.0%} of tools are used."
            )
        
        # Sequence length recommendations
        if stats.sequence_lengths:
            avg_len = sum(stats.sequence_lengths) / len(stats.sequence_lengths)
            if avg_len < 2.5:
                recommendations.append(
                    f"SHORT SEQUENCES: Average length is {avg_len:.1f}. "
                    "Consider generating more complex multi-tool tasks."
                )
            elif avg_len > 5:
                recommendations.append(
                    f"LONG SEQUENCES: Average length is {avg_len:.1f}. "
                    "Tasks may be too complex; consider simpler scenarios."
                )
        
        # Complexity recommendations
        if stats.complexity_distribution:
            simple_count = stats.complexity_distribution.get("simple", 0)
            complex_count = stats.complexity_distribution.get("complex", 0)
            total = sum(stats.complexity_distribution.values())
            
            if simple_count / total > 0.6:
                recommendations.append(
                    "MOSTLY SIMPLE: Over 60% of tasks are simple. "
                    "Generate more medium/complex tasks for better coverage."
                )
            elif complex_count / total > 0.5:
                recommendations.append(
                    "MANY COMPLEX: Over 50% are complex. "
                    "Add more simple/medium tasks for balanced difficulty."
                )
        
        # Duplicate recommendations
        if stats.duplicate_sequences > 0:
            dup_pct = stats.duplicate_sequences / len(stats.all_sequences) * 100
            recommendations.append(
                f"DUPLICATES: {stats.duplicate_sequences} duplicate sequences ({dup_pct:.1f}%). "
                "These should be removed for uniqueness."
            )
        
        # Verifier recommendations
        if stats.avg_verifiers_per_task < 1:
            recommendations.append(
                f"FEW VERIFIERS: Average {stats.avg_verifiers_per_task:.1f} verifiers per task. "
                "Tasks may lack proper validation."
            )
        
        # Print recommendations
        if not recommendations:
            recommendations.append("Task set looks well-balanced. No immediate recommendations.")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    def export_json(self, stats: TaskStats, output_path: str) -> None:
        """Export statistics to JSON file."""
        export_data = {
            "overview": {
                "total_tasks": stats.total_tasks,
                "unique_tool_sequences": stats.unique_tool_sequences,
                "duplicate_sequences": stats.duplicate_sequences,
            },
            "tool_coverage": {
                "total_tools_in_spec": stats.total_tools_in_spec,
                "tools_used_count": len(stats.tools_used),
                "coverage_percentage": round(stats.tool_coverage * 100, 2),
                "tools_used": sorted(list(stats.tools_used)),
                "unused_tools": sorted(list(stats.unused_tools)),
            },
            "sequence_length": {
                "distribution": stats.length_distribution,
                "average": round(sum(stats.sequence_lengths) / len(stats.sequence_lengths), 2) if stats.sequence_lengths else 0,
                "min": min(stats.sequence_lengths) if stats.sequence_lengths else 0,
                "max": max(stats.sequence_lengths) if stats.sequence_lengths else 0,
            },
            "complexity_distribution": stats.complexity_distribution,
            "algorithm_distribution": stats.algorithm_distribution,
            "tool_frequency": stats.tool_frequency,
            "verifiers": {
                "total": stats.total_verifiers,
                "average_per_task": round(stats.avg_verifiers_per_task, 2),
                "type_distribution": stats.verifier_type_distribution,
            },
            "prompts": {
                "average_length": round(stats.avg_prompt_length, 2),
                "min_length": stats.min_prompt_length,
                "max_length": stats.max_prompt_length,
            },
            "tool_pairs": {f"{k[0]}->{k[1]}": v for k, v in stats.tool_pairs.items()},
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\nStatistics exported to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze coverage and complexity of generated tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python task_analyzer.py --generated-dir generated_teams
  python task_analyzer.py --generated-dir generated_teams --export stats.json
  python task_analyzer.py --generated-dir generated_teams --quiet --export stats.json
        """
    )
    parser.add_argument(
        "--generated-dir", 
        required=True,
        help="Directory containing spec.json and tasks/ subfolder"
    )
    parser.add_argument(
        "--export",
        help="Export statistics to JSON file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output (only show summary)"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = TaskAnalyzer(args.generated_dir)
        stats = analyzer.analyze()
        
        if not args.quiet:
            analyzer.print_report(stats)
        else:
            # Print brief summary
            print(f"Tasks: {stats.total_tasks}")
            print(f"Unique sequences: {stats.unique_tool_sequences}")
            print(f"Tool coverage: {stats.tool_coverage:.1%}")
        
        if args.export:
            analyzer.export_json(stats, args.export)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise
    
    return 0


if __name__ == "__main__":
    exit(main())
