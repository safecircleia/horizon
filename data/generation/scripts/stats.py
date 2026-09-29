#!/usr/bin/env python3
"""Dataset statistics and validation script for SafeCircle.

This script analyzes generated conversation datasets and provides:
- Comprehensive statistics by category, severity, and generator
- Vocabulary and message length analysis
- Dataset quality validation
- Pretty-printed reports with visual formatting

Usage:
    python stats.py                          # Analyze data/raw/
    python stats.py --input data/raw/        # Analyze specific directory
    python stats.py --validate               # Run full validation checks
    python stats.py --input data/raw/ --validate
"""

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonlines

from data.generation.validators.quality import validate_conversation_quality
from data.generation.validators.schemas import (
    SyntheticConversation,
)


class DatasetStats:
    """Analyzer for dataset statistics and quality."""

    def __init__(self, data_dir: str = "data/raw/"):
        """Initialize stats analyzer.

        Args:
            data_dir: Directory containing JSONL files
        """
        self.data_dir = Path(data_dir)
        self.conversations: list[SyntheticConversation] = []
        self.errors: list[dict[str, Any]] = []

    def load_conversations(self) -> bool:
        """Load all conversations from JSONL files.

        Returns:
            True if successful, False otherwise
        """
        if not self.data_dir.exists():
            print(f"Error: Directory not found: {self.data_dir}")
            return False

        jsonl_files = list(self.data_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"Warning: No JSONL files found in {self.data_dir}")
            return False

        total_loaded = 0
        for jsonl_file in sorted(jsonl_files):
            try:
                with jsonlines.open(jsonl_file) as reader:
                    for line_num, obj in enumerate(reader, 1):
                        try:
                            conv = SyntheticConversation(**obj)
                            self.conversations.append(conv)
                            total_loaded += 1
                        except Exception as e:
                            self.errors.append(
                                {
                                    "file": str(jsonl_file),
                                    "line": line_num,
                                    "error": str(e),
                                    "type": "validation",
                                }
                            )
            except Exception as e:
                self.errors.append(
                    {"file": str(jsonl_file), "error": str(e), "type": "read"}
                )

        print(f"Loaded {total_loaded} conversations from {len(jsonl_files)} files")
        if self.errors:
            print(f"Encountered {len(self.errors)} errors during loading")

        return len(self.conversations) > 0

    def get_basic_stats(self) -> dict[str, Any]:
        """Calculate basic dataset statistics.

        Returns:
            Dictionary with basic stats
        """
        if not self.conversations:
            return {}

        total = len(self.conversations)

        # Category distribution
        category_counts = Counter(c.category.value for c in self.conversations)

        # Risk level distribution
        risk_counts = Counter(c.label.risk_level.value for c in self.conversations)

        # Generator distribution
        generator_counts = Counter(
            c.metadata.get("generator", "unknown") for c in self.conversations
        )

        # Message statistics
        message_counts = [len(c.messages) for c in self.conversations]
        avg_messages = sum(message_counts) / total if message_counts else 0
        min_messages = min(message_counts) if message_counts else 0
        max_messages = max(message_counts) if message_counts else 0

        # Content length statistics (total characters)
        content_lengths = [
            sum(len(msg.content) for msg in c.messages) for c in self.conversations
        ]
        avg_content_length = sum(content_lengths) / total if content_lengths else 0

        # Vocabulary statistics
        all_tokens = []
        for conv in self.conversations:
            for msg in conv.messages:
                tokens = msg.content.lower().split()
                all_tokens.extend(tokens)

        unique_tokens = set(all_tokens)
        vocabulary_size = len(unique_tokens)

        return {
            "total_conversations": total,
            "total_messages": sum(message_counts),
            "categories": dict(category_counts),
            "risk_levels": dict(risk_counts),
            "generators": dict(generator_counts),
            "message_stats": {
                "average": round(avg_messages, 2),
                "min": min_messages,
                "max": max_messages,
            },
            "content_stats": {
                "average_length": round(avg_content_length, 2),
                "min_length": min(content_lengths) if content_lengths else 0,
                "max_length": max(content_lengths) if content_lengths else 0,
            },
            "vocabulary": {
                "unique_tokens": vocabulary_size,
                "total_tokens": len(all_tokens),
            },
        }

    def get_category_breakdown(self) -> dict[str, dict[str, Any]]:
        """Get detailed breakdown by category.

        Returns:
            Dictionary with per-category statistics
        """
        breakdown = defaultdict(
            lambda: {
                "total": 0,
                "risk_levels": Counter(),
                "avg_messages": 0,
                "avg_content_length": 0,
                "generators": Counter(),
            }
        )

        for conv in self.conversations:
            cat = conv.category.value
            breakdown[cat]["total"] += 1
            breakdown[cat]["risk_levels"][conv.label.risk_level.value] += 1
            breakdown[cat]["avg_messages"] += len(conv.messages)

            content_len = sum(len(msg.content) for msg in conv.messages)
            breakdown[cat]["avg_content_length"] += content_len

            gen = conv.metadata.get("generator", "unknown")
            breakdown[cat]["generators"][gen] += 1

        # Convert counters and calculate averages
        result = {}
        for cat, stats in breakdown.items():
            total = stats["total"]
            result[cat] = {
                "total": total,
                "risk_levels": dict(stats["risk_levels"]),
                "avg_messages": (
                    round(stats["avg_messages"] / total, 2) if total > 0 else 0
                ),
                "avg_content_length": (
                    round(stats["avg_content_length"] / total, 2) if total > 0 else 0
                ),
                "generators": dict(stats["generators"]),
            }

        return result

    def validate_dataset(self) -> dict[str, Any]:
        """Run quality validation checks on dataset.

        Returns:
            Dictionary with validation results
        """
        if not self.conversations:
            return {"valid": False, "total_checked": 0, "issues": []}

        valid_count = 0
        invalid_conversations = []

        for conv in self.conversations:
            is_valid, errors = validate_conversation_quality(
                conv.messages,
                min_length=4,
                max_length=30,
                min_unique_tokens=20,
                allow_consecutive_roles=True,
            )

            if is_valid:
                valid_count += 1
            else:
                invalid_conversations.append(
                    {
                        "conversation_id": conv.conversation_id,
                        "category": conv.category.value,
                        "errors": errors,
                    }
                )

        total = len(self.conversations)
        validity_rate = (valid_count / total * 100) if total > 0 else 0

        return {
            "total_checked": total,
            "valid_conversations": valid_count,
            "invalid_conversations": len(invalid_conversations),
            "validity_rate_percent": round(validity_rate, 2),
            "passing": validity_rate >= 95.0,
            "issues": invalid_conversations[:10],  # Report first 10 issues
        }

    def print_header(self, title: str) -> None:
        """Print formatted section header.

        Args:
            title: Header text
        """
        width = 70
        print()
        print("=" * width)
        print(f" {title.center(width - 2)}")
        print("=" * width)

    def print_stats(self) -> None:
        """Print formatted statistics report."""
        stats = self.get_basic_stats()

        if not stats:
            print("No data to display")
            return

        self.print_header("DATASET STATISTICS")

        # Overall statistics
        print()
        print(f"Total Conversations: {stats['total_conversations']}")
        print(f"Total Messages: {stats['total_messages']}")
        print()

        # Category distribution
        print("Category Distribution:")
        total = stats["total_conversations"]
        for cat in sorted(stats["categories"].keys()):
            count = stats["categories"][cat]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {cat:20s}: {count:6d} ({percentage:5.1f}%)")
        print()

        # Risk level distribution
        print("Risk Level Distribution:")
        for level in sorted(stats["risk_levels"].keys()):
            count = stats["risk_levels"][level]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {level:20s}: {count:6d} ({percentage:5.1f}%)")
        print()

        # Generator distribution
        print("Generator Distribution:")
        for gen in sorted(stats["generators"].keys()):
            count = stats["generators"][gen]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {gen:20s}: {count:6d} ({percentage:5.1f}%)")
        print()

        # Message statistics
        print("Message Statistics:")
        msg_stats = stats["message_stats"]
        print(f"  Average per conversation: {msg_stats['average']}")
        print(f"  Range: {msg_stats['min']} to {msg_stats['max']}")
        print()

        # Content statistics
        print("Content Statistics (characters):")
        content_stats = stats["content_stats"]
        print(f"  Average per conversation: {content_stats['average_length']}")
        print(
            f"  Range: {content_stats['min_length']} to {content_stats['max_length']}"
        )
        print()

        # Vocabulary statistics
        print("Vocabulary Statistics:")
        vocab_stats = stats["vocabulary"]
        print(f"  Unique tokens: {vocab_stats['unique_tokens']}")
        print(f"  Total tokens: {vocab_stats['total_tokens']}")
        if vocab_stats["total_tokens"] > 0:
            uniqueness = (
                vocab_stats["unique_tokens"] / vocab_stats["total_tokens"] * 100
            )
            print(f"  Uniqueness ratio: {uniqueness:.1f}%")

    def print_category_breakdown(self) -> None:
        """Print per-category breakdown."""
        breakdown = self.get_category_breakdown()

        self.print_header("CATEGORY BREAKDOWN")

        for category in sorted(breakdown.keys()):
            stats = breakdown[category]
            print()
            print(f"Category: {category.upper()}")
            print(f"  Total: {stats['total']}")
            print()

            print("  Risk Levels:")
            for level in sorted(stats["risk_levels"].keys()):
                count = stats["risk_levels"][level]
                print(f"    {level:15s}: {count}")

            print()
            print("  Generators:")
            for gen in sorted(stats["generators"].keys()):
                count = stats["generators"][gen]
                print(f"    {gen:15s}: {count}")

            print()
            print(f"  Average Messages: {stats['avg_messages']}")
            print(f"  Average Content Length: {stats['avg_content_length']} chars")

    def print_validation_report(self) -> None:
        """Print validation report."""
        validation = self.validate_dataset()

        self.print_header("DATASET VALIDATION")

        print()
        print(f"Total Conversations Checked: {validation['total_checked']}")
        print(f"Valid Conversations: {validation['valid_conversations']}")
        print(f"Invalid Conversations: {validation['invalid_conversations']}")
        print()
        print(f"Validity Rate: {validation['validity_rate_percent']:.1f}%")
        print()

        if validation["passing"]:
            print("Status: PASSED - Dataset meets quality requirements (>= 95%)")
        else:
            print("Status: FAILED - Dataset quality below requirement (< 95%)")

        if validation["issues"]:
            print()
            print("Sample Issues (first 10):")
            for i, issue in enumerate(validation["issues"], 1):
                print(f"  {i}. Conv ID: {issue['conversation_id']}")
                print(f"     Category: {issue['category']}")
                for error in issue["errors"][:2]:  # Show first 2 errors
                    print(f"     - {error}")
                if len(issue["errors"]) > 2:
                    print(f"     ... and {len(issue['errors']) - 2} more errors")

    def print_summary(self) -> None:
        """Print executive summary."""
        self.print_header("SAFECIRCLE DATASET SUMMARY")

        stats = self.get_basic_stats()
        validation = self.validate_dataset()

        if not stats:
            print("No data available")
            return

        print()
        print(f"Dataset generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print(f"Total Conversations: {stats['total_conversations']:,}")
        print(f"Total Messages: {stats['total_messages']:,}")
        print(f"Unique Vocabulary: {stats['vocabulary']['unique_tokens']:,}")
        print()

        # Quality status
        status = "PASSED" if validation["passing"] else "FAILED"
        validity = validation["validity_rate_percent"]
        print(f"Quality Status: {status} ({validity:.1f}% valid)")
        print()

        # Coverage
        categories = stats["categories"]
        print("Category Coverage:")
        for cat in sorted(categories.keys()):
            count = categories[cat]
            bar_width = int(count / 100)  # Scale for display
            bar = "█" * bar_width
            print(f"  {cat:20s}: {count:6d} {bar}")

        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze SafeCircle dataset statistics and quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze default data/raw directory
  python stats.py

  # Analyze specific directory
  python stats.py --input data/processed/

  # Run full validation checks
  python stats.py --validate

  # Combined analysis and validation
  python stats.py --input data/raw/ --validate
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/",
        help="Input directory containing JSONL files (default: data/raw/)",
    )

    parser.add_argument(
        "--validate", action="store_true", help="Run full dataset validation checks"
    )

    args = parser.parse_args()

    # Create analyzer and load data
    analyzer = DatasetStats(args.input)

    print()
    if not analyzer.load_conversations():
        print("Failed to load any conversations")
        sys.exit(1)

    print()

    # Print summary first
    analyzer.print_summary()

    # Print detailed statistics
    analyzer.print_stats()

    # Print category breakdown
    analyzer.print_category_breakdown()

    # Print validation if requested
    if args.validate:
        analyzer.print_validation_report()

    print()


if __name__ == "__main__":
    main()
