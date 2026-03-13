#!/usr/bin/env python3
"""
Wraps bare Java method completions in a class so SonarQube can parse them.

Scans all .java files in CoderEval-Java-* folders under both emp-quant destinations.
If a file doesn't already start with 'import' or 'class' or 'package', wraps it in:
    class Generated { ... }

Usage:
    python fix_java_class_wrapper.py [--dry-run]
"""

import os
import sys
import glob

DRY_RUN = "--dry-run" in sys.argv

JAVA_DIRS = [
    "/scratch/oldhome/safrin/projects/SonarQube-Analysis-java/emp-quant",
]

def needs_wrapping(content):
    """Check if the Java file is a bare method (no class/import/package declaration)."""
    stripped = content.strip()
    if not stripped:
        return False
    # If it already has a class structure, skip
    for prefix in ["import ", "class ", "package ", "public class ", "abstract class ", "interface "]:
        if stripped.startswith(prefix):
            return False
    return True

def wrap_in_class(content):
    """Wrap bare Java code in a class declaration."""
    # Indent each line
    lines = content.rstrip().split("\n")
    indented = "\n".join("    " + line for line in lines)
    return f"class Generated {{\n{indented}\n}}\n"

def main():
    print()
    print("=" * 55)
    print("  Fix Java files: Add class wrapper for SonarQube")
    print("=" * 55)
    if DRY_RUN:
        print("  *** DRY RUN MODE — nothing will be modified ***")
    print()

    total_wrapped = 0
    total_skipped = 0

    for base_dir in JAVA_DIRS:
        # Find all CoderEval-Java-* directories
        for folder in sorted(glob.glob(os.path.join(base_dir, "CoderEval-Java-*"))):
            folder_name = os.path.basename(folder)
            java_files = sorted(glob.glob(os.path.join(folder, "*.java")))

            if not java_files:
                continue

            wrapped_count = 0
            skipped_count = 0

            for java_path in java_files:
                with open(java_path, "r") as f:
                    content = f.read()

                if needs_wrapping(content):
                    if not DRY_RUN:
                        wrapped_content = wrap_in_class(content)
                        with open(java_path, "w") as f:
                            f.write(wrapped_content)
                    wrapped_count += 1
                else:
                    skipped_count += 1

            print(f"  {folder_name}: wrapped {wrapped_count}, skipped {skipped_count}")
            total_wrapped += wrapped_count
            total_skipped += skipped_count

    print()
    print("=" * 55)
    print(f"  Total wrapped : {total_wrapped}")
    print(f"  Total skipped : {total_skipped}")
    print("=" * 55)
    if DRY_RUN:
        print("  This was a dry run. Re-run without --dry-run to execute.")
    print()

if __name__ == "__main__":
    main()