#!/usr/bin/env python3
"""
Test script for the secure git functions with updated 'git -C' syntax.
"""

import sys
import os
import json

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

from tools import git_status, git_add, git_add_all, git_commit, git_log

def test_git_functions():
    """Test the git functions with the current repository."""
    repo_path = "/run/media/fratq/4593fc5e-12d7-4064-8a55-3ad61a661126/CODE/groq_multitool_test"

    print("Testing git_status...")
    result = git_status(repo_path)
    print(f"git_status result: {result}")

    print("\nTesting git_add_all...")
    result = git_add_all(repo_path)
    print(f"git_add_all result: {result}")

    print("\nTesting git_log...")
    result = git_log(repo_path, limit=5)
    print(f"git_log result: {result}")

    # Test git_commit only if there are changes
    print("\nTesting git_commit...")
    result = git_commit(repo_path, "Test commit from secure executor")
    print(f"git_commit result: {result}")

if __name__ == "__main__":
    test_git_functions()