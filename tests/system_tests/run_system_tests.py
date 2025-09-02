#!/usr/bin/env python3
"""System test runner with configuration validation and reporting."""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import validate_api_keys, check_test_readiness


def print_banner(text: str):
    """Print a formatted banner."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_section(text: str):
    """Print a formatted section header."""
    print(f"\n--- {text} ---")


def validate_environment() -> Dict[str, bool]:
    """Validate the test environment and API keys."""
    print_banner("ENVIRONMENT VALIDATION")
    
    validation = validate_api_keys()
    
    print("API Key Status:")
    print(f"  OpenAI:    {'✅ Configured' if validation['openai'] else '❌ Not configured'}")
    print(f"  Pinecone:  {'✅ Configured' if validation['pinecone'] else '❌ Not configured'}")
    print(f"  Anthropic: {'✅ Configured' if validation['anthropic'] else '❌ Not configured'}")
    
    print("\nTest Readiness:")
    local_ready = check_test_readiness(use_remote=False)
    remote_ready = check_test_readiness(use_remote=True)
    
    print(f"  Local tests:  {'✅ Ready' if local_ready else '❌ Not ready'}")
    print(f"  Remote tests: {'✅ Ready' if remote_ready else '❌ Not ready'}")
    
    if not local_ready:
        print("\n❌ Cannot run system tests: OpenAI API key required")
        print("   Set OPENAI_API_KEY environment variable or edit config.py")
        return {"can_run_local": False, "can_run_remote": False}
    
    if not remote_ready:
        print("\n⚠️  Remote tests will be skipped: Pinecone API key not configured")
        print("   Set PINECONE_API_KEY environment variable to enable remote tests")
    
    return {"can_run_local": local_ready, "can_run_remote": remote_ready}


def run_test_suite(test_file: str, description: str) -> Tuple[bool, float, str]:
    """Run a test suite and return results."""
    print_section(f"Running {description}")
    
    start_time = time.time()
    
    cmd = [
        sys.executable, "-m", "pytest", 
        f"tests/system_tests/{test_file}",
        "-v", "--tb=short"
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600  # 10 minute timeout
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        if success:
            print(f"✅ {description} completed successfully in {duration:.1f}s")
        else:
            print(f"❌ {description} failed after {duration:.1f}s")
            print("STDOUT:", result.stdout[-500:] if result.stdout else "None")
            print("STDERR:", result.stderr[-500:] if result.stderr else "None")
        
        return success, duration, result.stdout + result.stderr
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ {description} timed out after {duration:.1f}s")
        return False, duration, "Test timed out"
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 {description} crashed: {e}")
        return False, duration, str(e)


def run_quick_tests() -> Dict[str, Tuple[bool, float]]:
    """Run quick system tests (excluding slow and performance tests)."""
    print_banner("QUICK SYSTEM TESTS")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/system_tests/",
        "-v", "-m", "not slow and not performance",
        "--tb=short"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        duration = time.time() - start_time
        success = result.returncode == 0
        
        if success:
            print(f"✅ Quick tests completed successfully in {duration:.1f}s")
        else:
            print(f"❌ Quick tests failed after {duration:.1f}s")
            print("Output:", result.stdout[-1000:] if result.stdout else "None")
        
        return {"quick_tests": (success, duration)}
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ Quick tests timed out after {duration:.1f}s")
        return {"quick_tests": (False, duration)}


def run_comprehensive_tests() -> Dict[str, Tuple[bool, float]]:
    """Run comprehensive system tests."""
    print_banner("COMPREHENSIVE SYSTEM TESTS")
    
    test_suites = [
        ("test_basic_operations.py", "Basic Operations"),
        ("test_memory_core_scenario.py", "Memory Core Scenario"),
        ("test_advanced_operations.py", "Advanced Operations"),
        ("test_integration_scenarios.py", "Integration Scenarios")
    ]
    
    results = {}
    
    for test_file, description in test_suites:
        success, duration, output = run_test_suite(test_file, description)
        results[test_file] = (success, duration)
        
        # Small delay between test suites to avoid rate limiting
        if test_file != test_suites[-1][0]:  # Not the last test
            print("   (Pausing 10s to avoid rate limits...)")
            time.sleep(10)
    
    return results


def generate_report(results: Dict[str, Tuple[bool, float]]):
    """Generate a summary report."""
    print_banner("TEST SUMMARY REPORT")
    
    total_tests = len(results)
    passed_tests = sum(1 for success, _ in results.values() if success)
    total_time = sum(duration for _, duration in results.values())
    
    print(f"Total test suites: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, (success, duration) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:<30} {status} ({duration:.1f}s)")
    
    if passed_tests == total_tests:
        print("\n🎉 All system tests passed!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test suite(s) failed")
        return False


def main():
    """Main test runner function."""
    print_banner("CONCEPT GRAPH SYSTEM TEST RUNNER")
    
    # Check if we're in the right directory
    if not Path("concept_graph").exists():
        print("❌ Error: Must run from project root directory")
        print("   Current directory should contain 'concept_graph' folder")
        sys.exit(1)
    
    # Validate environment
    env_status = validate_environment()
    
    if not env_status["can_run_local"]:
        print("\n❌ Cannot run tests: Environment not ready")
        sys.exit(1)
    
    # Get test mode from command line
    mode = "quick"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    if mode not in ["quick", "comprehensive", "full"]:
        print(f"❌ Unknown mode: {mode}")
        print("Usage: python run_system_tests.py [quick|comprehensive|full]")
        print("  quick:        Run fast tests only (default)")
        print("  comprehensive: Run all test suites") 
        print("  full:         Same as comprehensive")
        sys.exit(1)
    
    # Run tests based on mode
    if mode == "quick":
        results = run_quick_tests()
    else:  # comprehensive or full
        results = run_comprehensive_tests()
    
    # Generate report
    success = generate_report(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()