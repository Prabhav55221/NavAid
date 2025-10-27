#!/usr/bin/env python3
"""
Quick test script to verify all TTS models can be loaded.

Tests model instantiation and basic synthesis without full evaluation.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import create_all_models, print_model_summary


def test_model_instantiation():
    """Test that all models can be instantiated."""
    print("\n" + "=" * 60)
    print("TEST 1: Model Instantiation")
    print("=" * 60)

    try:
        models = create_all_models()
        print(f"✓ Successfully created {len(models)} model instances")

        for name, model in models.items():
            info = model.get_model_info()
            print(f"  ✓ {info.display_name} ({info.architecture})")

        return True

    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading(model_names=None):
    """
    Test loading models (downloads weights if needed).

    Args:
        model_names: List of model names to test (None = all models)
    """
    print("\n" + "=" * 60)
    print("TEST 2: Model Loading")
    print("=" * 60)

    from models import create_model

    if model_names is None:
        from models import list_available_models
        model_names = list_available_models()

    results = {}

    for name in model_names:
        print(f"\n--- Testing {name} ---")
        try:
            model = create_model(name)
            print(f"Loading model...")
            model.load()
            print(f"✓ {name} loaded successfully")
            results[name] = "success"

        except Exception as e:
            print(f"✗ {name} failed to load: {e}")
            results[name] = f"failed: {e}"

    # Summary
    print("\n" + "=" * 60)
    print("LOADING TEST SUMMARY")
    print("=" * 60)

    success_count = sum(1 for v in results.values() if v == "success")
    total_count = len(results)

    for name, result in results.items():
        status = "✓" if result == "success" else "✗"
        print(f"{status} {name}: {result}")

    print(f"\nPassed: {success_count}/{total_count}")

    return success_count == total_count


def test_synthesis(model_name="espeak"):
    """
    Test basic synthesis with a single model (espeak = fastest).

    Args:
        model_name: Model to test synthesis with
    """
    print("\n" + "=" * 60)
    print(f"TEST 3: Basic Synthesis ({model_name})")
    print("=" * 60)

    try:
        from models import create_model
        import numpy as np

        test_text = "Turn left at the next intersection."

        print(f"Creating {model_name} model...")
        model = create_model(model_name)

        print(f"Loading model...")
        model.load()

        print(f"Synthesizing: '{test_text}'")
        audio, sample_rate = model.synthesize(test_text)

        # Verify output
        assert isinstance(audio, np.ndarray), "Audio should be numpy array"
        assert audio.dtype == np.float32, "Audio should be float32"
        assert audio.ndim == 1, "Audio should be 1D array"
        assert len(audio) > 0, "Audio should not be empty"
        assert sample_rate > 0, "Sample rate should be positive"

        duration_seconds = len(audio) / sample_rate
        print(f"✓ Synthesis successful!")
        print(f"  Audio shape: {audio.shape}")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Duration: {duration_seconds:.2f}s")
        print(f"  Range: [{audio.min():.3f}, {audio.max():.3f}]")

        return True

    except Exception as e:
        print(f"✗ Synthesis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TTS MODEL VALIDATION TESTS")
    print("=" * 60)

    # Print model summary
    print_model_summary()

    # Test 1: Instantiation
    test1_pass = test_model_instantiation()

    if not test1_pass:
        print("\n✗ Instantiation test failed. Stopping.")
        return False

    # Test 2: Loading (optional - can be slow)
    print("\n" + "=" * 60)
    print("NOTE: Model loading test will download weights (~500MB total)")
    print("This may take several minutes on first run.")
    print("=" * 60)

    response = input("\nRun loading test? [y/N]: ").strip().lower()

    if response == 'y':
        # Test a subset first (fastest models)
        print("\nTesting fastest models first...")
        test2_pass = test_model_loading(["espeak"])

        if test2_pass:
            response = input("\nLoad remaining models? [y/N]: ").strip().lower()
            if response == 'y':
                test_model_loading([
                    "coqui_vits_ljspeech",
                    "coqui_vits_vctk",
                    "coqui_tacotron2",
                    "piper"
                ])
    else:
        print("Skipping loading test.")

    # Test 3: Basic synthesis
    test3_pass = test_synthesis("espeak")

    # Summary
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
    print(f"✓ Instantiation: {'PASS' if test1_pass else 'FAIL'}")
    print(f"✓ Synthesis: {'PASS' if test3_pass else 'FAIL'}")

    return test1_pass and test3_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
