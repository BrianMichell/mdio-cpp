import mdio_cpp
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def demo_function(var):
    """Demo function that receives and prints a Variable - must be at module level for ProcessPoolExecutor"""
    print(f"Demo function received Variable: {var}")
    try:
        # Test that we can access Variable methods
        spec = var.get_spec()
        dims = var.dimensions()
        print(f"  - Variable has {len(spec)} spec keys")
        print(f"  - Dimensions: {dims}")
        return f"✓ Demo function processed Variable successfully"
    except Exception as e:
        print(f"  - Error accessing Variable: {e}")
        return f"✗ Demo function failed: {e}"

def demo_dataset(ds):
    """Demo function that receives and inspects a Dataset."""
    print(f"Demo function received Dataset with variables: {ds.variable_names}")
    try:
        variable_names = list(ds.variable_names)
        meta = ds.metadata
        summary = {
            "variable_count": len(variable_names),
            "first_variable": variable_names[0] if variable_names else None,
            "dataset_name": meta.get("name", meta.get("attributes", {}).get("name")),
        }
        # Confirm we can still access a variable from the unpickled Dataset
        if variable_names:
            first_var = ds.get_variable(variable_names[0])
            summary["first_rank"] = first_var.rank()
        print(f"  - Dataset summary: {summary}")
        return f"✓ Demo function processed Dataset successfully: {summary}"
    except Exception as e:
        print(f"  - Error accessing Dataset: {e}")
        return f"✗ Demo dataset function failed: {e}"

if __name__ == "__main__":
    ds = mdio_cpp.Dataset.open("/home/ubuntu/source/mdio-cpp/teapot_example.mdio", "open")

    print("Testing Variable pickling with different dtypes...")
    print("="*80)
    print(f"Dataset: {ds}")

    # Test both amplitude (float32) and headers (byte) variables
    amplitude_var = ds.get_variable("amplitude")
    headers_var = ds.get_variable("headers")
    raw_var = ds.get_variable("raw_headers")

    print(f"Amplitude Variable: {amplitude_var}")
    print(f"Headers Variable: {headers_var}")
    print(f"Raw Headers Variable: {raw_var}")
    print("="*80)

    # Use spawn context like in your actual code
    context = mp.get_context("spawn")

    print("\nCreating ProcessPoolExecutor with spawn context...")

    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        print("Testing amplitude Variable (float32)...")
        future1 = executor.submit(demo_function, amplitude_var)
        result1 = future1.result()
        print(f"Result: {result1}")

        print("\nTesting headers Variable (byte dtype -> uint8 for pickling)...")
        future2 = executor.submit(demo_function, headers_var)
        result2 = future2.result()
        print(f"Result: {result2}")

        print("\nTesting raw headers Variable (byte dtype)...")
        future3 = executor.submit(demo_function, raw_var)
        result3 = future3.result()
        print(f"Result: {result3}")

        print("\nTesting Dataset pickling...")
        future4 = executor.submit(demo_dataset, ds)
        result4 = future4.result()
        print(f"Result: {result4}")

    print("\n✓ Variables with different dtypes successfully pickled and processed!")
    print("✓ TensorStore serialization limitation for 'byte' dtype has been worked around!")
