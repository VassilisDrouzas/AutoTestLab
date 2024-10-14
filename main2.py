import os
import random
import argparse
from api_explorer import APIExplorer
from experiment_runner import ExperimentRunner
import json
from llama_llm import LLamaLLM
from gemini_llm import GeminiLLM


import random
from collections import defaultdict
import pickle
import time


def save_fixed_evaluation_methods(file_name, data):
    """Save the selected evaluation methods to a file."""
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_fixed_evaluation_methods(file_name):
    """Load the saved fixed evaluation methods from the file if it exists."""
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    return None



def write_method_split_to_file(file_name, split_data):

    if os.path.exists(file_name):
        os.remove(file_name)
    """Write the methods used for evaluation and training (same and different package) to a human-readable file."""
    with open(file_name, 'w') as f:
        for package_name, data in split_data.items():
            f.write(f"Package: {package_name}\n")
            
            # Write the development set (methods under test)
            f.write("Development Set (Evaluation):\n")
            for method in data['dev_set']:
                f.write(f"  - {method['class_name']}.{method['method_name']}\n")
            
            # Write the few-shot examples from the same package
            f.write("\nTraining Set (Few-shot from the Same Package):\n")
            for method in data['train_set_same_package']:
                f.write(f"  - {method['class_name']}.{method['method_name']}\n")
            
            # Write the few-shot examples from a different package
            f.write("\nTraining Set (Few-shot from a Different Package):\n")
            for method in data['train_set_diff_package']:
                f.write(f"  - {method['class_name']}.{method['method_name']}\n")
            
            f.write("\n" + "-" * 40 + "\n\n")



def split_sets(api_info, dev_method_count, seed=None, fixed_file=None, split_output_file="method_split.txt"):
    # Load fixed evaluation methods if they exist
    if fixed_file:
        fixed_evaluation_methods = load_fixed_evaluation_methods(fixed_file)
        if fixed_evaluation_methods:
            
            return fixed_evaluation_methods

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Organize methods by package
    methods_by_package = defaultdict(list)

    # Populate methods by their respective packages
    for class_name, class_data in api_info.items():
        
        for method_name, method_info in class_data['methods'].items():
            package_name = method_info['package_name']
            
            methods_by_package[package_name].append({
                'package_name': package_name,
                'class_name': class_name,
                'method_name': method_name,
                'info': method_info
            })

    # Shuffle packages for randomization
    packages = list(methods_by_package.keys())
    random.shuffle(packages)

    #print(packages)

    # Process each package individually
    evaluation_methods_by_package = {}
    for selected_package in packages:
        print(f"Processing package: {selected_package}")

        train_dev_methods = methods_by_package[selected_package]
        total_methods = len(train_dev_methods)

        # Check if the number of methods is less than the dev_method_count
        if total_methods < (dev_method_count + 5):  # +5 because we need some examples for few-shot
            print(f"Warning: Package '{selected_package}' has only {total_methods} methods, fewer than the requested {dev_method_count}.")
            
            train_size = int(total_methods * 0.4)
            dev_size = total_methods - train_size
        else:
            # Normal case: use requested dev_method_count
            train_size = total_methods - dev_method_count
            dev_size = dev_method_count

        # Shuffle methods in the selected package
        random.shuffle(train_dev_methods)

        # Split methods into development set (evaluation) and training set (for few-shot learning)
        dev_set = train_dev_methods[:dev_size]  # First N methods for development (fixed)
        train_set_same_package = train_dev_methods[dev_size:]  # Remaining methods for few-shot learning from the same package

        # Pick another package for few-shot examples from a different package
        few_shot_package = packages[0] if selected_package != packages[0] else packages[1]
        few_shot_set_different_package = methods_by_package[few_shot_package][:len(train_set_same_package)]  # Take few-shot examples from another package
        mixed_few_shot_set = train_set_same_package[:len(train_set_same_package)//2] + few_shot_set_different_package[:len(train_set_same_package)//2]

        # Store the selected methods for evaluation (development set) and training
        evaluation_methods_by_package[selected_package] = {
            'train_set_same_package': train_set_same_package,
            'train_set_diff_package': few_shot_set_different_package,
            'mixed_few_shot_set': mixed_few_shot_set,
            'dev_set': dev_set
        }

    # Save the selected evaluation methods to a file for future runs
    if fixed_file:
        save_fixed_evaluation_methods(fixed_file, evaluation_methods_by_package)

    # Write method splits to a file for review
    write_method_split_to_file(split_output_file, evaluation_methods_by_package)

    return evaluation_methods_by_package



def load_json_file(file_path):
    """Utility function to load a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoTestLab - Generate and Validate Unit Tests")
    parser.add_argument('--package_dir', required=True, help="Base directory where all the package files are located.")
    parser.add_argument('--selected_model', required=True, choices=['gemini', 'llama'], help="Choose the LLM model for test generation.")
    parser.add_argument('--few_shot', action='store_true', help="Enable few-shot learning")
    parser.add_argument('--few_shot_count', type=int, default=2, help="Number of examples to include in the few-shot prompt")
    parser.add_argument('--dev_method_count', type=int, default=1, help="Number of methods for the development set")
    parser.add_argument('--num_runs', type=int, default=50, help="Number of times to run the experiment")
    parser.add_argument('--fixed_methods_file', help="File to save/load the fixed evaluation methods")


    args = parser.parse_args()

    explorer = APIExplorer(args.package_dir)

    

    # Check if the required JSON files already exist
    api_info_file = os.path.join(os.getcwd(), "api_info.json")
    test_info_file = os.path.join(os.getcwd(), "api_tests.json")

    if os.path.exists(api_info_file) and os.path.exists(test_info_file):
        api_info = load_json_file(api_info_file)
        test_info = load_json_file(test_info_file)

    else:
        api_info = explorer.explore_dirs(args.package_dir, False)
        test_info = explorer.explore_dirs(args.package_dir, True)

        # Store the explored API and test information into JSON files
        explorer.store_api_info("api_info.json")
        explorer.store_test_info("api_tests.json")

    start_time = time.time()

    for run in range(args.num_runs):
        print(f"Running execution {run + 1}/{args.num_runs}...")

        

        package_results = split_sets(
            api_info,
            dev_method_count=args.dev_method_count,
            seed=1,  # Ensure reproducibility with a seed
            fixed_file=args.fixed_methods_file  # Save/load the methods split from a file
        )

        
        

        # Select the model (Gemini or Llama)
        if args.selected_model == "gemini":
            
            gemini_api_key = os.getenv("GEMINI_API_KEY")
            llm_model = GeminiLLM(
                api_url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
                api_key=gemini_api_key
            )
        elif args.selected_model == "llama":
            
            llama_api_key = os.getenv("LLAMA_API_KEY")
            llm_model = LLamaLLM(api_key=llama_api_key)

        # Initialize ExperimentRunner
        runner = ExperimentRunner(llm_model, api_info=api_info, package_dir=args.package_dir)

        
        
        for package_num, package_data in enumerate(package_results.items(), start=1):                #few shot from the same package
            selected_package, data = package_data
            train_set_same_package, dev_set = data['train_set_same_package'], data['dev_set']

            print(f"\n------ Running for package {selected_package} (few shot from the same package)-------------")

            # Run experiment on the dev set with few-shot learning
            runner.run_experiment(
                methods=dev_set,
                method_count=args.dev_method_count,
                few_shot_count=args.few_shot_count,
                test_info=test_info,
                output_dir="generated_tests",
                few_shot=args.few_shot,
                train_set=train_set_same_package,  # Pass the appropriate few-shot set
                few_shot_type = 'Same package'
            )
            

        for package_num, package_data in enumerate(package_results.items(), start=1):   #few shot from different package
            selected_package, data = package_data
            train_set_diff_package, dev_set = data['train_set_diff_package'], data['dev_set']

            print(f"\n------ Running for package {selected_package} (few shot from different package) -------------")

            # Run experiment on the dev set with few-shot learning
            runner.run_experiment(
                methods=dev_set,
                method_count=args.dev_method_count,
                few_shot_count=args.few_shot_count,
                test_info=test_info,
                output_dir="generated_tests",
                few_shot=args.few_shot,
                train_set=train_set_diff_package,  # Pass the appropriate few-shot set
                few_shot_type = 'Different package'
            )
        
        for package_num, package_data in enumerate(package_results.items(), start=1):                #mixed few shot
            selected_package, data = package_data
            train_set_mixed, dev_set = data['mixed_few_shot_set'], data['dev_set']

            print(f"\n------ Running for package {selected_package} (mixed few shot)-------------")

            # Run experiment on the dev set with few-shot learning
            runner.run_experiment(
                methods=dev_set,
                method_count=args.dev_method_count,
                few_shot_count=args.few_shot_count,
                test_info=test_info,
                output_dir="generated_tests",
                few_shot=args.few_shot,
                train_set=train_set_mixed,  # Pass the appropriate few-shot set
                few_shot_type = 'Mixed few-shot'
            )
            
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.4f} seconds")

   
