import os
import random
import argparse
from api_explorer import APIExplorer
from experiment_runner import ExperimentRunner
from test_validator import TestValidator
import json
from gemini_llm import GeminiLLM
from llama_llm import LLamaLLM
import random
from collections import defaultdict



def split_sets(api_info, target_package, target_function):
    methods_by_package = defaultdict(list)

    # Organize methods by package
    for class_name, class_data in api_info.items():
        package_name = class_name.rsplit('.', 1)[0]  # Extract package name
        for method_name, method_info in class_data['methods'].items():
            methods_by_package[package_name].append({
                'package_name': package_name,
                'class_name': class_name,
                'method_name': method_name,
                'info': method_info
            })

    # Ensure the target function is placed in the dev set
    train_set = []
    dev_set = []
    test_set = []

    # Collect methods from the same package as the target function
    if target_package in methods_by_package:
        #print("PACKAGE FOUND!!!")
        package_methods = methods_by_package[target_package]
        for method in package_methods:
            if method['method_name'] == target_function:
                #print("METHOD FOUND!!!")
                dev_set.append(method)
            else:
                
                train_set.append(method)

        # Ensure the training set has at most 20 methods
        train_set = train_set[:20]
    else:
        raise ValueError(f"Package {target_package} not found.")

    

    print(f"Training set ({len(train_set)} methods):")
    for method in train_set:
        print(f"  - Class: {method['class_name']}, Method: {method['method_name']}")

    print(f"\nDev set ({len(dev_set)} methods):")
    for method in dev_set:
        print(f"  - Class: {method['class_name']}, Method: {method['method_name']}")

    test_set = None

    return train_set, dev_set, test_set



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
        parser.add_argument('--package_name', required=True, help="Package name for the function under test")
        parser.add_argument('--function_name', required=True, help="Function name to keep in the dev set")
        

        args = parser.parse_args()

    

        package_dir = args.package_dir
        explorer = APIExplorer(package_dir)
       
        

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


        # Split sets based on user inputs
        train_set, dev_set, test_set = split_sets(api_info, args.package_name, args.function_name)

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

        # Run experiment on the dev set
        print("\n------ Running for development set (few-shot learning) -------------")
        runner.run_one_experiment(
            methods=dev_set,
            few_shot_count=args.few_shot_count,
            test_info=test_info,
            output_dir="generated_tests",
            few_shot=args.few_shot,
            train_set=train_set
        )

       
   