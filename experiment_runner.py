import os
import re
import csv
from typing import List
from gemini_llm import GeminiLLM
from test_validator import TestValidator
from documentation_miner import DocumentationMiner
import javalang
import json
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
import chardet


class ExperimentRunner:
    def __init__(self, llm_model: GeminiLLM, api_info=None, max_refinements: int = 3, k: int = 5, csv_file: str = 'prompts_responses.csv', package_dir: str = ''):
        self.llm_model = llm_model
        self.api_info = api_info
        self.package_dir = package_dir
        #self.package_name = package_name
        self.validator = TestValidator(self.package_dir)  
        self.doc_miner = DocumentationMiner()  
        self.max_refinements = max_refinements
        self.k = k  # Number of tries for pass@k metric
        self.csv_file = csv_file

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                writer.writerow(['Prompt', 'Response', 'Pass@k'])  


    import re

    def load_api_info(self, json_path):
        """
        Load and return API information from a JSON file.
        """
        with open(json_path, 'r') as file:
            data = json.load(file)
        return data
    
    def combine_method_info(self, method_info):
        """
        Combine method body and documentation into a single text string.
        """
        method_body = method_info.get('definition', '')
        documentation = method_info.get('documentation', '')
        return f"{documentation}\n{method_body}"
    
    def vectorize_methods(self, api_info):
        """
        Create TF-IDF vectors for each method using both method body and documentation.
        """
        method_texts = []
        method_list = []

        for class_name, class_data in api_info.items():
            for method_name, method_info in class_data['methods'].items():
                combined_text = self.combine_method_info(method_info)
                method_texts.append(combined_text)
                method_list.append({'class_name': class_name, 'method_name': method_name})

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(method_texts)

        return method_list, tfidf_matrix, vectorizer
    
    
    


    def get_method_info(self, api_info, target_class, target_method):
        """
        Retrieve method info from api_info dynamically, based on the class name.
        """
        
        # Attempt to find the full class path based on the provided class name
        full_class_path = None
        for class_path in api_info.keys():
            if class_path.endswith(f".{target_class}"):
                full_class_path = class_path
                break
        
        # If no matching class is found, return an error
        if full_class_path is None:
            print(f"Class {target_class} not found in the API info.")
            return None

        # Proceed with the method lookup
        if target_method in api_info[full_class_path]['methods']:
            return api_info[full_class_path]['methods'][target_method]
        else:
            print(f"Method {target_method} not found in {target_class}.")
            return None
    
    


    def calculate_pass_at_k(self, passed_tests, k, total_tests):
        """
        Calculate pass@k for a single method.
        This is the probability of selecting at least one correct solution in k samples from total_tests.
        """
        import math
        if passed_tests == 0 or total_tests == 0 or k == 0:
            return 0  # No tests passed, or no tests were generated, so pass@k is zero
        
        # If the number of tests is less than k, use the total number of tests as k
        k = min(k, total_tests)
        
        # Calculate the probability that all k selected tests are incorrect
        fail_prob = math.comb(total_tests - passed_tests, k) / math.comb(total_tests, k)

        # pass@k is 1 minus the probability that all k selected tests fail
        return 1 - fail_prob


    


    import re

    def extract_test_output(self, test_definition: str) -> str:
        """
        Extracts key assertions or output from the test definition. Focuses on what the test checks.
        """
        # Simplified version: Extract assertions (e.g., assertEquals, assertTrue, etc.)
        output_lines = []
        for line in test_definition.splitlines():
            if "assert" in line:
                output_lines.append(line.strip())

        return "\n".join(output_lines) if output_lines else "No specific test output found"


    def create_few_shot_prompt(self, prompt: str, few_shot_count: int, test_info: dict, train_set: List[dict]) -> str:
        """
        Create a few-shot prompt by appending examples from the train set to the base prompt.
        """
        few_shot_examples = ""

        prompt += "\nNow I will give you some examples in the format: Class name & method name, method documentation & output(ready unit tests for these methods). Here are some examples:\n"
        
        # Add examples from the train set (few-shot examples)
        for example in train_set[:few_shot_count]:
            example_class = example['class_name']
            example_method = example['method_name']
            #example_docs = doc
            #example_test = test  # Add a test if available, or placeholder

                # Find the corresponding test class and method in test_info
            test_class_name = example_class + "Test"
            test_data = test_info.get(test_class_name, {}).get('methods', {})
            
            test_example = None
            #print(test_data == False)
            if test_data:
                found_test = False
                for test_method_name, test_method_info in test_data.items():
                    if example_method.lower() in test_method_name.lower():  # Attempt to match test to method
                        #print(example_method.lower())
                        #print(test_method_name.lower())
                        test_example = test_method_info.get('definition', 'No test example found')
                        found_test = True
                        break
                
                if not found_test:
                    random_test_method = random.choice(list(test_data.values()))  # Select a random test of the same class if the desired not found
                    test_example = random_test_method.get('definition', 'No test example found')

            

            # Construct few-shot example
            few_shot_examples += (
                f"### Example:\n"
                f"Class: {example_class}\n"
                f"Method Name: {example_method}\n"
                #f"Method Documentation: {example_docs}\n"
                f"Output:\n"
                "```java\n"
                f"{test_example}\n"
                "```\n\n"
            )

        # Append the few-shot examples to the prompt
        prompt += few_shot_examples
        


        return prompt
    
    def replace_class_name_with_filename(self, test_code: str, original_file_path: str) -> str:
        new_class_name = os.path.basename(original_file_path).replace(".java", "")
        class_name_match = re.search(r'\bpublic class (\w+)', test_code)

        if class_name_match:
            old_class_name = class_name_match.group(1)
            updated_code = test_code.replace(old_class_name, new_class_name)
            with open(original_file_path, 'w') as file:
                file.write(updated_code)

            print(f"Class name '{old_class_name}' replaced with '{new_class_name}' in {original_file_path}")
            return updated_code, new_class_name
        else:
            print(f"No class name found in {original_file_path}. No changes made.")
            return test_code, None
    
    
    
    def run_experiment(self,  methods: List[dict], method_count: int, few_shot_count: int,  test_info:dict, output_dir: str, few_shot: bool = False, train_set: List[dict] = None, few_shot_type: str='') -> None:
        
        total_methods_tested = 0
        total_pass_at_k = 0
        total_count_at_n = 0
        total_passing_tests = 0
        total_tests = 0
        instruction_coverage_values = []  # To store coverage for std calculation
        branch_coverage_values = []
        test = ''
        num_methods = len(methods[:method_count])

        methods_above_instruction_threshold = 0
        methods_above_branch_threshold = 0
        coverage_threshold = 70

        for i, method_info in enumerate (methods[:method_count], start = 1):

            method_name = method_info['method_name']
            class_name = method_info['class_name']
            package_name = method_info['package_name'] 
            
            
            total_methods_tested += 1
            
            print(f"Test {i}/{num_methods}")
            
            test = few_shot_type

            prompt = self.create_prompt(package_name, class_name, method_name)

            if few_shot:
                
                prompt = self.create_few_shot_prompt(prompt = prompt, few_shot_count = few_shot_count, test_info = test_info,  train_set = train_set)

            print(f"Prompt before sending to LLM:\n{prompt}\n")

            total_method_tests = 0
            passed_tests = 0
            max_coverage = 0
            max_instruction_coverage = 0 
            max_branch_coverage = 0
            count_at_n = 0  # Initialize count@n as 0
            found_any_successful_test = False
            best_code = None

            for i in range(self.k):  # Assuming self.k is the number of test attempts
                print(f"Test generation attempt {i + 1}/{self.k}")
                
                # Step 1: Generate the test code using the LLM
                test_code = self.llm_model.generate_response(prompt)
                
                # Step 2: Validate and potentially fix the generated code
                validated_code = self.validator.validate_and_fix(test_code)
                
                
                
                if validated_code:
                    # Extract the last part of the class name
                    simple_class_name = class_name.split('.')[-1] 
                    
                    # Step 3: Save the validated test to a file
                    test_file_path = self.save_test(output_dir, simple_class_name, method_name, validated_code)
                    print("Tests generated and saved.")

                    updated_code, new_class_name = self.replace_class_name_with_filename(validated_code, test_file_path)

                    # Step 4: Compile the test
                    test_result = self.validator.compile_test(test_file_path)
                    if test_result["status"] == "pass":
                        print(f"Test passed compilation and saved to: {test_file_path}")

                        # Step 5: Run the compiled test
                        run_result = self.validator.run_test(new_class_name, output_dir, method_name = method_name)

                        passed_tests += run_result.get('passed_tests', 0)  # Add the passed tests from this run
                        total_method_tests += run_result.get('total_tests', 0)  # Ensure total tests count includes all runs

                        current_instruction_coverage = run_result.get("instruction coverage", 0)
                        current_branch_coverage = run_result.get("branch coverage", 0)

                        if current_instruction_coverage is not None:
                            instruction_coverage = current_instruction_coverage
                        else:
                            instruction_coverage = 0  # Treat None as 0

                        if current_branch_coverage is not None:
                            branch_coverage = current_branch_coverage
                        else:
                            branch_coverage = 0  # Treat None as 0
                            
                        total_coverage = instruction_coverage + branch_coverage

                        
                        
                        if (total_coverage > max_coverage):
                            max_total_coverage = total_coverage
                            max_branch_coverage = branch_coverage
                            max_instruction_coverage = instruction_coverage
                            best_code = validated_code                 #keep the best code

                        
                    

                        # Check for partial success or full success
                        if run_result["status"] in ["pass", "partial"]:
                            found_any_successful_test = True
                            if run_result["status"] == "partial":
                                # Extract information about the failing tests and refine the prompt
                                print(f"Partial success: some tests passed, but some failed for {new_class_name}.{method_name}")
                                print(f"Test Branch Coverage for this run: {branch_coverage}%")
                                print(f"Test Instruction Coverage for this run: {instruction_coverage}%")
                                
                                # Refine prompt for the failed tests
                                prompt = self.refine_prompt(prompt, updated_code, run_result.get("output", "No output"))
        
                            if run_result["status"] == "pass":
                                print(f"Test passed and ran successfully: {run_result['output']}")
                                print(f"Test Branch Coverage for this run: {branch_coverage}%")
                                print(f"Test Instruction Coverage for this run: {instruction_coverage}%")
                        else:
                            print(f"Test execution failed for {new_class_name}.{method_name}: {run_result.get('output', 'No output')}")
                            prompt = self.refine_prompt(prompt, updated_code, run_result.get("output", "No output"))
                    else:
                        print(f"Test failed to compile for {new_class_name}.{method_name}: {test_result['error']}")
                        prompt = self.refine_prompt(prompt, updated_code, test_result["error"])
                else:
                    print(f"Failed to generate a valid test for {class_name}.{method_name}")
                    prompt = self.refine_prompt(prompt, test_code, "Validation failed, no valid code produced.")

                if best_code:
                    self.save_test(output_dir, class_name, method_name, best_code)       #we want to keep in the file the best code

            # Summary of the results
           
            
            # Step 6: Calculate pass@k using the correct formula
            pass_at_k = self.calculate_pass_at_k(passed_tests, self.k, total_method_tests)

            

            count_at_n = 1 if found_any_successful_test else 0
            instruction_coverage_values.append(max_instruction_coverage)
            branch_coverage_values.append(max_branch_coverage)


            print(f"\n--- Results for method {method_name} ---")
            if total_method_tests > 0:
                print(f"Passed tests: {passed_tests}/{total_method_tests}, ({(passed_tests / total_method_tests) * 100} %) ")
                print(f"Total tests: {total_method_tests}")
            else:
                print("No tests generated.")

            print(f"Pass@k: {pass_at_k}")
            print(f"Count@n: {count_at_n}")
            print(f"Instruction Coverage: {round(max_instruction_coverage, 2)}")
            print(f"Branch Coverage: {round(max_branch_coverage, 2)}")

            #total_methods_tested += 1
            total_pass_at_k += pass_at_k
            total_count_at_n += count_at_n
            total_passing_tests += passed_tests
            total_tests += total_method_tests
            

         # Track methods with coverage above the threshold
        if max_instruction_coverage >= coverage_threshold:
            methods_above_instruction_threshold += 1
        if max_branch_coverage >= coverage_threshold:
            methods_above_branch_threshold += 1

            # Step 8: Log the prompt, test code, pass@k, count@n, and coverage in the CSV file
            #self.save_to_csv(prompt, test_code, pass_at_k, count_at_n, round(max_branch_coverage / 100, 2), round(max_instruction_coverage / 100, 2))
        if total_methods_tested > 0:
            average_pass_at_k = total_pass_at_k / total_methods_tested
            instruction_coverage_above_threshold_percentage = (methods_above_instruction_threshold / total_methods_tested) * 100
            branch_coverage_above_threshold_percentage = (methods_above_branch_threshold / total_methods_tested) * 100
        else:
            average_pass_at_k = 0
            instruction_coverage_above_threshold_percentage = 0
            branch_coverage_above_threshold_percentage = 0

        if total_tests > 0:
            passing_test_percentage = (total_passing_tests / total_tests) * 100
        else:
            passing_test_percentage = 0
        
        if instruction_coverage_values:
            instruction_coverage_std = np.std(instruction_coverage_values)
        else:
            instruction_coverage_std = 0
        

        if branch_coverage_values:
            branch_coverage_std = np.std(branch_coverage_values)
        else:
            branch_coverage_std = 0
        
        print("\n--- Overall Set Results ---")
        print(f"Total methods tested: {total_methods_tested}")
        print(f"Average pass@k: {average_pass_at_k}")
        print(f"Total count@n: {total_count_at_n}")
        print(f"Passing test %: {round(passing_test_percentage, 3)}%")
        print(f"Percentage of methods with Instruction Coverage ≥ {coverage_threshold}%: {round(instruction_coverage_above_threshold_percentage, 3)}%")
        print(f"Percentage of methods with Branch Coverage ≥ {coverage_threshold}%: {round(branch_coverage_above_threshold_percentage, 3)}%")
        print(f"Instruction coverage standard deviation: {round(instruction_coverage_std, 3)}")
        print(f"Branch coverage standard deviation: {round(branch_coverage_std, 3)}")

        self.save_to_csv(test, package_name, total_methods_tested, round(passing_test_percentage, 3), average_pass_at_k, total_count_at_n, round(instruction_coverage_above_threshold_percentage, 3), round(branch_coverage_above_threshold_percentage, 3), round(instruction_coverage_std, 3), round(branch_coverage_std, 3))
        #self.save_to_csv(test, total_methods_tested, average_pass_at_k, total_count_at_n, round(instruction_coverage_std, 3), round(branch_coverage_std, 3), round(passing_test_percentage, 3))



    def run_one_experiment(self, methods: List[dict], few_shot_count: int, test_info: dict, output_dir: str, few_shot: bool = False, train_set: List[dict] = None) -> None:
        ''' the run_experiment() function in the case we run only for one function'''
    
        total_methods_tested = 0
        
        instruction_coverage_values = []  # To store coverage for std calculation
        branch_coverage_values = []
       
        
        # Handle only the first method in dev_set
        if len(methods) > 0:
            method_info = methods[0]  # We only process one method in the dev set
            method_name = method_info['method_name']
            class_name = method_info['class_name']
            package_name = method_info['package_name']
            
            total_methods_tested += 1
            print(f"Running experiment for method: {method_name} from {class_name} ")

            prompt = self.create_prompt(package_name, class_name, method_name)

            if few_shot:
                prompt = self.create_few_shot_prompt(prompt=prompt, few_shot_count=few_shot_count, test_info=test_info, train_set=train_set)

            print(f"Prompt before sending to LLM:\n{prompt}\n")

            total_method_tests = 0
            passed_tests = 0
            max_instruction_coverage = 0
            max_branch_coverage = 0
            found_any_successful_test = False
            best_code = None

            for i in range(self.k):  # Assuming self.k is the number of test attempts
                print(f"Test generation attempt {i + 1}/{self.k}")

                # Step 1: Generate the test code using the LLM
                test_code = self.llm_model.generate_response(prompt)

                # Step 2: Validate and potentially fix the generated code
                validated_code = self.validator.validate_and_fix(test_code)

                if validated_code:
                    simple_class_name = class_name.split('.')[-1]  # Extract the last part of the class name

                    # Step 3: Save the validated test to a file
                    test_file_path = self.save_test(output_dir, simple_class_name, method_name, validated_code)
                    print("Tests generated and saved.")

                    updated_code, new_class_name = self.replace_class_name_with_filename(validated_code, test_file_path)

                    # Step 4: Compile the test
                    test_result = self.validator.compile_test(test_file_path)
                    if test_result["status"] == "pass":
                        print(f"Test passed compilation and saved to: {test_file_path}")

                        # Step 5: Run the compiled test
                        run_result = self.validator.run_test(new_class_name, output_dir, method_name=method_name)

                        passed_tests += run_result.get('passed_tests', 0)
                        total_method_tests += run_result.get('total_tests', 0)

                        instruction_coverage = run_result.get("instruction coverage", 0) or 0
                        branch_coverage = run_result.get("branch coverage", 0) or 0
                        total_coverage = instruction_coverage + branch_coverage

                        if total_coverage > max_instruction_coverage + max_branch_coverage:
                            max_instruction_coverage = instruction_coverage
                            max_branch_coverage = branch_coverage
                            best_code = validated_code  # Keep the best code

                        if run_result["status"] in ["pass", "partial"]:
                            found_any_successful_test = True
                            if run_result["status"] == "partial":
                                print(f"Partial success for {new_class_name}.{method_name}")
                                prompt = self.refine_prompt(prompt, updated_code, run_result.get("output", "No output"))

                            if run_result["status"] == "pass":
                                print(f"Test passed for {new_class_name}.{method_name}")
                                print(f"Branch Coverage: {branch_coverage}%, Instruction Coverage: {instruction_coverage}%")
                        else:
                            print(f"Test execution failed for {new_class_name}.{method_name}: {run_result.get('output', 'No output')}")
                            prompt = self.refine_prompt(prompt, updated_code, run_result.get("output", "No output"))
                    else:
                        print(f"Test failed to compile for {new_class_name}.{method_name}: {test_result['error']}")
                        prompt = self.refine_prompt(prompt, updated_code, test_result["error"])
                else:
                    print(f"Failed to generate a valid test for {class_name}.{method_name}")
                    prompt = self.refine_prompt(prompt, test_code, "Validation failed, no valid code produced.")

            if best_code:
                self.save_test(output_dir, class_name, method_name, best_code)

            # Summary of the results
            pass_at_k = self.calculate_pass_at_k(passed_tests, self.k, total_method_tests)
            count_at_n = 1 if found_any_successful_test else 0

            instruction_coverage_values.append(max_instruction_coverage)
            branch_coverage_values.append(max_branch_coverage)

            print(f"\n--- Results for method {method_name} ---")
            if total_method_tests > 0:
                print(f"Passed tests: {passed_tests}/{total_method_tests}, ({(passed_tests / total_method_tests) * 100}%)")
            else:
                print("No tests generated.")

            print(f"Pass@k: {pass_at_k}")
            print(f"Count@n: {count_at_n}")
            print(f"Instruction Coverage: {round(max_instruction_coverage, 2)}")
            print(f"Branch Coverage: {round(max_branch_coverage, 2)}")

            

    

        


     #self.save_to_csv(test, package_name, total_methods_tested, round(passing_test_percentage, 3), average_pass_at_k, total_count_at_n, round(instruction_coverage_above_threshold_percentage, 3), round(branch_coverage_above_threshold_percentage, 3), round(instruction_coverage_std, 3), round(branch_coverage_std, 3))
    def save_to_csv(self, test: str, package_name:str, total_methods_tested: int, passing_test_percentage: float, pass_at_k: float, count_at_n: int, instruction_coverage_above_threshold_percentage: float, branch_coverage_above_threshold_percentage: float,instruction_coverage_std:float, branch_coverage_std: float):
        with open(self.csv_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow([test, package_name, total_methods_tested, passing_test_percentage, pass_at_k, count_at_n, instruction_coverage_above_threshold_percentage, branch_coverage_above_threshold_percentage, instruction_coverage_std, branch_coverage_std])
            csvfile.flush()


    def replace_class_name_with_filename(self, test_code: str, original_file_path: str) -> str:
        new_class_name = os.path.basename(original_file_path).replace(".java", "")
        class_name_match = re.search(r'\bpublic class (\w+)', test_code)

        if class_name_match:
            old_class_name = class_name_match.group(1)
            updated_code = test_code.replace(old_class_name, new_class_name)

            try:
                with open(original_file_path, 'w', encoding='utf-8') as file:
                    file.write(updated_code)
            except UnicodeEncodeError:
                try:
                    with open(original_file_path, 'w', encoding='ISO-8859-1') as file:
                        file.write(updated_code)
                except UnicodeEncodeError:
                    raw_data = json.dumps(self.test_info).encode('utf-8')
                    result = chardet.detect(raw_data)
                    encoding_method = result['encoding']
                    print(f"Detected encoding: {encoding_method}")

                    with open(original_file_path, 'w', encoding = encoding_method):
                        file.write(updated_code)



            print(f"Class name '{old_class_name}' replaced with '{new_class_name}' in {original_file_path}")
            return updated_code, new_class_name
        else:
            print(f"No class name found in {original_file_path}. No changes made.")
            return test_code, None
        
    
    
    

    
    
    def create_prompt(self, package_name: str, class_name: str, method_name: str) -> str:
        
        method_info = None
        full_class_path = f"{class_name}"

        print(full_class_path)

        #package_docs = self.doc_miner.find_package_documentation(os.path.join(package_dir, "package.html"))

        # Extract the last part of the class name
        simple_class_name = class_name.split('.')[-1]  

        # A clear, simplified, and concise task definition to focus the model's effort on what matters most
        prompt = (
            f"As a highly experienced Java developer, your task is to generate a thorough, well-structured JUnit5 test suite for the method '{method_name}' "
            f"in the class '{simple_class_name}' from the '{package_name}' package. " 
            f" IMPORTANT!! Your response should contain EXCLUSIVELY the java code and NOTHING else. Not any comments from you, not anything. I will consider your response as a "
            f"valid java executable, that's why I want only the code in your response."
            f"\n\n### Chain of Thought - Steps: ###\n"
            f"1. Understand the method's purpose and functionality.\n"
            f"2. Identify critical inputs, edge cases, and potential failure scenarios.\n"
            f"3. Design tests that cover both typical and edge cases, including invalid inputs and exceptions.\n"
            f"4. Use appropriate assertions to validate the method's behavior.\n"
            f"Documentation, method details, and test requirements are outlined below."
            f"\n\n"
            f"### Method Overview ###\n"
            f"Package: {package_name}\n"
            f"Class: {simple_class_name}\n"
            f"Method: {method_name}\n"
        )

        #if package_docs:
            #prompt += f"\n### Package Documentation ###\n{package_docs}\n\n"

        
        
        # Add available documentation to give additional context to the LLM
        if self.api_info and full_class_path in self.api_info:
            if method_name in self.api_info[full_class_path]["methods"]:
                method_info = self.api_info[full_class_path]["methods"][method_name]
                file_path = self.api_info[full_class_path]['file_path']

                docs = self.doc_miner.find_documentation_for_method_in_file(file_path, method_name)
                class_docs = self.doc_miner.find_documentation_for_class_in_file(file_path, class_name)

                if class_docs:
                    prompt += f"\n### Class Documentation ###\n" + "\n".join(class_docs)
                    #print("no doc")
                if docs:
                    prompt += f"\n### Method Documentation ###\n" + "\n".join(docs)
                    #print("no doc")

        # Add method signature and definition to provide precise information
        if method_info:
            prompt += f"\nMethod Signature: {method_info['signature']}"
            prompt += f"\nMethod Definition: {method_info['definition']}"
            #print("no doc")
        

        

        


        prompt += (
            f"\n\n### Test Generation Guidelines ###\n"
            f"1. **Class Name**: 'Test_{simple_class_name}_{method_name}.java'.\n"
            f"2. **Coverage**: Include edge cases (nulls, invalid values) and different execution paths (success, failure).\n"
            f"3. **Assertions**: Use assertTrue, assertFalse, assertEquals for valid, invalid, and edge cases.\n"
            f"4. **Paths**: Cover all branches and exceptions for maximum coverage.\n"
            f"5. **JUnit5**: Follow JUnit5 with @Test, @BeforeEach, @AfterEach.\n"
            f"6. **Correctness**: Ensure the test is executable, error-free, and follows best practices.\n"
        )

        return prompt
    

    

    

    def refine_prompt(self, original_prompt: str, generated_code: str, error_message: str) -> str:
        refined_prompt = (
            #original_prompt + "\n\n"
            "The previous attempt to generate a JUnit test encountered the following issue:\n"
            f"Error Message: {error_message}\n"
            "\nPlease refine the unit test based on the issues above, and ensure the test now compiles and runs correctly."
            f" IMPORTANT!! Your response should contain EXCLUSIVELY the java code and NOTHING else. Not any comments from you, not anything. I will consider your response as a"
            f" valid java executable, that's why I want only the code in your response."
            "\n"
            "The code that provoked the error is (Counter Example - AVOID IT!):\n"
            f"{generated_code}\n"
            
        )
        #print(f"Refined Prompt:\n{refined_prompt}\n")
        return refined_prompt

    

    def save_test(self, output_dir: str, class_name: str, method_name: str, test_code: str) -> str:
        """Saves the generated test code to a file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        test_file_path = os.path.join(output_dir, f"Test_{class_name}_{method_name}.java")
        
        # Ensure test_code is a string (if it's a tuple, we need the first element)
        if isinstance(test_code, tuple):
            test_code = test_code[0]  # Extract the first element of the tuple
        try:
            with open(test_file_path, 'w') as test_file:
                test_file.write(test_code)
        except UnicodeEncodeError:
            print("Encoding error occurred with UTF-8. Attempting to detect encoding...")
        
            # If there is an error, you can determine the best encoding to use
            # For demonstration, let's assume you have a fallback encoding here.
            # Alternatively, you could use the chardet module to find an encoding.

            # Detect encoding of test_code if needed (optional)
            raw_data = test_code.encode('utf-8', errors='ignore')  # Safely encode and ignore errors
            result = chardet.detect(raw_data)
            encoding_method = result['encoding']
            print(f"Detected encoding: {encoding_method}")

            # Write with the detected encoding
            with open(test_file_path, 'w', encoding=encoding_method) as test_file:
                test_file.write(test_code)

        
        
        return test_file_path
    
    
