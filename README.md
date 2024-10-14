# AutoTestLab 

AutoTestLab is a test generation framework designed to automate the process of generating, validating, and running unit tests for Java projects using various language models (e.g., Gemini, Llama). The app extracts documentation from Java source files, generates tests using LLMs, and evaluates those tests based on metrics like coverage, correctness, and pass rate.

## Project Structure

### 1. **main2.py**
This script performs benchmarking tests on randomly selected **whole** packages. It orchestrates the entire process, from the selection of a package to the generation, validation, and evaluation of the unit tests.

- **Main Methods**:
  - `fetch_generated_test(output_dir: str, class_name: str, method_name: str)`: Retrieves generated test files based on the class and method.
  - `run_experiment(package_name: str, methods: List[str], output_dir: str, few_shot: bool, train_set)`: Runs the core experiment for generating and evaluating tests on all functions in the selected package.

### 2. **main3.py**
This script allows the user to specify a **particular function** in a package to be tested. The function specified by the user is placed in the development set, while 20 functions from the same package are placed in the training set and 1 random function from a different package forms the test set.

- **Main Methods**:
  - `select_methods_for_dev(methods, count)`: Allows the user to input methods for the dev set.
  - `run_experiment_for_function(function_name: str, package_name: str, few_shot_count: int)`: Runs the experiment on the user-specified function, generating and evaluating tests.

### 3. **experiment_runner.py**
This class is responsible for orchestrating the entire test generation process. It interfaces with the LLM to generate the tests, validates the generated tests, and runs them to collect metrics.

- **Main Methods**:
  - `run_experiment(package_name: str, methods: List[str], output_dir: str, few_shot: bool, train_set)`: Runs the core experiment for generating and evaluating tests.
  - `create_prompt(package_name: str, class_name: str, method_name: str)`: Generates the prompt used to instruct the LLM on how to generate the test.
  - `generate_coverage_report(jacoco_exec_file, class_name, package_jar, method_name)`: Generates the JaCoCo coverage report.

### 4. **test_validator.py**
This class is responsible for validating and running the generated tests, ensuring they are functional, compilable, and executable.

- **Main Methods**:
  - `validate_and_fix(test_code: str)`: Validates and potentially fixes syntax errors in the generated test code.
  - `compile_test(test_file_path: str)`: Compiles the generated Java test file and checks for errors.
  - `run_test(test_class_name: str, output_dir: str, method_name: str)`: Runs the test using JUnit and JaCoCo to collect coverage and test results.

### 5. **api_explorer.py**
This class is used to explore Java packages and extract API information, including method signatures and documentation.

- **Main Methods**:
  - `explore_package()`: Recursively explores a package directory and extracts class and method information.
  - `store_api_info(filename: str)`: Stores the extracted API information into a JSON file for later use.

### 6. **documentation_miner.py**
This class is responsible for extracting Javadoc comments and documentation from Java source files to provide additional context for test generation.

- **Main Methods**:
  - `find_documentation_for_class_in_file(file_path: str, class_name: str)`: Extracts documentation for a specific class.
  - `find_documentation_for_method_in_file(file_path: str, method_name: str)`: Extracts documentation for a specific method.

### 7. **gemini_llm.py**
This class defines the Gemini LLM and interacts with its API to generate Java unit tests.

- **Main Methods**:
  - `generate_response(prompt: str)`: Sends the prompt to the Gemini API and retrieves the generated test code.

### 8. **llama_llm.py**
This class defines the Llama LLM, which also generates Java unit tests based on the provided prompt.

- **Main Methods**:
  - `generate_response(prompt: str)`: Sends the prompt to the Llama API and retrieves the generated test code.

## Evaluation Process

AutoTestLab evaluates the generated tests in three key stages:

### 1. Compilation
After generating the JUnit5 test code using the LLM, the code is compiled using the `javac` compiler. If there are any errors, AutoTestLab attempts to fix them using its built-in `validate_and_fix` method.

### 2. Test Execution
Once the test is compiled successfully, it is run using JUnit5. The results of the test run are captured, including the number of tests passed and failed.

### 3. Coverage Evaluation
For each compiled and run test, AutoTestLab generates a code coverage report using the JaCoCo library. The coverage report shows the percentage of code covered by the generated tests.

#### Coverage Metrics
The coverage is reported in terms of percentage, showing how much of the method's logic was exercised by the generated test cases.

#### Generalization
The test set is used to evaluate the generalization of the LLM. Tests are run on methods that were not part of the few-shot learning process to see how well the model can generalize its test generation capabilities.

---

## How Evaluation Works

### Development Set
In **main3.py**, the dev set is formed based on the user-specified function. This function is refined and optimized through iterative testing processes. The model is refined until a good set of tests is generated. This serves as an internal evaluation phase.



## Packages Used

The following packages were used for testing the effectiveness of the generated tests:

| Package Name           | Version            |
|------------------------|--------------------|
| **commons-lang3**       | 3.17.0             |
| **commons-beanutils**   | 1.9.4              |
| **commons-cli**         | 1.9.0              |
| **commons-codec**       | 1.17.1             |
| **commons-collections4**| 4.5.0-M2           |
| **commons-configuration2**| 2.11.0           |
| **commons-dbcp2**       | 2.12.0             |
| **commons-logging**     | 1.3.4              |
| **commons-math4-core**  | 4.0-beta1          |
| **commons-pool2**       | 2.12.0             |
| **commons-text**        | 1.12.0             |
| **commons-validator**   | 1.9.0              |
| **commons-io**          | 2.17.0             |


---

## Running AutoTestLab

### Prerequisites
1. Python 3.8+
2. Java JDK
3. Some required Python libraries (in the *requirements.txt* file).
Install with: 
```bash
pip install -r requirements.txt
```
4. JaCoCo for test coverage analysis (already configured in the project).
5. LLM API Keys:
   - For **Gemini**, set the API key in your environment:
     ```bash
     export GEMINI_API_KEY="your_api_key_here"
     ```
   - For **Llama**, set the API key in your environment:
     ```bash
     export LLAMA_API_KEY="your_api_key_here"
     ```

---

### Running `main2.py` (Random Package Benchmarking)

#### Purpose:
`main2.py` is used to run benchmarking tests on **randomly selected methods** from the collection of the 13 packages. The number of methods selected from each package was based on the number of subdirectories within the package, ensuring that one method was randomly chosen from each subdirectory.



#### Command Example:
```bash
python main2.py --package_dir <path_to_package> --selected_model <gemini|llama> --few_shot --few_shot_count <number_of_examples> --dev_method_count <number_of_methods_for_dev_set> --num_runs <number_of_runs> --fixed_methods_file <fixed_methods_pkl_file>
```

#### Arguments:
--**package_dir**: Specifies the path to the package directory.

--**selected_model**: Choose between 'gemini' or 'llama' as the LLM for test generation.

--**few_shot**: Enables few-shot learning (omit this for zero-shot learning).

--**few_shot_count**: Number of examples to include in the few-shot learning prompt.

--**dev_method_count**: Number of methods to be tested in the development set.

--**num_runs**: Number of runs to do (repeated process).

--**fixed_methods_file**: The file containing the already made method split for repeatability (skipping this will generate a new one)

### Running `main3.py` (User-Specified Function Testing)
#### Purpose:
main3.py allows the user to specify a particular function from a package to be tested. The user-selected function is placed in the development set, and the system generates tests specifically for this method. The training set consists of 20 other functions from the same package, and one random function from a different package forms the test set.

#### Command example:
```bash
python main3.py --package_dir <path_to_package> --selected_model <gemini|llama> --few_shot --few_shot_count <number_of_examples> --package_name <package_name> --function_name <function_to_test>
```

#### Arguments:
--**package_dir**: Specifies the path to the package directory.

--**selected_model**: Choose between 'gemini' or 'llama' as the LLM for test generation.

--**few_shot**: Enables few-shot learning (omit this for zero-shot learning).

--**few_shot_count**: Number of examples to include in the few-shot learning prompt.

--**package_name**: Name of the package containing the function to be tested (e.g., org.apache.commons.lang3).

--**function_name**: Name of the specific function to test (e.g., equals).



