import subprocess
import re
import time
import javalang
from typing import List
import os


class TestValidator:

    def __init__(self, package_dir):
        self.package_dir = package_dir
        

    def validate_and_fix(self, test_code: str) -> str:
        """
        Validate and fix the test code by removing markdown artifacts, fixing common issues in the code,
        and ensuring that the code is syntactically correct.
        """
        # Step 1: Remove markdown artifacts and fix basic syntax issues

        test_code = self.clean_generated_code(test_code)
        test_code = self.remove_markdown(test_code)
        test_code = self.fix_import_statements(test_code)
        
        test_code = self.fix_assert_statements(test_code)
        
        

        # Step 2: Check if the code can be parsed by javalang (a Java syntax validator)
        try:
            javalang.parse.parse(test_code)
            return test_code  # Code is valid
        except javalang.parser.JavaSyntaxError as e:
            print(f"Syntax error found: {e}")
            return ""  # Return empty string to indicate failure
        except:
            print("The LLM did not produce code (Nonetype returned).")
            return ""
        
    

    

    import re

    def clean_generated_code(self, test_code):
        """
        Remove everything above and below the backticks and remove the backticks themselves 
        along with any content on the same line.
        """
        # Remove everything above the first occurrence of ```
        test_code = re.sub(r'^.*```[^\n]*\n', '', test_code, flags=re.DOTALL)

        # Remove everything after the last occurrence of ```
        test_code = re.sub(r'```[^\n]*$', '', test_code, flags=re.DOTALL)

        # Remove the rest of the lines containing ```
        test_code = re.sub(r'```[^\n]*', '', test_code)

        # Remove any extra blank lines
        test_code = '\n'.join([line for line in test_code.splitlines() if line.strip()])

        return test_code.strip()



    def remove_markdown(self, test_code: str) -> str:
        """
        Comment out markdown artifacts like ```java or ``` from the generated code.
        Ensure that no actual Java code on the same line is commented out.
        """
        # Replace only the ` ```java ` or ` ``` ` and leave the rest of the line intact
        test_code = re.sub(r"```[^\n]*", " ", test_code)

        # Replace the closing ``` with a comment
        test_code = re.sub(r"```", " ", test_code)

        return test_code

    
    
    

    def fix_import_statements(self, test_code: str) -> str:
        """
        Ensure that common Java package imports are included.
        """
        if 'import org.junit.jupiter.api.' not in test_code:
            test_code = 'import org.junit.jupiter.api.*;\n' + test_code
        if 'import static org.junit.jupiter.api.Assertions.' not in test_code:
            test_code = 'import static org.junit.jupiter.api.Assertions.*;\n' + test_code
        return test_code

    def fix_assert_statements(self, test_code: str) -> str:
        """
        Fix common issues with assert statements, like spacing or missing parentheses.
        """
        return test_code.replace('assertEquals(', 'assertEquals(')
    
    

    def compile_test(self, test_file_path: str) -> dict:
        """
        Compile the Java test file and return a status dictionary with compilation results.
        """
        

        # Step 1: Compile the test file
        compile_result = subprocess.run(
            ["javac", "-cp", "lib/*", test_file_path],  # Adjust the classpath as needed
            capture_output=True,
            text=True
        )

        
        if compile_result.returncode != 0:
            print("Compilation failed with the following error:")
            print(compile_result.stderr)
            if 'package does not exist' in compile_result.stderr:
                missing_packages = self.extract_missing_packages(compile_result.stderr)
                if missing_packages:
                    print(f"Missing packages: {', '.join(missing_packages)}")
                    print("Please install the missing packages before running the tests again.")
                    time.sleep(5)  # Wait for 5 seconds

            # Return status with the error message
            return {"status": "fail", "error": compile_result.stderr}
        else:
            print("Compilation succeeded.")
            # Return status as 'pass' if compilation succeeds
            return {"status": "pass"}
        
    
    

    def delete_jacoco_exec_if_exists(self, jacoco_exec_file: str):
        """
        Deletes the jacoco.exec file if it exists, to avoid any leftover data from previous test runs.
        """
        import os
        if os.path.exists(jacoco_exec_file):
            try:
                os.remove(jacoco_exec_file)
                print(f"Deleted existing {jacoco_exec_file} file.")
            except OSError as e:
                print(f"Error deleting {jacoco_exec_file}: {e}")
        else:
            print(f"{jacoco_exec_file} does not exist, skipping deletion.")

        

    
  
    def run_test(self, test_class_name: str, output_dir: str, method_name: str):
        """
        Runs the test with JaCoCo coverage and returns the coverage result or fail status.
        A coverage report is generated even if the test run fails.
        """
        current_dir = os.getcwd()

        jacoco_agent = os.path.join(current_dir, "lib", "jacocoagent.jar")
        junit_api_jar = os.path.join(current_dir, "lib", "junit-jupiter-api-5.11.0.jar")
        junit_engine_jar = os.path.join(current_dir, "lib", "junit-jupiter-engine-5.11.0.jar")
        hamcrest_jar = os.path.join(current_dir, "lib", "hamcrest-core-1.3.jar")

        commons_lang3_jar = os.path.join(current_dir, "lib", "commons-lang3-3.17.0.jar")
        commons_beanutils_jar = os.path.join(current_dir, "lib", "commons-beanutils-1.9.4.jar")
        commons_cli_jar = os.path.join(current_dir, "lib", "commons-cli-1.9.0.jar")
        commons_codec_jar = os.path.join(current_dir, "lib", "commons-codec-1.17.1.jar")
        commons_collections_jar = os.path.join(current_dir, "lib", "commons-collections4-4.5.0-M2.jar")
        commons_configuration_jar = os.path.join(current_dir, "lib", "commons-configuration2-2.11.0.jar")
        commons_dbcp2_jar = os.path.join(current_dir, "lib", "commons-dbcp2-2.12.0.jar")
        commons_logging_jar = os.path.join(current_dir, "lib", "commons-logging-1.3.4.jar")
        commons_math_jar = os.path.join(current_dir, "lib", "commons-math4-core-4.0-beta1.jar")
        commons_pool_jar = os.path.join(current_dir, "lib", "commons-pool2-2.12.0.jar")
        commons_text_jar = os.path.join(current_dir, "lib", "commons-text-1.12.0.jar")
        commons_validator_jar = os.path.join(current_dir, "lib", "commons-validator-1.9.0.jar")
        commons_io_jar = os.path.join(current_dir, "lib", "commons-io-2.17.0.jar")

        package_jars = [
            os.path.join(current_dir, "lib", "commons-lang3-3.17.0.jar"),
            os.path.join(current_dir, "lib", "commons-beanutils-1.9.4.jar"),
            os.path.join(current_dir, "lib", "commons-cli-1.9.0.jar"),
            os.path.join(current_dir, "lib", "commons-codec-1.17.1.jar"),
            os.path.join(current_dir, "lib", "commons-collections4-4.5.0-M2.jar"),
            os.path.join(current_dir, "lib", "commons-configuration2-2.11.0.jar"),
            os.path.join(current_dir, "lib", "commons-dbcp2-2.12.0.jar"),
            os.path.join(current_dir, "lib", "commons-logging-1.3.4.jar"),
            os.path.join(current_dir, "lib", "commons-math4-core-4.0-beta1.jar"),
            os.path.join(current_dir, "lib", "commons-pool2-2.12.0.jar"),
            os.path.join(current_dir, "lib", "commons-text-1.12.0.jar"),
            os.path.join(current_dir, "lib", "commons-validator-1.9.0.jar"),
            os.path.join(current_dir, "lib", "commons-io-2.17.0.jar")
        ]
        jar_classpath = ";".join(package_jars)

        junit_platform_console_jar = os.path.join(current_dir, "lib", "junit-platform-console-standalone-1.11.0.jar")
        mockito_core_jar = os.path.join(current_dir, "lib", "mockito-core-5.13.0.jar")
        mockito_jupiter_jar = os.path.join(current_dir, "lib", "mockito-junit-jupiter-5.13.0.jar")
        dest_file = os.path.join(current_dir, "jacoco.exec")

        
        run_cmd = [
            "java",
            f"-javaagent:{jacoco_agent}=destfile={dest_file}",
            "-cp", f"{output_dir};{junit_api_jar};{junit_engine_jar};{hamcrest_jar};{commons_lang3_jar};{commons_logging_jar};{commons_validator_jar};{commons_text_jar};{commons_pool_jar};{commons_math_jar};{commons_io_jar};{commons_dbcp2_jar};{commons_beanutils_jar};{commons_configuration_jar};{commons_codec_jar};{commons_collections_jar};{commons_cli_jar};{junit_platform_console_jar};{mockito_core_jar};{mockito_jupiter_jar}",
            "org.junit.platform.console.ConsoleLauncher",
            "--select-class", test_class_name
            ]
        
        self.delete_jacoco_exec_if_exists(dest_file)                     #Delete executable if exists
        

        try:
            try:
                # Run the test and capture output
                result = subprocess.run(run_cmd, capture_output=True, text=True, timeout = 500)
                stdout = result.stdout
                stderr = result.stderr
                
                # Output debug information
                print("\n------Test Output (stdout)------:", stdout)
                #print("Test Errors (stderr):", stderr)
                
                
                
                num_tests_passed, num_tests_failed = self.extract_test_count_from_output(stdout)

                num_tests_found = num_tests_passed + num_tests_failed

            except subprocess.TimeoutExpired:
                # If the command takes longer than 60 seconds, handle the timeout
                print("The test command timed out after 60 seconds.")
                num_tests_failed = num_tests_passed = num_tests_found = 0
                # You can log the issue or handle it as needed, then continue with the next command
                pass

            
            #simple_class_name = test_class_name.split('.')[-1] 
            # Even if some tests fail, proceed to generate the coverage report
            self.generate_coverage_report(dest_file, test_class_name, self.package_dir, jar_classpath)
            

            current_dir = os.getcwd()

            # Define the XML report path relative to the current directory
            report_xml_path = os.path.join(current_dir, "combined_report.xml")


            instruction_coverage, branch_coverage = self.parse_coverage_report(report_xml_path, test_class_name, method_name)
    

            #All tests failed
            if num_tests_passed == 0 and num_tests_failed == num_tests_found:
                return {
                    "status": "fail",  # Indicating that all tests failed
                    "passed_tests": num_tests_passed,
                    "failed_tests": num_tests_failed,
                    "total_tests": num_tests_found,
                    "output": stdout,
                    "instruction coverage": instruction_coverage,
                    "branch coverage": branch_coverage
                }

            # If some tests failed, mark the result as 'partial success'
            if num_tests_failed > 0:
                return {
                    "status": "partial",  # Indicating that some tests passed, but others failed
                    "passed_tests": num_tests_passed,
                    "failed_tests": num_tests_failed,
                    "total_tests": num_tests_found,
                    "output": stdout,
                    "instruction coverage": instruction_coverage,
                    "branch coverage": branch_coverage
                }

            # If all tests passed
            return {
                "status": "pass",
                "passed_tests": num_tests_passed,
                "failed_tests": 0,
                "total_tests": num_tests_found,
                "output": stdout,
                "instruction coverage": instruction_coverage,
                "branch coverage": branch_coverage
            }

        except Exception as e:
            print(f"Error running test with coverage: {e}")
            return {"status": "fail", "error": str(e)}


    def extract_test_count_from_output(self, output: str):
        """
        Extract the number of tests found, passed, and failed from the JUnit test output.
        This function processes the output line by line to capture the test results.
        """
        
        num_tests_passed = 0
        num_tests_failed = 0

        # Iterate through each line in the output
        for line in output.splitlines():
            
            if "tests successful" in line:
                # Extract the number of successful tests
                num_tests_passed = int(re.search(r'(\d+)', line).group(1))
            elif "tests failed" in line:
                # Extract the number of failed tests
                num_tests_failed = int(re.search(r'(\d+)', line).group(1))
            
        
        return num_tests_passed, num_tests_failed
    
    def find_final_source_dirs(self, package_dir, class_name):
        source_dirs = ''

        # Walk through the directory structure starting from the provided package directory
        for root, dirs, files in os.walk(package_dir):
            # Check if we've reached a directory that ends with 'src/main/java'
            if root.endswith(os.path.join('src', 'main', 'java')):
                # Now recursively check all subdirectories to reach the ones with .java files
                for current_root, current_dirs, current_files in os.walk(root):
                    # Check if there are any .java files in the current directory
                    if any(file.endswith('.java') for file in current_files):

                        if f"{class_name}.java" in current_files:
                            source_dirs = current_root

                        

        return source_dirs


    def generate_coverage_report(self, jacoco_exec_file: str, test_class_name: str,  package_dir:str, jar_path:str):
        """
        Generates a JaCoCo coverage report in HTML and XML formats, focused on the specific class.
        """

        current_dir = os.getcwd()
        jacoco_cli =  os.path.join(current_dir, "lib", "jacococli.jar")

        
        simple_class_name = test_class_name.split('_')[-2]
        #print(simple_class_name)
        source_dir = self.find_final_source_dirs(package_dir, simple_class_name)
        

        if not source_dir:
            print("No source directory found .")
            return {"status": "fail", "error": "No source directories found."}
        
        
        source_dir = os.path.normpath(source_dir)
        
        package_jars = jar_path.split(';')
        
        
        combined_xml_path = os.path.join(current_dir, "combined_report.xml")
        if os.path.exists(combined_xml_path):
            os.remove(combined_xml_path)                #Remove if it already exists
            
        print("Generating coverage report...")

            
            
        for jar_file in package_jars:    

                # Command to run the JaCoCo CLI and generate the coverage report
            cmd = [
                    "java", "-jar", jacoco_cli, "report", jacoco_exec_file,
                    "--classfiles", jar_file,  # Include all class files
                    "--sourcefiles", source_dir,  # Include the found source files
                    "--html", os.path.join(current_dir, "report.html"),  # Path to save the HTML report
                    "--xml", os.path.join(current_dir, "partial_report.xml")  # Path to save the XML report
            ]

                # Execute the JaCoCo CLI command
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error generating coverage report for {source_dir}: {result.stderr}")
                return {"status": "fail", "error": result.stderr}
            
            self.merge_coverage_reports(combined_xml_path, os.path.join(current_dir, "partial_report.xml"))

            
        print("Coverage reports generated successfully for all packages.")
        return {"status": "pass"}
    

    def merge_coverage_reports(self, combined_xml_path, new_report_path):
        """
        Merges the contents of the new XML report into the combined XML report.
        """
        import xml.etree.ElementTree as ET
        try:
            # If the combined XML file already exists, parse it; otherwise, create a new root
            if os.path.exists(combined_xml_path):
                combined_tree = ET.parse(combined_xml_path)
                combined_root = combined_tree.getroot()
            else:
                combined_root = ET.Element('report')

            # Parse the new report and get its root
            new_tree = ET.parse(new_report_path)
            new_root = new_tree.getroot()

            # Merge all package elements from the new report into the combined report
            for package_element in new_root.findall('.//package'):
                combined_root.append(package_element)

            # Write the updated combined report back to file
            combined_tree = ET.ElementTree(combined_root)
            combined_tree.write(combined_xml_path)

            

        except Exception as e:
            print(f"Error while merging coverage reports: {e}")
        
    
    def parse_coverage_report(self, report_path: str, test_class_name, method_name: str):
        """
        Parses the JaCoCo XML report and extracts the coverage percentage for the specified method.
        """
        import xml.etree.ElementTree as ET

        
            
        # Load the XML report
        tree = ET.parse(report_path)
        root = tree.getroot()


        
        # Find the coverage element for the method
        for package in root.findall('.//package'):
            for class_ in package.findall('.//class'):
                
                package_name = package.get('name')
                if test_class_name.startswith("Test_"):
                    # the format is "Test_<class_name>_<method_name>, as instructed to the LLM and made sure it is like that"
                    parts = test_class_name.split('_')
                    if len(parts) >= 3:
                        raw_class_name = parts[1]  
                        raw_method_name = parts[2]  

                        # Replace '/' with '.' in the class name
                        formatted_class_name = package_name + "/"+ raw_class_name        #make the class name in the appropriate format
                        

                    
                    if formatted_class_name == class_.get('name'):
                        
                              
                        for method in class_.findall('.//method'):
                            
                            if method.get('name') == method_name:
                                
                                branch_coverage_percentage = None
                                instruction_coverage_percentage = None
                                
                                # Get the instruction counter
                                instruction_counter = method.find('counter[@type="INSTRUCTION"]')
                                if instruction_counter is not None:
                                    covered = int(instruction_counter.get('covered', 0))
                                    missed = int(instruction_counter.get('missed', 0))
                                    total = covered + missed
                                    

                                    if total == 0:
                                        print(f"No instructions found for method {method_name}")
                                        return 0  # Avoid division by zero if no instructions are found

                                    # Calculate coverage percentage
                                    instruction_coverage_percentage = (covered / total) * 100
                                    print(f"Instruction Coverage for method {method_name}: {instruction_coverage_percentage:.2f}%")

                                    
                                
                                branch_counter = method.find('counter[@type="BRANCH"]')
                                #print(branch_counter == None)
                                if branch_counter is not None:
                                    
                                    
                                    covered = int(instruction_counter.get('covered', 0))
                                    missed = int(instruction_counter.get('missed', 0))
                                    total = covered + missed
                                    

                                    if total == 0:
                                        print(f"No branches found for method {method_name}")
                                        return 0  # Avoid division by zero if no branches are found

                                    # Calculate coverage percentage
                                    branch_coverage_percentage = (covered / total) * 100
                                    print(f"Branch Coverage for method {method_name}: {branch_coverage_percentage:.2f}%")

                                    if instruction_coverage_percentage == 0 or branch_coverage_percentage == 0:
                                        continue

                                    

                                    


                            
                                return instruction_coverage_percentage, branch_coverage_percentage
                        
        print(f"No coverage found for method {method_name}")
        return 0,0
 
   
