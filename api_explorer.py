import os
import javalang
import json
import chardet
import re

class APIExplorer:
    def __init__(self, package_dir: str):
        self.package_dir = package_dir
        self.api_info = {}
        self.test_info = {}

    def explore_dirs(self, packages_root_dir: str, is_test: bool):
        """
        Explores all packages within a root directory. 
        Each package is expected to have a 'src/main/java/org' structure.
        """
        for root, dirs, files in os.walk(packages_root_dir):
            for dir_name in dirs:
                if is_test:
                # Check for the "src/main/java/org" structure in each package
                    package_dir = os.path.join(root, dir_name, 'src', 'test', 'java', 'org')
                else:
                    package_dir = os.path.join(root, dir_name, 'src', 'main', 'java', 'org')
                    
                if os.path.exists(package_dir):
                    print(f"Exploring package directory: {package_dir}")  # Debugging output
                    self.explore_directory(package_dir, is_test)
        if is_test:
            return self.test_info
        else:
            return self.api_info

    

    def explore_directory(self, dir: str, is_test: bool):
        """
        Explores all .java files within a specific package directory.
        """
        

        for root, dirs, files in os.walk(dir):
            
            for file in files:
                
                if file.endswith(".java"):
                    file_path = os.path.join(root, file)
                    #print(f"Parsing file: {file_path}")  # Debugging output

                    self.parse_java_file(file_path, is_test)
        if is_test:
            return self.test_info
        else:
            return self.api_info
    
    

    
    
    

    def explore_package(self):
        for root, dirs, files in os.walk(self.package_dir):
            for file in files:
                if file.endswith(".java"):
                    file_path = os.path.join(root, file)
                    

        return self.api_info
    
    

    def parse_java_file(self, file_path: str, is_test: bool):
        try:
            # Try opening the file with a default encoding first
            with open(file_path, 'r', encoding='ascii') as file:
                java_code = file.read()
        except UnicodeDecodeError:
            try: 
                with open(file_path, 'r', encoding='ISO-8859-1') as file:
                    java_code = file.read()
            except UnicodeDecodeError:
                # If there's a decoding error, use chardet to detect encoding
                with open(file_path, 'rb') as file: 
                    raw_data = file.read()
                    result = chardet.detect(raw_data)
                    encoding_method = result['encoding']
                    print(f"Detected encoding: {encoding_method}")

                with open(file_path, 'r', encoding=encoding_method) as file:
                    java_code = file.read()

        try:
            tree = javalang.parse.parse(java_code)
            
            self.extract_api_info(tree, file_path, is_test)
        except javalang.parser.JavaSyntaxError as e:
            #print(f"Syntax error in file {file_path}: {e}")
            pass
    
    

        


    def extract_api_info(self, tree, file_path: str, is_test: bool):
        package_path = self.get_package_path(tree)

        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            class_name = node.name
            full_class_path = f"{package_path}.{class_name}"

            if is_test:
                if full_class_path not in self.test_info:
                    self.test_info[full_class_path] = {
                        "methods": {},
                        "file_path": file_path, # Store the file path here
                        "package_name": package_path  # Add the package name here
                    }

                for method in node.methods:
                    method_signature = self.get_method_signature(method)
                    method_definition = self.get_method_definition(file_path, method)

                    method_info = {
                        "return_type": method.return_type.name if method.return_type else "void",
                        "parameters": [(param.type.name, param.name) for param in method.parameters],
                        "modifiers": list(method.modifiers),
                        "signature": method_signature,
                        "definition": method_definition,
                        "package_name": package_path 

                    }
                    self.test_info[full_class_path]["methods"][method.name] = method_info
            else:
                    if full_class_path not in self.api_info:
                        self.api_info[full_class_path] = {
                            "methods": {},
                            "file_path": file_path, # Store the file path here
                            "package_name": package_path  # Add the package name here
                        }

                    for method in node.methods:
                        method_signature = self.get_method_signature(method)
                        method_definition = self.get_method_definition(file_path, method)

                        method_info = {
                            "return_type": method.return_type.name if method.return_type else "void",
                            "parameters": [(param.type.name, param.name) for param in method.parameters],
                            "modifiers": list(method.modifiers),
                            "signature": method_signature,
                            "definition": method_definition,
                            "package_name": package_path 

                        }
                        self.api_info[full_class_path]["methods"][method.name] = method_info

    def get_package_path(self, tree):
        package_path = ""
        for _, node in tree.filter(javalang.tree.PackageDeclaration):
            package_path = node.name
        return package_path

    def get_method_signature(self, method):
        param_str = ", ".join([f"{param.type.name} {param.name}" for param in method.parameters])
        return_type = method.return_type.name if method.return_type else "void"
        return f"{return_type} {method.name}({param_str})"

    

    def get_method_definition(self, file_path: str, method):
        # Attempt to use common encodings first
        try:
            with open(file_path, 'r', encoding='ascii') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='ISO-8859-1') as file:
                    lines = file.readlines()
            except UnicodeDecodeError:
                # Use chardet as a last resort
                with open(file_path, 'rb') as file:
                    raw_data = file.read()
                    result = chardet.detect(raw_data)
                    encoding_method = result['encoding']
                    print(f"Detected encoding: {encoding_method}")
                with open(file_path, 'r', encoding=encoding_method) as file:
                    lines = file.readlines()
        
        method_line = method.position.line - 1  # Adjust for 0-based index
        definition_lines = []

        # Collect method signature and check for opening brace
        while method_line < len(lines):
            line = lines[method_line].strip()
            definition_lines.append(line)
            if '{' in line:
                break
            method_line += 1  # Move to the next line if no '{' found

        # Handle the case where the opening brace is on the next line
        if not '{' in definition_lines[-1]:
            method_line += 1
            while method_line < len(lines) and '{' not in lines[method_line]:
                definition_lines.append(lines[method_line].strip())
                method_line += 1
            # Add the line with the opening brace
            if method_line < len(lines):
                definition_lines.append(lines[method_line].strip())

        # Continue reading lines until we find matching closing brace
        open_braces = sum(line.count('{') for line in definition_lines)
        close_braces = sum(line.count('}') for line in definition_lines)

        # Continue reading lines until we find matching closing brace
        current_line = method_line + 1
        while open_braces > close_braces and current_line < len(lines):
            current_line_text = lines[current_line].strip()
            definition_lines.append(current_line_text)
            open_braces += current_line_text.count('{')
            close_braces += current_line_text.count('}')
            current_line += 1

        # Combine all lines to form the full method definition
        full_definition = "\n".join(definition_lines)
        return full_definition


    def store_api_info(self, output_file: str):
        try:
            # Try opening the file with UTF-8 encoding first
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(self.api_info, file, indent=4)
        except UnicodeEncodeError:
            try:
                # Try with ISO-8859-1 encoding if UTF-8 fails
                with open(output_file, 'w', encoding='ISO-8859-1') as file:
                    json.dump(self.api_info, file, indent=4)
            except UnicodeEncodeError:
                # If there's still a problem, detect the encoding using chardet
                raw_data = json.dumps(self.api_info).encode('utf-8')
                result = chardet.detect(raw_data)
                encoding_method = result['encoding']
                print(f"Detected encoding: {encoding_method}")

                with open(output_file, 'w') as file:
                    json.dump(self.api_info, file, indent=4)


    def store_test_info(self, output_file: str):

        try:
            # Try opening the file with UTF-8 encoding first
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(self.test_info, file, indent=4)
        except UnicodeEncodeError:
            try:
                # Try with ISO-8859-1 encoding if UTF-8 fails
                with open(output_file, 'w', encoding='ISO-8859-1') as file:
                    json.dump(self.test_info, file, indent=4)
            except UnicodeEncodeError:
                # If there's still a problem, detect the encoding using chardet
                raw_data = json.dumps(self.test_info).encode('utf-8')
                result = chardet.detect(raw_data)
                encoding_method = result['encoding']
                print(f"Detected encoding: {encoding_method}")
                
                with open(output_file, 'w') as file:
                    json.dump(self.test_info, file, indent=4)
