import javalang
from bs4 import BeautifulSoup

class DocumentationMiner:
    def __init__(self):
        pass

    def find_documentation_for_method_in_file(self, file_path, method_name):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                tree = javalang.parse.parse(content)
                comments = []

                for _, node in tree.filter(javalang.tree.MethodDeclaration):
                    if node.name == method_name:
                        method_start_line = node.position.line
                        comments += self.find_javadoc_comments_near_line(content, method_start_line)
                return comments
            except javalang.parser.JavaSyntaxError as e:
                print(f"Syntax error in file {file_path}: {e}")
                return ''
            

    def find_documentation_for_class_in_file(self, file_path, class_name):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            try:
                tree = javalang.parse.parse(content)
                comments = []

                for _, node in tree.filter(javalang.tree.ClassDeclaration):
                    if node.name == class_name:
                        class_start_line = node.position.line
                        comments += self.find_javadoc_comments_near_line(content, class_start_line)
                return comments
            except javalang.parser.JavaSyntaxError as e:
                print(f"Syntax error in file {file_path}: {e}")
                return ''


    def find_javadoc_comments_near_line(self, content, line_number):
        lines = content.splitlines()
        comments = []

        for i in range(line_number - 2, -1, -1):
            if lines[i].strip().startswith("*/"):
                comment = []
                while i >= 0 and not lines[i].strip().startswith("/**"):
                    comment.append(lines[i].strip())
                    i -= 1
                if i >= 0:
                    comment.append(lines[i].strip())
                    comments.append('\n'.join(reversed(comment)))
                break

            elif lines[i].strip().startswith("*/"):
                comment = []
                while i >= 0 and not lines[i].strip().startswith("/*"):
                    comment.append(lines[i].strip())
                    i -= 1
                if i >= 0:
                    comment.append(lines[i].strip())
                    comments.append('\n'.join(reversed(comment)))
                break
            
            elif lines[i].strip().startswith("//"):
                comments.append(lines[i].strip())

        return comments
    
    def find_package_documentation(self, html_file_path):
        with open(html_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            soup = BeautifulSoup(content, 'html.parser')

            # Find the package description (adjust tags based on the HTML structure)
            description = soup.find('body').get_text(strip=True)

            return description if description else "No package documentation found."
    
    
