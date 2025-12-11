import tree_sitter_java
from tree_sitter import Language, Parser


class JavaParser:
    def __init__(self):
        self.JAVA_LANGUAGE = Language(tree_sitter_java.language())
        self.parser = Parser()
        self.parser.set_language(self.JAVA_LANGUAGE)

    def parse(self, code: str) -> dict:
        tree = self.parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        functions = []
        classes = []
        imports = []
        issues = []
        complexity = 1

        # Helper to get text from node
        def get_text(node):
            return code[node.start_byte : node.end_byte]

        # Traverse the tree
        stack = [root_node]
        while stack:
            node = stack.pop()

            if node.type == "method_declaration":
                # Extract method name
                name_node = node.child_by_field_name("name")
                if name_node:
                    functions.append(get_text(name_node))

            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    classes.append(get_text(name_node))

            elif node.type == "import_declaration":
                # Extract import path
                # import com.example.MyClass;
                # The structure is usually import_declaration -> scoped_identifier
                imports.append(
                    get_text(node).replace("import ", "").replace(";", "").strip()
                )

            elif node.type in [
                "if_statement",
                "for_statement",
                "while_statement",
                "case_label",
                "catch_clause",
            ]:
                complexity += 1

            elif node.type == "binary_expression":
                operator = node.child_by_field_name("operator")
                if operator and get_text(operator) in ["&&", "||"]:
                    complexity += 1

            # Add children to stack
            for child in node.children:
                stack.append(child)

        return {
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "complexity": complexity,
            "lines_of_code": len(code.splitlines()),
            "issues": issues,
        }
