from code_scalpel.symbolic_execution_tools.taint_tracker import SINK_PATTERNS
from code_scalpel.symbolic_execution_tools.security_analyzer import SecurityAnalyzer
import ast

print(f"RawSQL in SINK_PATTERNS: {'RawSQL' in SINK_PATTERNS}")
if "RawSQL" in SINK_PATTERNS:
    print(f"RawSQL maps to: {SINK_PATTERNS['RawSQL']}")

code = """
from django.db.models.expressions import RawSQL
user_input = request.GET.get("order")
queryset = MyModel.objects.annotate(val=RawSQL(user_input, []))
"""

analyzer = SecurityAnalyzer()
result = analyzer.analyze(code)
print(f"Has vulnerabilities: {result.has_vulnerabilities}")
print(f"Taint flows: {result.taint_flows}")

# Debugging _analyze_call
tree = ast.parse(code)
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        print(f"Found call: {analyzer._get_call_name(node)}")
