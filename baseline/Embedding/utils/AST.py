from __future__ import annotations
from tree_sitter import Language, Parser, Node
import tree_sitter_python

PY_LANGUAGE = Language(tree_sitter_python.language())
_PARSER = Parser(PY_LANGUAGE)
MAX_DEPTH = 50
DEEP_NODE_PLACEHOLDER = "(DEEP_NODE)"

def _simplify_ast_recursive(node: Node, depth: int = 0) -> str:
    """Recursive helper to simplify AST nodes with depth limit."""
    if depth >= MAX_DEPTH:
        return DEEP_NODE_PLACEHOLDER
    if node.type == 'parenthesized_expression' and node.named_child_count > 0:
        return _simplify_ast_recursive(node.named_children[0], depth)
    if node.type == 'comment':
        return ""
    if node.named_child_count == 0:
        return f"({node.type})"
    child_sexprs = []
    for child in node.named_children:
        simplified_child = _simplify_ast_recursive(child, depth + 1)
        if simplified_child:
            child_sexprs.append(simplified_child)
    if len(child_sexprs) == 1 and child_sexprs[0] == DEEP_NODE_PLACEHOLDER:
        return DEEP_NODE_PLACEHOLDER
    if not child_sexprs:
        return f"({node.type})"

    return f"({node.type} {' '.join(child_sexprs)})"



def get_simplified_ast(code: str) -> str:
    """Return simplified AST S-expression string for given Python code."""
    tree = _PARSER.parse(code.encode("utf8"))
    return _simplify_ast_recursive(tree.root_node)

def code_with_ast(code: str, *, sep: str = " <AST> ") -> str:
    simplified_ast_sexpr = get_simplified_ast(code)
    return f"{code}{sep}{simplified_ast_sexpr}"



def code_with_task(code: str, task: str, *, sep: str = " <TASK> ") -> str:
    """Concatenate code and task description."""
    return f"{code}{sep}{task}"


def code_with_ast_task(code: str, task: str, *, sep_ast: str = " <AST> ", sep_task: str = " <TASK> ") -> str:
    """Concatenate code, AST, and task description."""
    code_plus_ast = code_with_ast(code, sep=sep_ast)
    return f"{code_plus_ast}{sep_task}{task}"
if __name__ == "__main__":
    src = "def add(a, b):\n    return a + b\n"
    print(code_with_ast(src))
