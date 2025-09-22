import re
from dataclasses import dataclass
from typing import List

from baseutils.ioutils import Solution


@dataclass
class ExtractorConfig:
    remove_comments: bool = True
    remove_package_and_imports: bool = True
    rewrite_class_names: bool = True


def _remove_comments(code: str, lang: str) -> str:
    if lang in ("java", "c", "c++"):
        pattern = r'(//.*?$)|(/\*.*?\*/)'
        return re.sub(pattern, '', code, flags=re.MULTILINE | re.DOTALL)
    if lang == "python":
        code = re.sub(r'#.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''[^']*?'''", '', code, flags=re.DOTALL)
        return code
    return code


def _remove_package_and_imports(code: str, lang: str) -> str:
    if lang == "java":
        code = re.sub(r'^\s*package\s+[^;]+;\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^\s*import\s+[^;]+;\s*', '', code, flags=re.MULTILINE)
        return code
    if lang in ("c", "c++"):
        code = re.sub(r'^\s*#\s*include\s+[^\n]+\n', '', code, flags=re.MULTILINE)
        using_ns = re.sub(r'^\s*using\s+namespace\s+[^;]+;\s*', '', code, flags=re.MULTILINE)
        return using_ns
    if lang == "python":
        code = re.sub(r'^\s*from\s+[^\s]+\s+import\s+[^\n]+\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'^\s*import\s+[^\n]+\n', '', code, flags=re.MULTILINE)
        return code
    return code


def _rewrite_class_names(code: str, lang: str) -> str:
    if lang == "java":
        code = re.sub(r'(class)\s+[A-Za-z_][A-Za-z0-9_]*', r'\1 ClassName', code)
        return code
    if lang in ("c++",):
        code = re.sub(r'(class|struct)\s+[A-Za-z_][A-Za-z0-9_]*', r'\1 ClassName', code)
        return code
    if lang == "python":
        code = re.sub(r'(class)\s+[A-Za-z_][A-Za-z0-9_]*', r'\1 ClassName', code)
        return code
    return code


def apply_extractor(solutions: List[Solution], config: ExtractorConfig) -> List[Solution]:
    processed: List[Solution] = []
    for s in solutions:
        new_code = s.code
        lang = (s.language or "").lower()
        if config.remove_comments:
            new_code = _remove_comments(new_code, lang)
        if config.remove_package_and_imports:
            new_code = _remove_package_and_imports(new_code, lang)
        if config.rewrite_class_names:
            new_code = _rewrite_class_names(new_code, lang)

        processed.append(Solution(
            code=new_code,
            label=s.label,
            task_name=s.task_name,
            description=s.description,
            language=s.language,
            model=s.model,
        ))
    return processed

