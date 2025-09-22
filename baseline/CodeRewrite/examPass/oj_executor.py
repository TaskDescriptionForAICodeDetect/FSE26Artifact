#!/usr/bin/env python3

from __future__ import annotations

import os
import time
import signal
import uuid
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
from typing import Set, Tuple
import traceback
import re

from tree_sitter import Language, Parser
from tree_sitter_c import language as TS_LANG_C
from tree_sitter_cpp import language as TS_LANG_CPP
from tree_sitter_java import language as TS_LANG_JAVA


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    compile_error: bool
    time_ms: int
    workdir: str


class OJExecutor:

    def __init__(self, base_workdir: Optional[Path | str] = None):
        self.base_workdir: Path = Path(base_workdir or Path(__file__).parent / "workdir")
        self.base_workdir.mkdir(parents=True, exist_ok=True)
        self._debug_enabled = os.environ.get("OJ_DEBUG", "").strip() not in ("", "0", "false", "False")

    def run(
        self,
        code: str,
        language: str,
        stdin: str = "",
        time_limit_sec: float = 2.0,
        origin_code: Optional[str] = None,
        run_dir_name: Optional[str] = None,
    ) -> ExecutionResult:
        lang = (language or "").strip().lower()
        run_dir = self._create_run_dir(run_dir_name)

        try:
            if lang in ("py", "python"):
                return self._run_python(code, stdin, time_limit_sec, run_dir)
            if lang == "c":
                return self._run_c(code, stdin, time_limit_sec, run_dir, origin_code)
            if lang in ("cpp", "c++"):
                return self._run_cpp(code, stdin, time_limit_sec, run_dir, origin_code)
            if lang == "java":
                return self._run_java(code, stdin, time_limit_sec, run_dir)
            return ExecutionResult(
                stdout="",
                stderr=f"Unsupported language: {language}",
                exit_code=1,
                timed_out=False,
                compile_error=True,
                time_ms=0,
                workdir=str(run_dir),
            )
        finally:
            pass

    def _run_python(self, code: str, stdin: str, tl: float, cwd: Path) -> ExecutionResult:
        src = cwd / "Main.py"
        self._write_text(src, code)
        cmd = ["python3", "-u", str(src)]
        return self._execute(cmd, stdin, tl, cwd)

    def _run_c(self, code: str, stdin: str, tl: float, cwd: Path, origin_code: Optional[str]) -> ExecutionResult:
        src = cwd / "main.c"
        out = cwd / "main"
        if not src.exists():
            processed = self._preprocess_source_for_lang(code, lang="c")
            self._write_text(src, processed)
        comp = self._execute(["gcc", str(src), "-std=gnu11", "-w", "-o", str(out)], "", tl, cwd, is_compile=True)
        if comp.exit_code != 0:
            if origin_code:
                globals_txt = self._extract_global_declarations(origin_code, is_cpp=False)
                if globals_txt.strip():
                    repaired = self._preprocess_source_for_lang(code, lang="c", extra_globals=globals_txt)
                    temp_src = cwd / "main_injected.c"
                    self._write_text(temp_src, repaired)
                    comp2 = self._execute(["gcc", str(temp_src), "-std=gnu11", "-w", "-o", str(out)], "", tl, cwd, is_compile=True)
                    if comp2.exit_code != 0:
                        try:
                            temp_src.unlink()
                        except Exception:
                            pass
                        return comp2
                else:
                    return comp
            else:
                return comp
        return self._execute([str(out)], stdin, tl, cwd)

    def _run_cpp(self, code: str, stdin: str, tl: float, cwd: Path, origin_code: Optional[str]) -> ExecutionResult:
        src = cwd / "main.cpp"
        out = cwd / "main"
        if not src.exists():
            processed = self._preprocess_source_for_lang(code, lang="cpp")
            self._write_text(src, processed)
        comp = self._execute(["g++", str(src), "-std=gnu++17", "-w", "-o", str(out)], "", tl, cwd, is_compile=True)
        if comp.exit_code != 0:
            if origin_code:
                globals_txt = self._extract_global_declarations(origin_code, is_cpp=True)
                if globals_txt.strip():
                    repaired = self._preprocess_source_for_lang(code, lang="cpp", extra_globals=globals_txt)
                    temp_src = cwd / "main_injected.cpp"
                    self._write_text(temp_src, repaired)
                    comp2 = self._execute(["g++", str(temp_src), "-std=gnu++17", "-w", "-o", str(out)], "", tl, cwd, is_compile=True)
                    if comp2.exit_code != 0:
                        try:
                            temp_src.unlink()
                        except Exception:
                            pass
                        return comp2
                else:
                    return comp
            else:
                return comp
        return self._execute([str(out)], stdin, tl, cwd)

    def _run_java(self, code: str, stdin: str, tl: float, cwd: Path) -> ExecutionResult:
        pkg_name, cls_name = self._extract_java_package_and_class(code)
        if pkg_name:
            pkg_path = Path(*pkg_name.split('.'))
            src_dir = cwd / pkg_path
            src_dir.mkdir(parents=True, exist_ok=True)
            src = src_dir / f"{cls_name}.java"
            run_class = f"{pkg_name}.{cls_name}"
        else:
            src = cwd / f"{cls_name}.java"
            run_class = cls_name
        self._write_text(src, code)
        comp = self._execute(["javac", "-encoding", "UTF-8", "-d", str(cwd), str(src)], "", tl, cwd, is_compile=True)
        if comp.exit_code != 0:
            return comp
        return self._execute(["java", "-cp", str(cwd), run_class], stdin, tl, cwd)

    def _create_run_dir(self, name: Optional[str] = None) -> Path:
        if name:
            safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)
            d = self.base_workdir / safe
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
            return d
        d = self.base_workdir / uuid.uuid4().hex
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_text(self, path: Path, content: str) -> None:
        path.write_text(content if isinstance(content, str) else str(content), encoding="utf-8")

    def _preprocess_source_for_lang(self, code: str, lang: str, extra_globals: Optional[str] = None) -> str:
        if lang == "cpp":
            common_headers = [
                "#include <iostream>",
                "#include <vector>",
                "#include <string>",
                "#include <sstream>",
                "#include <cstdio>",
                "#include <algorithm>",
                "#include <map>",
                "#include <set>",
                "#include <queue>",
                "#include <stack>",
                "#include <deque>",
                "#include <unordered_map>",
                "#include <unordered_set>",
                "#include <cmath>",
                "#include <cctype>",
                "#include <limits>",
                "#include <numeric>",
                "#include <functional>",
                "#include <utility>",
                "#include <bitset>",
                "#include <iomanip>",
                "#include <tuple>",
                "#include <cstring>",
                "#include <climits>",
            ]
        else:
            common_headers = [
                "#include <stdio.h>",
                "#include <stdlib.h>",
                "#include <string.h>",
                "#include <math.h>",
                "#include <ctype.h>",
                "#include <stdbool.h>",
                "#include <limits.h>",
                "#include <stdint.h>",
                "#include <time.h>",
            ]
        header_block = "\n".join(common_headers) + "\n\n"
        middle = (extra_globals.strip() + "\n\n") if extra_globals else ""
        return header_block + middle + (code or "")

    def _strip_comments_c_like(self, code: str) -> str:
        import re
        code = re.sub(r"/\*.*?\*/", " ", code, flags=re.S)
        code = re.sub(r"//.*?$", " ", code, flags=re.M)
        return code

    def _extract_global_declarations_c_like(self, code: str, is_cpp: bool) -> str:
        cleaned = self._strip_comments_c_like(code or "")
        lines = cleaned.splitlines()
        depth = 0
        buffer = []
        current = []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            opens = line.count("{")
            closes = line.count("}")
            if depth == 0:
                if line.startswith('#'):
                    depth += opens
                    depth -= closes
                    continue
                current.append(raw)
                if ';' in line:
                    stmt = "\n".join(current).strip()
                    if not (is_cpp and stmt.startswith("using namespace")):
                        buffer.append(stmt)
                    current = []
            else:
                pass
            depth += opens
            depth -= closes
        globals_txt = "\n".join(buffer)
        return globals_txt

    def _extract_global_declarations_tree_sitter(self, code: str, is_cpp: bool) -> str:
        try:
            source_bytes = (code or "").encode("utf-8")
            lang_ptr = TS_LANG_CPP() if is_cpp else TS_LANG_C()
            ts_lang = Language(lang_ptr)
            parser = Parser(ts_lang)
            tree = parser.parse(source_bytes)
            root = tree.root_node

            def node_text(n):
                return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

            collected: List[str] = []
            def collect_top_level(n):
                t = n.type
                if t in ("preproc_include", "preproc_ifdef", "preproc_if", "preproc_else", "preproc_elif", "preproc_endif"):
                    return
                if t in ("preproc_def", "preproc_function_def"):
                    collected.append(node_text(n))
                    return
                if t in ("declaration", "type_definition", "using_declaration", "function_declaration"):
                    collected.append(node_text(n))
                    return
                if t == "declaration_list":
                    for g in n.children:
                        if g.type == "declaration":
                            collected.append(node_text(g))
                    return

            for child in root.children:
                collect_top_level(child)

            return "\n".join(collected)
        except Exception as e:
            print("[tree-sitter] extraction failed:", e)
            traceback.print_exc()
            return ""

    def _extract_global_declarations(self, code: str, is_cpp: bool) -> str:
        return self._extract_global_declarations_tree_sitter(code, is_cpp=is_cpp)

    def _collect_defined_identifiers(self, code: str, is_cpp: bool) -> Set[str]:
        try:
            source_bytes = (code or "").encode("utf-8")
            lang_ptr = TS_LANG_CPP() if is_cpp else TS_LANG_C()
            ts_lang = Language(lang_ptr)
            parser = Parser(ts_lang)
            tree = parser.parse(source_bytes)
            root = tree.root_node

            def node_text(n):
                return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

            names: Set[str] = set()

            def find_ident_in_declarator(n) -> Optional[str]:
                if n.type == "identifier":
                    return node_text(n)
                for ch in n.children:
                    res = find_ident_in_declarator(ch)
                    if res:
                        return res
                return None

            for child in root.children:
                t = child.type
                if t in ("declaration", "type_definition", "declaration_list"):
                    stack = [child]
                    while stack:
                        n = stack.pop()
                        if n.type == "init_declarator" or n.type == "declarator":
                            ident = find_ident_in_declarator(n)
                            if ident:
                                names.add(ident)
                        for c in n.children:
                            stack.append(c)
                else:
                    continue
            return names
        except Exception:
            traceback.print_exc()
            return set()

    def _rename_conflicting_globals(self, extra_globals: str, existing: Set[str], is_cpp: bool) -> str:
        try:
            source_bytes = (extra_globals or "").encode("utf-8")
            if not source_bytes:
                return extra_globals
            lang_ptr = TS_LANG_CPP() if is_cpp else TS_LANG_C()
            ts_lang = Language(lang_ptr)
            parser = Parser(ts_lang)
            tree = parser.parse(source_bytes)
            root = tree.root_node

            def text(n):
                return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

            def find_ident_in_declarator_node(n) -> Tuple[Optional[str], Optional[int], Optional[int]]:
                if n.type == "identifier":
                    return text(n), n.start_byte, n.end_byte
                for ch in n.children:
                    name, s, e = find_ident_in_declarator_node(ch)
                    if name is not None:
                        return name, s, e
                return None, None, None

            edits: List[Tuple[int, int, str]] = []
            used: Set[str] = set(existing)

            def unique_name(base: str) -> str:
                cand = f"{base}_ext"
                idx = 1
                while cand in used:
                    cand = f"{base}_ext{idx}"
                    idx += 1
                used.add(cand)
                return cand

            for child in root.children:
                if child.type not in ("declaration", "declaration_list"):
                    continue
                stack = [child]
                while stack:
                    n = stack.pop()
                    if n.type == "init_declarator" or n.type == "declarator":
                        name, s, e = find_ident_in_declarator_node(n)
                        if name and s is not None and e is not None:
                            if name in used:
                                new_name = unique_name(name)
                                edits.append((s, e, new_name))
                            else:
                                used.add(name)
                    for c in n.children:
                        stack.append(c)

            if not edits:
                return extra_globals

            edits.sort(key=lambda x: x[0])
            out_parts: List[bytes] = []
            last = 0
            for s, e, repl in edits:
                out_parts.append(source_bytes[last:s])
                out_parts.append(repl.encode("utf-8"))
                last = e
            out_parts.append(source_bytes[last:])
            return b"".join(out_parts).decode("utf-8", errors="replace")
        except Exception:
            traceback.print_exc()
            return extra_globals

    def _extract_undefined_identifiers_from_stderr(self, stderr: str) -> List[str]:
        if not stderr:
            return []
        candidates: List[str] = []
        patterns = [
            r"['\u2018]([^'\u2019]+)['\u2019]\s+undeclared",
            r"['\u2018]([^'\u2019]+)['\u2019].*?was not declared in this scope",
            r"use of undeclared identifier\s*'([^']+)'",
            r"identifier \"([^\"]+)\" is undefined",
            r"error:\s*['\u2018]([^'\u2019]+)['\u2019].*?has not been declared",
        ]
        for line in stderr.splitlines():
            for pat in patterns:
                m = re.search(pat, line)
                if m:
                    name = m.group(1).strip()
                    if name and name not in candidates:
                        candidates.append(name)
        if self._debug_enabled:
            print("[debug] parsed undefined identifiers:", candidates)
            print("[debug] raw stderr (first 500 chars):", (stderr or "")[:500])
        return candidates

    def _rename_globals_assign_names(self, extra_globals: str, target_names: List[str], is_cpp: bool, avoid: Set[str]) -> str:
        try:
            if not extra_globals.strip() or not target_names:
                return extra_globals
            source_bytes = extra_globals.encode("utf-8")
            lang_ptr = TS_LANG_CPP() if is_cpp else TS_LANG_C()
            ts_lang = Language(lang_ptr)
            parser = Parser(ts_lang)
            tree = parser.parse(source_bytes)
            root = tree.root_node

            def text(n):
                return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

            def find_ident_in_declarator_node(n) -> Tuple[Optional[int], Optional[int]]:
                if n.type == "identifier":
                    return n.start_byte, n.end_byte
                for ch in n.children:
                    s, e = find_ident_in_declarator_node(ch)
                    if s is not None:
                        return s, e
                return None, None

            edits: List[Tuple[int, int, str]] = []
            used: Set[str] = set(avoid)
            idx = 0

            def next_target_name() -> Optional[str]:
                nonlocal idx
                while idx < len(target_names):
                    name = target_names[idx]
                    idx += 1
                    if name and name not in used:
                        used.add(name)
                        return name
                return None

            for child in root.children:
                if child.type not in ("declaration", "declaration_list"):
                    continue
                stack = [child]
                while stack:
                    n = stack.pop()
                    if n.type == "init_declarator" or n.type == "declarator":
                        s, e = find_ident_in_declarator_node(n)
                        if s is not None and e is not None:
                            new_name = next_target_name()
                            if new_name is not None:
                                edits.append((s, e, new_name))
                    for c in n.children:
                        stack.append(c)

            if not edits:
                return extra_globals

            edits.sort(key=lambda x: x[0])
            out_parts: List[bytes] = []
            last = 0
            for s, e, repl in edits:
                out_parts.append(source_bytes[last:s])
                out_parts.append(repl.encode("utf-8"))
                last = e
            out_parts.append(source_bytes[last:])
            return b"".join(out_parts).decode("utf-8", errors="replace")
        except Exception:
            traceback.print_exc()
            return extra_globals

    def _get_parser(self, lang_ptr) -> Parser:
        ts_lang = Language(lang_ptr)
        return Parser(ts_lang)

    def _extract_java_package_and_class(self, code: str) -> tuple[str | None, str]:
        try:
            source = (code or "").encode("utf-8")
            parser = self._get_parser(TS_LANG_JAVA())
            tree = parser.parse(source)
            root = tree.root_node

            def text(n):
                return source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

            pkg_name: str | None = None
            public_class: str | None = None
            first_class: str | None = None

            for child in root.children:
                if child.type == "package_declaration":
                    pkg_name = text(child).strip()
                    if pkg_name.startswith("package "):
                        pkg_name = pkg_name[len("package "):].strip()
                    if pkg_name.endswith(";"):
                        pkg_name = pkg_name[:-1].strip()
                    break

            for child in root.children:
                if child.type in ("class_declaration", "normal_class_declaration"):
                    cls = None
                    for c in child.children:
                        if c.type == "identifier":
                            cls = text(c)
                            break
                    if first_class is None and cls:
                        first_class = cls
                    header = text(child[:child.end_byte - child.start_byte and child]) if False else text(child)
                    if cls and ("public class" in header or header.strip().startswith("public class")):
                        public_class = cls
                        break

            cls_name = public_class or first_class or "Main"
            return pkg_name, cls_name
        except Exception as e:
            print("[java] parse failed:", e)
            traceback.print_exc()
            return None, "Main"

    def _execute(
        self,
        cmd: List[str],
        stdin: str,
        time_limit_sec: float,
        cwd: Path,
        is_compile: bool = False,
        env_extra: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        start = time.time()
        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(cwd),
                env=env,
                start_new_session=True,
            )
            try:
                out, err = proc.communicate(input=stdin.encode("utf-8"), timeout=time_limit_sec)
                elapsed = int((time.time() - start) * 1000)
                return ExecutionResult(
                    stdout=out.decode("utf-8", errors="replace"),
                    stderr=err.decode("utf-8", errors="replace"),
                    exit_code=proc.returncode,
                    timed_out=False,
                    compile_error=is_compile and proc.returncode != 0,
                    time_ms=elapsed,
                    workdir=str(cwd),
                )
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception as e:
                    print("[timeout] killpg failed:", e)
                    traceback.print_exc()
                    try:
                        proc.kill()
                    except Exception as e2:
                        print("[timeout] proc.kill failed:", e2)
                        traceback.print_exc()
                        pass
                try:
                    out, err = proc.communicate(timeout=0.2)
                except Exception as e:
                    print("[timeout] communicate after kill failed:", e)
                    traceback.print_exc()
                    out, err = b"", b""
                elapsed = int((time.time() - start) * 1000)
                return ExecutionResult(
                    stdout=out.decode("utf-8", errors="replace"),
                    stderr=err.decode("utf-8", errors="replace"),
                    exit_code=-1,
                    timed_out=True,
                    compile_error=is_compile,
                    time_ms=elapsed,
                    workdir=str(cwd),
                )
        except FileNotFoundError as e:
            print("[exec] command not found:", e)
            traceback.print_exc()

