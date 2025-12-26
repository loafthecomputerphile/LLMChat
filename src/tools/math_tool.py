from typing import Callable

from llama_index.core.tools.tool_spec.base import BaseToolSpec
import sympy as sp


class AutoChainedSympyMathToolSpec(BaseToolSpec):
    
    spec_functions = [
        "auto_chain",
        "explain",
    ]

    _SYM_LOCALS: dict[str, Callable] = {
        "sqrt": sp.sqrt,
        "log": sp.log,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "pi": sp.pi,
        "e": sp.E,
    }

    _REL_OPS: list[str] = ["<=", ">=", "<", ">", "="]

    def _parse(self, expr: str) -> sp.Expr:
        return sp.sympify(expr, locals=self._SYM_LOCALS)

    def _detect_relation(self, expr: str) -> tuple[str | None, str, str | None]:
        for op in self._REL_OPS:
            if op in expr:
                left, right = expr.split(op, 1)
                return op, left.strip(), right.strip()
        return None, expr, None

    # ----------------------------
    # 🔁 Auto-chain symbolic → numeric
    # ----------------------------

    def auto_chain(self, expression: str) -> str:
        """
        Automatically process a math expression.

        Handles:
        - expressions
        - equations (=)
        - inequalities (<, <=, >, >=)

        Args:
            expression (str): Math input

        Returns:
            str: Result
        """
        try:
            op, left, right = self._detect_relation(expression)

            # ----------------------------
            # Equation or inequality
            # ----------------------------
            if op:
                lhs: sp.Expr = self._parse(left)
                rhs: sp.Expr = self._parse(right)

                if op == "=":
                    expr: sp.Expr = lhs - rhs
                    symbols = list(expr.free_symbols)
                    sol: sp.Expr = sp.solve(expr, symbols, dict=True)
                    return str(sol)

                # Inequalities
                rel_map: dict[str, sp.Rational] = {
                    "<": sp.Lt,
                    "<=": sp.Le,
                    ">": sp.Gt,
                    ">=": sp.Ge,
                }

                rel: sp.Expr = rel_map[op](lhs, rhs)
                symbols: list[sp.Basic] = list(rel.free_symbols)

                if not symbols:
                    return str(bool(rel))

                sol: sp.Expr = sp.solve_univariate_inequality(rel, symbols[0])
                return str(sol)

            # ----------------------------
            # Pure expression
            # ----------------------------
            expr: sp.Expr = self._parse(expression)

            if expr.free_symbols:
                return str(sp.simplify(expr))

            return str(expr.evalf())

        except Exception as e:
            return f"Error: {e}"

    def explain(self, expression: str) -> str:
        """
        Explain step-by-step how an expression,
        equation, or inequality is processed.

        Args:
            expression (str): Math input

        Returns:
            str: Explanation
        """
        try:
            steps: list[str] = []

            op, left, right = self._detect_relation(expression)

            if op:
                steps.append(f"Detected relation: '{op}'")

                lhs: sp.Expr = self._parse(left)
                rhs: sp.Expr = self._parse(right)

                if op == "=":
                    steps.append(f"Move all terms to one side: {lhs - rhs}")
                    simplified: sp.Expr = sp.simplify(lhs - rhs)
                    steps.append(f"Simplify: {simplified}")

                    symbols: list[sp.Basic] = list(simplified.free_symbols)
                    sol: sp.Expr = sp.solve(simplified, symbols)
                    steps.append(f"Solve for {symbols}: {sol}")
                    return "\n".join(str(s) for s in steps)

                # Inequalities
                steps.append(f"Left-hand side: {lhs}")
                steps.append(f"Right-hand side: {rhs}")

                rel_map: dict[str, sp.Rational] = {
                    "<": sp.Lt,
                    "<=": sp.Le,
                    ">": sp.Gt,
                    ">=": sp.Ge,
                }

                rel: sp.Expr = rel_map[op](lhs, rhs)
                steps.append(f"Construct inequality: {rel}")

                symbols: list[sp.Basic] = list(rel.free_symbols)
                sol: sp.Expr = sp.solve_univariate_inequality(rel, symbols[0])
                steps.append(f"Solve inequality: {sol}")
                return "\n".join(str(s) for s in steps)

            # ----------------------------
            # Expression
            # ----------------------------
            expr: sp.Expr = self._parse(expression)
            steps.append(f"Parsed expression: {expr}")

            simplified: sp.Expr = sp.simplify(expr)
            if simplified != expr:
                steps.append(f"Simplify: {simplified}")

            if not simplified.free_symbols:
                steps.append(f"Evaluate numerically: {simplified.evalf()}")

            return "\n".join(str(s) for s in steps)

        except Exception as e:
            return f"Error: {e}"
