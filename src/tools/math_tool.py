from typing import Callable, Union, Literal
import math

from llama_index.core.tools.tool_spec.base import BaseToolSpec
import sympy as sp


class AutoChainedSympyMathToolSpec(BaseToolSpec):
    
    spec_functions = [
        "sympy_auto_chain",
        "sympy_explain",
    ]

    _SYM_LOCALS: dict[str, Callable] = {
        # Arithmetic / core
        "abs": sp.Abs,
        "sign": sp.sign,

        # Roots & powers
        "sqrt": sp.sqrt,
        "cbrt": sp.cbrt,

        # Exponentials & logs
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,

        # Trigonometry
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "csc": sp.csc,
        "sec": sp.sec,
        "cot": sp.cot,

        # Inverse trig
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,

        # Hyperbolic
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,

        # Calculus (callable inside expressions)
        "diff": sp.diff,
        "integrate": sp.integrate,
        "limit": sp.limit,

        # Algebra helpers
        "simplify": sp.simplify,
        "factor": sp.factor,
        "expand": sp.expand,

        # Constants
        "pi": sp.pi,
        "e": sp.E,
        "E": sp.E,
        "I": sp.I,          # imaginary unit
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
    
    def _detect_calculus(self, expr: str) -> str | None:
        if expr.strip().startswith("integrate("):
            return "integrate"
        if expr.strip().startswith("diff("):
            return "diff"
        return None

    def sympy_auto_chain(self, expression: str) -> str:
        """
        Automatically process a math expression.

        Supported operations and syntax:

        • Arithmetic: +, -, *, /, **
        • Parentheses for grouping
        • Equations: x^2 = 4
        • Inequalities: x^2 <= 4
        • Simplification: simplify(expr)
        • Expansion: expand(expr)
        • Factoring: factor(expr)

        Calculus:
        • Differentiation: diff(expr, variable)
        Example: diff(x^2 + 3*x, x)

        • Integration: integrate(expr, variable)
        Example: integrate(sin(x), x)

        • Definite integrals:
        integrate(expr, (variable, a, b))

        Trigonometry:
        sin, cos, tan, asin, acos, atan

        Constants:
        pi, e

        Formatting rules:
        • Use ^ for powers
        • Use explicit multiplication (2*x, not 2x)
        • Variables must be alphabetic

        Args:
            expression (str): Math input
        """
        
        try:
            calc_op = self._detect_calculus(expression)

            if calc_op:
                expr = self._parse(expression)
                return str(expr)
            
            op, left, right = self._detect_relation(expression)
            
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

            expr: sp.Expr = self._parse(expression)

            if expr.free_symbols:
                return str(sp.simplify(expr))

            return str(expr.evalf())

        except Exception as e:
            return f"Error: {e}"

    def sympy_explain(self, expression: str) -> str:
        """
        Explain step-by-step how an expression,
        equation, or inequality is processed.

        Args:
            expression (str): Math input

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


class SimpleMathToolSpec(BaseToolSpec):
    """
    Simple numeric math tool.
    Supports arithmetic, trig, factorial, powers, logs.
    Accepts two inputs and normalizes worded numbers.
    """

    spec_functions = ["SimpleMathToolSpec"]

    _CONSTANTS = {
        "pi": math.pi,
        "π": math.pi,
        "e": math.e,
    }

    _TRIG_FUNCS = ["sin", "cos", "tan", "asin", "acos", "atan"]
    _ARITH_OPS = ["add", "subtract", "multiply", "divide", "power"]
    _LOG_OPS = ["log", "log10", "ln"]
    _UNARY_OPS = ["factorial"] + _TRIG_FUNCS + _LOG_OPS

    def SimpleMathToolSpec(
        self,
        op: Literal[
            "add", "subtract", "multiply", "divide", "power",
            "factorial", "sin", "cos", "tan", "asin", "acos", "atan",
            "log", "log10", "ln"
        ],
        a: Union[float, int],
        b: Union[float, int, None] = None,
        degrees: bool = False,
        log_base: float = None
    ) -> str:
        """
        Compute a simple math operation.

        Args:
            op: Operation (Literal)
            a: First input
            b: Second input if needed
            degrees: Interpret trig inputs in degrees
            log_base: Base for logarithm (default e)

        """
        try:
            x = a
            y = b

            # Unary operations
            if op == "factorial":
                if x < 0 or not float(x).is_integer():
                    return "Factorial only defined for non-negative integers"
                return str(math.factorial(int(x)))

            if op in self._TRIG_FUNCS:
                val = math.radians(x) if degrees else x
                return str(getattr(math, op)(val))

            if op == "log":
                base = log_base if log_base else math.e
                return str(math.log(x, base))
            if op == "ln":
                return str(math.log(x))
            if op == "log10":
                return str(math.log10(x))

            # Binary arithmetic operations
            if op == "add":
                if y is None:
                    return "Second input required for addition"
                return str(x + y)
            if op == "subtract":
                if y is None:
                    return "Second input required for subtraction"
                return str(x - y)
            if op == "multiply":
                if y is None:
                    return "Second input required for multiplication"
                return str(x * y)
            if op == "divide":
                if y is None:
                    return "Second input required for division"
                if y == 0:
                    return "Division by zero error"
                return str(x / y)
            if op == "power":
                if y is None:
                    return "Second input required for power"
                return str(x ** y)

            return "Unknown operation"

        except Exception as e:
            return f"Error: {e}"
