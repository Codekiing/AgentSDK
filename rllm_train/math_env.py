import ast
import math
import random
import re

from rllm_train.base import BaseEnv


class CalculateTool:
    json = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate concrete numeric arithmetic only. Supports fractions, factorial, "
                "comb/binomial, log/ln, exp, sqrt, and trig functions. Does not solve "
                "equations, simplify symbolic expressions, or accept variables like x/y/n."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Numeric expression after substituting values, e.g. '2 + 3', "
                            "'sqrt(16)', or 'frac(1, 2) + 0.25'. Invalid: 'x + 1', "
                            "'solve(x+1=2)', 'simplify(x/y)'."
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    }


class FinishTool:
    json = {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit your final answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "Your final answer",
                    }
                },
                "required": ["response"],
            },
        },
    }


class MathCalcEnv(BaseEnv):
    """Simple math calculation environment with a calculator tool."""

    def __init__(self, task=None, max_steps=3):
        self.task = task or {}
        self.max_steps = max_steps
        self.step_count = 0
        self.question = self.task.get("question", "")
        self.answer = self.task.get("answer", "")
        self._reset_reward_state()

    def _reset_reward_state(self):
        self.successful_calculates = 0
        self.calculator_errors = 0
        self.symbolic_calculator_errors = 0
        self.unknown_tools = 0
        self.parsed_tool_call = False
        self.synthetic_finish = False
        self.finished = False
        self.last_response = ""
        self.reward_breakdown_by_step = []

    def reset(self):
        self.step_count = 0
        self._reset_reward_state()
        observation = {"question": self.question}
        return observation, {}

    def step(self, action):
        self.step_count += 1

        if isinstance(action, str):
            self.finished = True
            self.synthetic_finish = True
            self.last_response = action
            reward, components = score_math_trajectory(
                action,
                self.answer,
                parsed_tool_call=False,
                synthetic_finish=True,
                finished=True,
                steps=self.step_count,
                max_steps=self.max_steps,
            )
            return {}, reward, True, self._reward_info(components)

        if isinstance(action, list):
            parsed_tool_call = any(not call.get("synthetic_finish") for call in action)
            synthetic_finish = any(call.get("synthetic_finish") for call in action)
            parser_errors = sum(1 for call in action if call.get("parse_error_type"))
            malformed_tool_calls = sum(1 for call in action if call.get("malformed_tool_call"))
            self.parsed_tool_call = self.parsed_tool_call or parsed_tool_call
            self.synthetic_finish = self.synthetic_finish or synthetic_finish

            tool_outputs = {}
            step_components = {"correctness": 0.0, "shaping": 0.0, "total": 0.0, "events": []}
            if parser_errors:
                step_components["events"].append("parser_error")
            if malformed_tool_calls:
                step_components["events"].append("malformed_tool_call")
            finish_response = None
            for tool_call in action:
                func = tool_call.get("function", {})
                name = func.get("name", "")
                args = self._parse_args(func.get("arguments", {}))

                if name == "calculate":
                    expr = args.get("expression", "")
                    result = self._safe_eval(expr)
                    result_text = self._format_tool_result(result)
                    tool_outputs[tool_call.get("id", "0")] = result_text
                    if result_text.startswith("Error:"):
                        self.calculator_errors += 1
                        step_components["events"].append("calculator_error")
                        if is_symbolic_calculator_error(result_text):
                            self.symbolic_calculator_errors += 1
                            step_components["events"].append("symbolic_calculator_error")
                    else:
                        self.successful_calculates += 1
                        step_components["events"].append("successful_calculate")
                elif name == "finish":
                    finish_response = args.get("response", "")
                else:
                    self.unknown_tools += 1
                    tool_outputs[tool_call.get("id", "0")] = f"Unknown tool: {name}"
                    step_components["events"].append("unknown_tool")

            if finish_response is not None:
                self.finished = True
                self.last_response = finish_response
                reward, components = score_math_trajectory(
                    finish_response,
                    self.answer,
                    parsed_tool_call=self.parsed_tool_call,
                    synthetic_finish=self.synthetic_finish,
                    finished=True,
                    steps=self.step_count,
                    max_steps=self.max_steps,
                    successful_calculates=self.successful_calculates,
                    calculator_errors=self.calculator_errors,
                    unknown_tools=self.unknown_tools,
                    symbolic_calculator_errors=self.symbolic_calculator_errors,
                    parser_errors=parser_errors,
                    malformed_tool_calls=malformed_tool_calls,
                )
                return {}, reward, True, self._reward_info(components)

            done = self.step_count >= self.max_steps
            if done and not self.finished:
                reward, step_components = score_math_trajectory(
                    "",
                    self.answer,
                    parsed_tool_call=self.parsed_tool_call,
                    synthetic_finish=self.synthetic_finish,
                    finished=False,
                    steps=self.step_count,
                    max_steps=self.max_steps,
                    successful_calculates=self.successful_calculates,
                    calculator_errors=self.calculator_errors,
                    unknown_tools=self.unknown_tools,
                    symbolic_calculator_errors=self.symbolic_calculator_errors,
                    parser_errors=parser_errors,
                    malformed_tool_calls=malformed_tool_calls,
                )
            else:
                reward = 0.0
            return {"tool_outputs": tool_outputs}, reward, done, self._reward_info(step_components)

        components = {"correctness": 0.0, "shaping": -0.03, "total": 0.0, "events": ["invalid_action"]}
        return {}, 0.0, True, self._reward_info(components)

    def _parse_args(self, args):
        if isinstance(args, str):
            import json
            try:
                return json.loads(args)
            except json.JSONDecodeError:
                return {"response": args}
        return args or {}

    def _reward_info(self, components):
        self.reward_breakdown_by_step.append(components)
        return {
            "reward_components": components,
            "reward_breakdown_by_step": self.reward_breakdown_by_step.copy(),
        }

    def _check_answer(self, response):
        return score_numeric_answer(response, self.answer)

    def _safe_eval(self, expr):
        try:
            expr = normalize_math_expression(str(expr))

            def calculate(nested_expr):
                return self._safe_eval(nested_expr)

            names = _math_eval_names(calculate)
            validate_calculator_expression(expr, names)
            return eval(expr, {"__builtins__": {}}, names)  # noqa: S307
        except Exception as e:
            return f"Error: {e}"

    def _format_tool_result(self, result):
        try:
            return str(result)
        except ValueError as e:
            return f"Error: {e}"

    def close(self):
        pass

    @staticmethod
    def from_dict(info):
        return MathCalcEnv(task=info, max_steps=info.get("max_steps", 3))

    @staticmethod
    def is_multithread_safe():
        return True


def _math_eval_names(calculate=None):
    names = {
        "abs": abs,
        "acos": math.acos,
        "asin": math.asin,
        "atan": math.atan,
        "binomial": math.comb,
        "ceil": math.ceil,
        "comb": math.comb,
        "cos": math.cos,
        "e": math.e,
        "exp": math.exp,
        "factorial": math.factorial,
        "floor": math.floor,
        "frac": lambda a, b: a / b,
        "ln": math.log,
        "log": math.log,
        "log10": math.log10,
        "pi": math.pi,
        "pow": pow,
        "sin": math.sin,
        "sqrt": math.sqrt,
        "tan": math.tan,
    }
    if calculate is not None:
        names["calculate"] = calculate
    return names


_SYMBOLIC_CALCULATOR_ERROR = "symbolic expression is not supported by calculate; use only numeric arithmetic"
_SYMBOLIC_FUNCTION_NAMES = {
    "derive",
    "differentiate",
    "diff",
    "expand",
    "factor",
    "integrate",
    "limit",
    "simplify",
    "solve",
    "subs",
    "sym",
}
_ALLOWED_CALCULATOR_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Tuple,
)


def is_symbolic_calculator_error(message):
    return _SYMBOLIC_CALCULATOR_ERROR in str(message)


def validate_calculator_expression(expr, names):
    if len(expr) > 240:
        raise ValueError("expression too long")
    if "=" in expr:
        raise ValueError(_SYMBOLIC_CALCULATOR_ERROR)

    identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", expr))
    if identifiers & _SYMBOLIC_FUNCTION_NAMES:
        raise ValueError(_SYMBOLIC_CALCULATOR_ERROR)
    unknown_identifiers = identifiers - set(names)
    if unknown_identifiers:
        raise ValueError(_SYMBOLIC_CALCULATOR_ERROR)

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError("invalid expression") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_CALCULATOR_NODES):
            raise ValueError("invalid expression")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("invalid expression")


def normalize_math_expression(expr):
    expr = str(expr).strip()
    expr = expr.replace("\\\\", "\\")
    expr = expr.replace("^", "**")
    expr = expr.replace("[", "(").replace("]", ")")
    expr = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", expr)
    expr = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", expr)
    expr = expr.replace("\\pi", str(math.pi))
    expr = re.sub(r"(?<![A-Za-z_])C\s*\(", "comb(", expr)
    if not re.fullmatch(r"[0-9A-Za-z_+\-*/=()., '\"\t]+", expr):
        raise ValueError("invalid expression")
    return expr


def eval_answer_to_float(answer_str):
    if answer_str is None:
        return None
    text = str(answer_str).strip()
    mixed = re.fullmatch(r"(-?\d+)\s+\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", text)
    if mixed:
        whole = float(mixed.group(1))
        numerator = eval_answer_to_float(mixed.group(2))
        denominator = eval_answer_to_float(mixed.group(3))
        if numerator is not None and denominator not in (None, 0):
            sign = -1 if whole < 0 else 1
            return whole + sign * numerator / denominator

    numbers = re.findall(r'-?\d+\.?\d*', text)
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return float(eval(normalize_math_expression(text), {"__builtins__": {}}, _math_eval_names()))
    except Exception:
        pass
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def score_numeric_answer(response, expected_answer):
    response_text = str(response)
    numbers = re.findall(r'-?\d+\.?\d*', response_text)
    predicted = eval_answer_to_float(response_text)
    if predicted is None:
        if not numbers:
            return 0.0
        predicted = float(numbers[-1])
    expected = eval_answer_to_float(expected_answer)
    if expected is None:
        return 0.0
    tolerance = max(1e-3, 1e-2 * abs(expected))
    if abs(predicted - expected) <= tolerance:
        return 1.0
    rel_err = abs(predicted - expected) / max(abs(expected), 1e-6)
    if rel_err < 0.1:
        return 0.5
    if rel_err < 0.5:
        return 0.2
    return 0.0


def _clamp(value, low, high):
    return max(low, min(high, value))


def score_math_trajectory(
    final_response,
    expected_answer,
    *,
    parsed_tool_call=False,
    synthetic_finish=False,
    finished=False,
    steps=0,
    max_steps=3,
    successful_calculates=0,
    calculator_errors=0,
    unknown_tools=0,
    symbolic_calculator_errors=0,
    parser_errors=0,
    malformed_tool_calls=0,
):
    correctness = score_numeric_answer(final_response, expected_answer)
    events = []

    answer_reward = 0.8 * correctness
    valid_finish = finished and not synthetic_finish and eval_answer_to_float(final_response) is not None
    protocol_bonus = 0.02 if parsed_tool_call else 0.0
    finish_bonus = 0.1 if valid_finish else 0.0
    tool_bonus = 0.08 if successful_calculates > 0 else 0.0
    penalty = 0.0

    if correctness >= 1.0:
        events.append("correct_answer")
    elif correctness > 0.0:
        events.append("partial_answer")
    if parsed_tool_call:
        events.append("parsed_tool_call")
    if valid_finish:
        events.append("valid_finish")
    elif finished:
        events.append("finish_used")
    if successful_calculates:
        events.append("successful_calculate")
    if calculator_errors:
        penalty += min(calculator_errors, 3) * 0.03
        events.append("calculator_error")
    if symbolic_calculator_errors:
        penalty += min(symbolic_calculator_errors, 3) * 0.02
        events.append("symbolic_calculator_error")
    if parser_errors:
        penalty += min(parser_errors, 3) * 0.02
        events.append("parser_error")
    if malformed_tool_calls:
        penalty += min(malformed_tool_calls, 3) * 0.02
        events.append("malformed_tool_call")
    if unknown_tools:
        penalty += min(unknown_tools, 3) * 0.03
        events.append("unknown_tool")
    if not finished and steps >= max_steps:
        penalty += 0.02
        events.append("no_finish")

    shaping = protocol_bonus + finish_bonus + tool_bonus - penalty
    total = _clamp(answer_reward + shaping, 0.0, 1.0)
    components = {
        "correctness": round(correctness, 6),
        "answer_reward": round(answer_reward, 6),
        "protocol_bonus": round(protocol_bonus, 6),
        "finish_bonus": round(finish_bonus, 6),
        "tool_bonus": round(tool_bonus, 6),
        "penalty": round(penalty, 6),
        "shaping": round(shaping, 6),
        "total": round(total, 6),
        "parsed_tool_call": bool(parsed_tool_call),
        "synthetic_finish": bool(synthetic_finish),
        "finished": bool(finished),
        "successful_calculates": int(successful_calculates),
        "calculator_errors": int(calculator_errors),
        "symbolic_calculator_errors": int(symbolic_calculator_errors),
        "parser_errors": int(parser_errors),
        "malformed_tool_calls": int(malformed_tool_calls),
        "unknown_tools": int(unknown_tools),
        "events": events,
    }
    return total, components


def generate_math_problems(n=100, seed=42, difficulty="mixed"):
    rng = random.Random(seed)
    problems = []

    def _simple(rng):
        ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b), ("*", lambda a, b: a * b)]
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        sym, fn = rng.choice(ops)
        return f"What is {a} {sym} {b}?", str(fn(a, b))

    def _multi_step(rng):
        templates = [
            lambda: _multi_step_chain(rng),
            lambda: _word_problem(rng),
            lambda: _percentage_problem(rng),
            lambda: _comparison_problem(rng),
        ]
        return rng.choice(templates)()

    def _multi_step_chain(rng):
        a, b, c = rng.randint(2, 50), rng.randint(2, 50), rng.randint(2, 20)
        op1, op2 = rng.choice([("+", "-"), ("*", "+"), ("+", "*"), ("-", "+"), ("*", "-")])
        expr = f"({a} {op1} {b}) {op2} {c}"
        answer = eval(expr)  # noqa: S307
        patterns = [
            f"First compute {a} {op1} {b}, then {op2} {c}. What is the result?",
            f"What is ({a} {op1} {b}) {op2} {c}?",
            f"Calculate: start with {a}, {_op_word(op1)} {b}, then {_op_word(op2)} {c}.",
        ]
        return rng.choice(patterns), str(answer)

    def _word_problem(rng):
        items = [("apples", "oranges"), ("books", "pens"), ("shirts", "pants"), ("tickets", "drinks")]
        item1, item2 = rng.choice(items)
        p1, p2 = rng.randint(2, 15), rng.randint(2, 15)
        q1, q2 = rng.randint(1, 10), rng.randint(1, 10)
        total = p1 * q1 + p2 * q2
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
        name = rng.choice(names)
        question = (
            f"{name} buys {q1} {item1} at ${p1} each and {q2} {item2} at ${p2} each. "
            f"How much does {name} spend in total?"
        )
        return question, str(total)

    def _percentage_problem(rng):
        base = rng.choice([50, 80, 100, 120, 150, 200, 250, 300, 400, 500])
        pct = rng.choice([10, 15, 20, 25, 30, 40, 50, 75])
        result = base * pct / 100
        patterns = [
            f"What is {pct}% of {base}?",
            f"A product costs ${base}. If there is a {pct}% discount, how much do you save?",
            f"Calculate {pct} percent of {base}.",
        ]
        answer = int(result) if result == int(result) else result
        return rng.choice(patterns), str(answer)

    def _comparison_problem(rng):
        a, b = rng.randint(5, 50), rng.randint(5, 50)
        c, d = rng.randint(1, 30), rng.randint(1, 30)
        val1, val2 = a * b, c * d
        question = (
            f"Store A sells {a} items at ${b} each. Store B sells {c} items at ${d} each. "
            f"How much more does the store with higher revenue earn?"
        )
        return question, str(abs(val1 - val2))

    def _op_word(op):
        return {"+" : "add", "-": "subtract", "*": "multiply by"}.get(op, op)

    for _ in range(n):
        if difficulty == "simple":
            q, a = _simple(rng)
        elif difficulty == "hard":
            q, a = _multi_step(rng)
        else:
            q, a = _multi_step(rng) if rng.random() > 0.8 else _simple(rng)
        problems.append({"question": q, "answer": str(a)})
    return problems
