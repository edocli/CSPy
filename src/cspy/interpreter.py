from .nodes import (
    AST,
    Block,
    Num,
    Str,
    Bool,
    ListLit,
    ListAccess,
    Var,
    BinOp,
    UnaryOp,
    Assign,
    IfStmt,
    RepeatTimes,
    RepeatUntil,
    ForEach,
    ProcDef,
    ProcCall,
    Return,
    BuiltinProc,
)
from .tokens import TokenType

class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

class Interpreter:
    def __init__(self) -> None:
        self.global_scope = {
            "DISPLAY": BuiltinProc(self._builtin_display),
            "INPUT": BuiltinProc(self._builtin_input),
            "RANDOM": BuiltinProc(self._builtin_random),
            "APPEND": BuiltinProc(self._builtin_append),
            "INSERT": BuiltinProc(self._builtin_insert),
            "REMOVE": BuiltinProc(self._builtin_remove),
            "LENGTH": BuiltinProc(self._builtin_length),
        }
        self.env = [self.global_scope]

    def current_scope(self) -> dict:
        return self.env[-1]

    def get_var(self, name: str):
        for scope in reversed(self.env):
            if name in scope:
                return scope[name]
        raise Exception(f"Undefined variable: {name}")

    def set_var(self, name: str, value):
        for scope in reversed(self.env):
            if name in scope:
                scope[name] = value
                return
        self.current_scope()[name] = value

    def visit(self, node: AST):
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: AST):
        raise Exception(f"No visit_{type(node).__name__} method")

    def visit_Block(self, node: Block):
        result = None
        for stmt in node.stmts:
            result = self.visit(stmt)
        return result

    def visit_Num(self, node: Num):
        return node.val

    def visit_Str(self, node: Str):
        return node.val

    def visit_Bool(self, node: Bool):
        return node.val

    def visit_ListLit(self, node: ListLit):
        return [self.visit(elt) for elt in node.elts]

    def visit_ListAccess(self, node: ListAccess):
        target = self.visit(node.target)
        idx = self.visit(node.idx) - 1
        if not isinstance(target, list):
            raise Exception("Cannot index non-list")
        if not isinstance(idx, int):
            raise Exception("Index must be an integer")
        try:
            return target[idx]
        except IndexError:
            raise Exception(f"List index out of range: {idx + 1}")

    def visit_Var(self, node: Var):
        return self.get_var(node.val)

    def visit_BinOp(self, node: BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = node.op.type

        if op == TokenType.PLUS:
            if isinstance(left, str) or isinstance(right, str):
                return str(left) + str(right)
            return left + right
        elif op == TokenType.MINUS:
            return left - right
        elif op == TokenType.MUL:
            return left * right
        elif op == TokenType.DIV:
            return left / right
        elif op == TokenType.MOD:
            return left % right
        elif op == TokenType.EQ:
            return left == right
        elif op == TokenType.NE:
            return left != right
        elif op == TokenType.LT:
            return left < right
        elif op == TokenType.GT:
            return left > right
        elif op == TokenType.LE:
            return left <= right
        elif op == TokenType.GE:
            return left >= right
        elif op == TokenType.AND:
            return left and right
        elif op == TokenType.OR:
            return left or right

    def visit_UnaryOp(self, node: UnaryOp):
        expr = self.visit(node.expr)
        if node.op.type == TokenType.NOT:
            return not expr
        elif node.op.type == TokenType.MINUS:
            return -expr

    def visit_Assign(self, node: Assign):
        val = self.visit(node.right)
        if isinstance(val, list):
            val = val[:]
        if isinstance(node.left, Var):
            self.set_var(node.left.val, val)
        elif isinstance(node.left, ListAccess):
            target = self.visit(node.left.target)
            idx = self.visit(node.left.idx) - 1
            target[idx] = val
        return val

    def visit_IfStmt(self, node: IfStmt):
        if self.visit(node.cond):
            return self.visit(node.then_blk)
        elif node.else_blk:
            return self.visit(node.else_blk)

    def visit_RepeatTimes(self, node: RepeatTimes):
        count = self.visit(node.times)
        for _ in range(int(count)):
            self.visit(node.body)

    def visit_RepeatUntil(self, node: RepeatUntil):
        while not self.visit(node.cond):
            self.visit(node.body)

    def visit_ForEach(self, node: ForEach):
        iterable = self.visit(node.iter)
        var_name = node.var.val
        for item in iterable:
            self.set_var(var_name, item)
            self.visit(node.body)

    def visit_ProcDef(self, node: ProcDef):
        self.set_var(node.name.val, node)

    def visit_ProcCall(self, node: ProcCall):
        proc_node = self.get_var(node.name.val)
        if isinstance(proc_node, BuiltinProc):
            arg_values = [self.visit(arg) for arg in node.args]
            return proc_node.func(arg_values)
        if isinstance(proc_node, ProcDef):
            if len(node.args) != len(proc_node.params):
                raise Exception(f"Argument count mismatch for {node.name.val}")
            arg_values = [self.visit(arg) for arg in node.args]
            local_scope = {}
            for param, val in zip(proc_node.params, arg_values):
                local_scope[param.val] = val
            self.env.append(local_scope)
            ret_val = None
            try:
                self.visit(proc_node.body)
            except ReturnException as e:
                ret_val = e.value
            finally:
                self.env.pop()
            return ret_val
        raise Exception(f"{node.name.val} is not a procedure")

    def visit_Return(self, node: Return):
        val = self.visit(node.val) if node.val else None
        raise ReturnException(val)

    def _builtin_display(self, args: list[list]):
        print(*args)
        return None

    def _builtin_input(self, args: list[list]):
        prompt = args[0] if args else ""
        try:
            val = input(prompt)
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            return val

    def _builtin_random(self, args: list[list]):
        import random
        if len(args) != 2:
            raise Exception("RANDOM takes exactly 2 arguments")
        return random.randint(args[0], args[1])

    def _builtin_append(self, args: list[list]):
        if len(args) != 2:
            raise Exception("APPEND takes exactly 2 arguments")
        lst, val = args
        if not isinstance(lst, list):
            raise Exception("First argument to APPEND must be a list")
        lst.append(val)
        return None

    def _builtin_insert(self, args: list[list]):
        if len(args) != 3:
            raise Exception("INSERT takes exactly 3 arguments")
        lst, idx, val = args
        if not isinstance(lst, list):
            raise Exception("First argument to INSERT must be a list")
        if not isinstance(idx, int):
            raise Exception("Second argument to INSERT must be an integer")
        lst.insert(idx - 1, val)
        return None

    def _builtin_remove(self, args: list[list]):
        if len(args) != 2:
            raise Exception("REMOVE takes exactly 2 arguments")
        lst, idx = args
        if not isinstance(lst, list):
            raise Exception("First argument to REMOVE must be a list")
        if not isinstance(idx, int):
            raise Exception("Second argument to REMOVE must be an integer")
        try:
            lst.pop(idx - 1)
        except IndexError:
            raise Exception(f"List index out of range: {idx}")
        return None

    def _builtin_length(self, args: list[list]):
        if len(args) != 1:
            raise Exception("LENGTH takes exactly 1 argument")
        return len(args[0])
