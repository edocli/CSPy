# /// script
# requires-python = ">=3.10"
# ///

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Any
from sys import argv
from os import path


class TokenType(Enum):
    NUMBER = "NUMBER"
    STRING = "STRING"
    TRUE = "TRUE"
    FALSE = "FALSE"

    ID = "ID"
    EOF = "EOF"
    NEWLINE = "NEWLINE"

    PLUS = "PLUS"  # +
    MINUS = "MINUS"  # -
    MUL = "MUL"  # *
    DIV = "DIV"  # /
    MOD = "MOD"

    LPAREN = "LPAREN"  # (
    RPAREN = "RPAREN"  # )
    LBRACKET = "LBRACKET"  # [
    RBRACKET = "RBRACKET"  # ]
    LBRACE = "LBRACE"  # {
    RBRACE = "RBRACE"  # }
    COMMA = "COMMA"  # ,
    ASSIGN = "ASSIGN"  # <-

    EQ = "EQ"  # =
    NE = "NE"  # !=
    LT = "LT"  # <
    GT = "GT"  # >
    LE = "LE"  # <=
    GE = "GE"  # >=
    NOT = "NOT"
    AND = "AND"
    OR = "OR"

    IF = "IF"
    ELSE = "ELSE"

    REPEAT = "REPEAT"
    TIMES = "TIMES"
    UNTIL = "UNTIL"

    FOR = "FOR"
    EACH = "EACH"
    IN = "IN"

    PROC = "PROC"
    RET = "RET"


@dataclass(frozen=True, slots=True)
class Token:
    type: TokenType
    val: str | int | float | None = None


@dataclass(frozen=True, slots=True)
class AST:
    pass


@dataclass(frozen=True, slots=True)
class Num(AST):
    val: int | float


@dataclass(frozen=True, slots=True)
class Str(AST):
    val: str


@dataclass(frozen=True, slots=True)
class Bool(AST):
    val: bool


@dataclass(frozen=True, slots=True)
class ListLit(AST):
    elts: list[AST]


@dataclass(frozen=True, slots=True)
class ListAccess(AST):
    target: AST
    idx: AST


@dataclass(frozen=True, slots=True)
class Var(AST):
    val: str


@dataclass(frozen=True, slots=True)
class BinOp(AST):
    left: AST
    op: Token
    right: AST


@dataclass(frozen=True, slots=True)
class UnaryOp(AST):
    op: Token
    expr: AST


@dataclass(frozen=True, slots=True)
class Assign(AST):
    left: Var | ListAccess
    right: AST


@dataclass(frozen=True, slots=True)
class Block(AST):
    stmts: list[AST]


@dataclass(frozen=True, slots=True)
class IfStmt(AST):
    cond: AST
    then_blk: Block
    else_blk: Block | None = None


@dataclass(frozen=True, slots=True)
class RepeatTimes(AST):
    times: AST
    body: Block


@dataclass(frozen=True, slots=True)
class RepeatUntil(AST):
    cond: AST
    body: Block


@dataclass(frozen=True, slots=True)
class ForEach(AST):
    var: Var
    iter: AST
    body: Block


@dataclass(frozen=True, slots=True)
class ProcDef(AST):
    name: Var
    params: list[Var]
    body: Block


@dataclass(frozen=True, slots=True)
class ProcCall(AST):
    name: Var
    args: list[AST]


@dataclass(frozen=True, slots=True)
class Return(AST):
    val: AST | None


@dataclass(frozen=True, slots=True)
class BuiltinProc:
    func: Callable[[list[Any]], Any]


class Lexer:
    keywords = {
        "TRUE": TokenType.TRUE,
        "FALSE": TokenType.FALSE,
        "MOD": TokenType.MOD,
        "NOT": TokenType.NOT,
        "AND": TokenType.AND,
        "OR": TokenType.OR,
        "IF": TokenType.IF,
        "ELSE": TokenType.ELSE,
        "REPEAT": TokenType.REPEAT,
        "TIMES": TokenType.TIMES,
        "UNTIL": TokenType.UNTIL,
        "FOR": TokenType.FOR,
        "EACH": TokenType.EACH,
        "IN": TokenType.IN,
        "PROCEDURE": TokenType.PROC,
        "RETURN": TokenType.RET,
    }

    single_tokens = {
        "+": TokenType.PLUS,
        "-": TokenType.MINUS,
        "*": TokenType.MUL,
        "/": TokenType.DIV,
        "%": TokenType.MOD,
        "(": TokenType.LPAREN,
        ")": TokenType.RPAREN,
        "[": TokenType.LBRACKET,
        "]": TokenType.RBRACKET,
        "{": TokenType.LBRACE,
        "}": TokenType.RBRACE,
        ",": TokenType.COMMA,
        "=": TokenType.EQ,
    }

    double_tokens = {
        "<": {"-": TokenType.ASSIGN, "=": TokenType.LE, None: TokenType.LT},
        ">": {"=": TokenType.GE, None: TokenType.GT},
        "!": {"=": TokenType.NE},  # TODO: add error for single !
    }

    def __init__(self, text: str) -> None:
        """
        Initialize the lexer with the input text.
        """
        self.text = text
        self.pos = 0
        self.curr = self.text[self.pos] if self.text else None

    def advance(self) -> None:
        """
        Advance the 'pos' pointer and set the 'curr' character.
        """
        self.pos += 1
        self.curr = self.text[self.pos] if self.pos < len(self.text) else None

    def peek(self) -> str | None:
        """
        Peek at the next character without advancing the position.
        """
        return self.text[self.pos + 1] if self.pos + 1 < len(self.text) else None

    def eat_num(self) -> Token:
        """
        Eat a number (integer or float) from the input and return a NUMBER token.
        """
        num = ""
        is_float = False
        while self.curr is not None:
            if self.curr.isdigit():
                num += self.curr
            elif (
                self.curr == "." and not is_float and (p := self.peek()) and p.isdigit()
            ):
                is_float = True
                num += self.curr
            else:
                break
            self.advance()
        return Token(TokenType.NUMBER, float(num) if is_float else int(num))

    def eat_str(self) -> Token:
        """
        Eat a string literal from the input and return a STRING token.
        """
        res = ""
        self.advance()
        while self.curr and self.curr != '"':
            if self.curr == "\\" and self.peek():
                res += self.curr
                self.advance()
            res += self.curr
            self.advance()
        if self.curr == '"':
            self.advance()
        return Token(TokenType.STRING, res)

    def eat_id(self) -> Token:
        """
        Eat an identifier or keyword from the input and return the appropriate token.
        """
        start_pos = self.pos
        while self.curr and (self.curr.isalnum() or self.curr == "_"):
            self.advance()
        res = self.text[start_pos : self.pos]
        tp = self.keywords.get(res.upper(), TokenType.ID)
        return Token(tp, res if tp == TokenType.ID else None)

    def eat_sym(self) -> Token:
        """
        Eat a single-character symbol from the input and return the corresponding token.
        """
        char = self.curr
        self.advance()
        return Token(self.single_tokens[char])

    def eat_db_sym(self) -> Token:
        """
        Eat a double-character symbol from the input and return the corresponding token.
        """
        char = self.curr
        nxt = self.peek()
        mp = self.double_tokens[char]
        tp = mp.get(nxt, mp.get(None))
        self.advance()
        if nxt in mp:
            self.advance()
        return Token(tp)

    def tokenize(self) -> list[Token]:
        """
        Tokenize the input text and return a list of tokens.
        """
        toks = []
        while self.curr is not None:
            if self.curr.isspace():
                if self.curr == "\n":
                    toks.append(Token(TokenType.NEWLINE))
                self.advance()
            elif self.curr.isdigit():
                toks.append(self.eat_num())
            elif self.curr == '"':
                toks.append(self.eat_str())
            elif self.curr.isalpha() or self.curr == "_":
                toks.append(self.eat_id())
            elif self.curr in self.single_tokens:
                toks.append(self.eat_sym())
            elif self.curr in self.double_tokens:
                toks.append(self.eat_db_sym())
            else:
                raise Exception(f"Illegal char: {self.curr}")
        toks.append(Token(TokenType.EOF))
        return toks


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        """
        Initialize the parser with a list of tokens.
        """
        self.tokens = tokens
        self.pos = 0
        self.curr = self.tokens[self.pos]

    def advance(self) -> None:
        """
        Advance to the next token in the list.
        """
        self.pos += 1
        self.curr = self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def peek(self) -> Token:
        """
        Peek at the next token without advancing the position.
        """
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        return Token(TokenType.EOF)

    def skip_nl(self) -> None:
        """
        Skip newline tokens.
        """
        while self.curr.type == TokenType.NEWLINE:
            self.eat(TokenType.NEWLINE)

    def eat(self, tp: TokenType) -> Token:
        """
        Consume a token of the expected type and return it.
        """
        if self.curr.type == tp:
            tok = self.curr
            self.advance()
            return tok
        raise Exception(f"Unexpected: {self.curr}, expected: {tp}")

    def block(self) -> AST:
        """
        Parse a block of statements enclosed in braces and return a Block AST node.
        """
        self.skip_nl()
        self.eat(TokenType.LBRACE)
        stmts = []
        self.skip_nl()
        while self.curr.type not in (TokenType.RBRACE, TokenType.EOF):
            stmts.append(self.stmt())
            self.skip_nl()
        self.eat(TokenType.RBRACE)
        return Block(stmts)

    def parse(self) -> AST:
        """
        Parse the list of tokens and return the root AST node.
        """
        stmts = []
        while self.curr.type != TokenType.EOF:
            if self.curr.type == TokenType.NEWLINE:
                self.eat(TokenType.NEWLINE)
            else:
                stmts.append(self.stmt())
        return Block(stmts)

    def stmt(self) -> AST:
        """
        Parse a single statement and return the corresponding AST node.
        """
        tp = self.curr.type
        if tp == TokenType.IF:
            return self.if_stmt()
        if tp == TokenType.REPEAT:
            return self.repeat_stmt()
        if tp == TokenType.FOR:
            return self.foreach_stmt()
        if tp == TokenType.PROC:
            return self.proc_def()
        if tp == TokenType.RET:
            return self.ret_stmt()
        if tp == TokenType.ID and self.peek().type in (
            TokenType.ASSIGN,
            TokenType.LBRACKET,
        ):
            return self.assign()
        return self.expr()

    def if_stmt(self) -> AST:
        """
        Parse an if statement and return the corresponding AST node.
        """
        self.eat(TokenType.IF)
        self.eat(TokenType.LPAREN)
        self.skip_nl()
        cond = self.expr()
        self.skip_nl()
        self.eat(TokenType.RPAREN)
        then_b = self.block()
        else_b = None
        self.skip_nl()
        if self.curr.type == TokenType.ELSE:
            self.eat(TokenType.ELSE)
            else_b = self.block()
        return IfStmt(cond, then_b, else_b)

    def repeat_stmt(self) -> AST:
        """
        Parse a repeat statement and return the corresponding AST node.
        """
        self.eat(TokenType.REPEAT)
        if self.curr.type == TokenType.UNTIL:
            self.eat(TokenType.UNTIL)
            self.eat(TokenType.LPAREN)
            self.skip_nl()
            cond = self.expr()
            self.skip_nl()
            self.eat(TokenType.RPAREN)
            return RepeatUntil(cond, self.block())
        times = self.expr()
        self.eat(TokenType.TIMES)
        return RepeatTimes(times, self.block())

    def foreach_stmt(self) -> AST:
        """
        Parse a foreach statement and return the corresponding AST node.
        """
        self.eat(TokenType.FOR)
        self.eat(TokenType.EACH)
        var = Var(self.eat(TokenType.ID).val)
        self.eat(TokenType.IN)
        iter_ = self.expr()
        return ForEach(var, iter_, self.block())

    def proc_def(self) -> AST:
        """
        Parse a procedure definition and return the corresponding AST node.
        """
        self.eat(TokenType.PROC)
        name = Var(self.eat(TokenType.ID).val)
        self.eat(TokenType.LPAREN)
        self.skip_nl()
        params = []
        while self.curr.type not in (TokenType.RPAREN, TokenType.EOF):
            params.append(Var(self.eat(TokenType.ID).val))
            if self.curr.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                self.skip_nl()
            else:
                break
        self.eat(TokenType.RPAREN)
        return ProcDef(name, params, self.block())

    def proc_call(self) -> AST:
        """
        Parse a procedure call and return the corresponding AST node.
        """
        name = Var(self.eat(TokenType.ID).val)
        self.eat(TokenType.LPAREN)
        self.skip_nl()
        args = []
        while self.curr.type != TokenType.RPAREN:
            args.append(self.expr())
            if self.curr.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                self.skip_nl()
            else:
                break
        self.skip_nl()
        self.eat(TokenType.RPAREN)
        return ProcCall(name, args)

    def ret_stmt(self) -> AST:
        """
        Parse a return statement and return the corresponding AST node.
        """
        self.eat(TokenType.RET)
        self.eat(TokenType.LPAREN)
        self.skip_nl()
        val = self.expr()
        self.skip_nl()
        self.eat(TokenType.RPAREN)
        return Return(val)

    def assign(self) -> AST:
        """
        Parse an assignment statement and return the corresponding AST node.
        """
        node = Var(self.eat(TokenType.ID).val)
        if self.curr.type == TokenType.LBRACKET:
            self.eat(TokenType.LBRACKET)
            idx = self.expr()
            self.eat(TokenType.RBRACKET)
            node = ListAccess(node, idx)
        self.eat(TokenType.ASSIGN)
        return Assign(node, self.expr())

    def expr(self) -> AST:
        """
        Parse an expression and return the corresponding AST node.
        """
        node = self.comp()
        while self.curr.type in (TokenType.AND, TokenType.OR):
            op = self.eat(self.curr.type)
            node = BinOp(node, op, self.comp())
        return node

    def comp(self) -> AST:
        """
        Parse a comparison expression and return the corresponding AST node.
        """
        node = self.arith()
        while self.curr.type in (
            TokenType.EQ,
            TokenType.NE,
            TokenType.LT,
            TokenType.LE,
            TokenType.GT,
            TokenType.GE,
        ):
            op = self.eat(self.curr.type)
            node = BinOp(node, op, self.arith())
        return node

    def arith(self) -> AST:
        """
        Parse an arithmetic expression and return the corresponding AST node.
        """
        node = self.term()
        while self.curr.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.eat(self.curr.type)
            node = BinOp(node, op, self.term())
        return node

    def term(self) -> AST:
        """
        Parse a term in an arithmetic expression and return the corresponding AST node.
        """
        node = self.factor()
        while self.curr.type in (TokenType.MUL, TokenType.DIV, TokenType.MOD):
            op = self.eat(self.curr.type)
            node = BinOp(node, op, self.factor())
        return node

    def factor(self) -> AST:
        """
        Parse a factor in an arithmetic expression and return the corresponding AST node.
        """
        tp = self.curr.type
        if tp == TokenType.PLUS:
            self.eat(TokenType.PLUS)
            return self.factor()
        if tp == TokenType.MINUS:
            return UnaryOp(self.eat(TokenType.MINUS), self.factor())
        if tp == TokenType.NOT:
            return UnaryOp(self.eat(TokenType.NOT), self.factor())
        if tp == TokenType.NUMBER:
            return Num(self.eat(TokenType.NUMBER).val)
        if tp == TokenType.STRING:
            return Str(self.eat(TokenType.STRING).val)
        if tp == TokenType.TRUE:
            self.eat(TokenType.TRUE)
            return Bool(True)
        if tp == TokenType.FALSE:
            self.eat(TokenType.FALSE)
            return Bool(False)
        if tp == TokenType.LBRACKET:
            return self.list_lit()
        if tp == TokenType.ID:
            return self.id_expr()
        if tp == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)
            return node
        raise Exception(f"Unexpected: {self.curr}")

    def list_lit(self) -> AST:
        """
        Parse a list literal and return the corresponding AST node.
        """
        self.eat(TokenType.LBRACKET)
        self.skip_nl()
        elts = []
        while self.curr.type != TokenType.RBRACKET:
            elts.append(self.expr())
            if self.curr.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                self.skip_nl()
            else:
                break
        self.skip_nl()
        self.eat(TokenType.RBRACKET)
        return ListLit(elts)

    def id_expr(self) -> AST:
        """
        Parse an identifier expression, which could be a variable, list access, or procedure call, and return the corresponding AST node.
        """
        if self.peek().type == TokenType.LPAREN:
            return self.proc_call()
        node = Var(self.eat(TokenType.ID).val)
        if self.curr.type == TokenType.LBRACKET:
            self.eat(TokenType.LBRACKET)
            idx = self.expr()
            self.eat(TokenType.RBRACKET)
            return ListAccess(node, idx)
        return node


class ReturnException(Exception):
    def __init__(self, value):
        """
        Initialize the ReturnException with a return value.
        """
        self.value = value


class Interpreter:
    def __init__(self) -> None:
        """
        Initialize the interpreter with a global scope and environment stack.
        """
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
        """
        Get the current scope from the environment stack.
        """
        return self.env[-1]

    def get_var(self, name: str):
        """
        Get the value of a variable from the environment stack.
        """
        for scope in reversed(self.env):
            if name in scope:
                return scope[name]
        raise Exception(f"Undefined variable: {name}")

    def set_var(self, name: str, value):
        """
        Set the value of a variable in the environment stack.
        """
        for scope in reversed(self.env):
            if name in scope:
                scope[name] = value
                return
        self.current_scope()[name] = value

    def visit(self, node: AST):
        """
        Visit an AST node and dispatch to the appropriate visit method.
        """
        method_name = f"visit_{type(node).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: AST):
        """
        Generic visit method for unsupported AST nodes.
        """
        raise Exception(f"No visit_{type(node).__name__} method")

    def visit_Block(self, node: Block):
        """
        Visit a block of statements and execute them in sequence.
        """
        result = None
        for stmt in node.stmts:
            result = self.visit(stmt)
        return result

    def visit_Num(self, node: Num):
        """
        Visit a numeric literal and return its value.
        """
        return node.val

    def visit_Str(self, node: Str):
        """
        Visit a string literal and return its value.
        """
        return node.val

    def visit_Bool(self, node: Bool):
        """
        Visit a boolean literal and return its value.
        """
        return node.val

    def visit_ListLit(self, node: ListLit):
        """
        Visit a list literal and return its value.
        """
        return [self.visit(elt) for elt in node.elts]

    def visit_ListAccess(self, node: ListAccess):
        """
        Visit a list access node and return the accessed element.
        """
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
        """
        Visit a variable node and return its value.
        """
        return self.get_var(node.val)

    def visit_BinOp(self, node: BinOp):
        """
        Visit a binary operation node and perform the operation.
        """
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
        """
        Visit a unary operation node and perform the operation.
        """
        expr = self.visit(node.expr)
        if node.op.type == TokenType.NOT:
            return not expr
        elif node.op.type == TokenType.MINUS:
            return -expr

    def visit_Assign(self, node: Assign):
        """
        Visit an assignment node and set the variable or list element.
        """
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
        """
        Visit an if statement node and execute the appropriate block.
        """
        if self.visit(node.cond):
            return self.visit(node.then_blk)
        elif node.else_blk:
            return self.visit(node.else_blk)

    def visit_RepeatTimes(self, node: RepeatTimes):
        """
        Visit a repeat times node and execute the body the specified number of times."""
        count = self.visit(node.times)
        for _ in range(int(count)):
            self.visit(node.body)

    def visit_RepeatUntil(self, node: RepeatUntil):
        """
        Visit a repeat until node and execute the body until the condition is met.
        """
        while not self.visit(node.cond):
            self.visit(node.body)

    def visit_ForEach(self, node: ForEach):
        """
        Visit a for-each node and iterate over the iterable, executing the body for each item.
        """
        iterable = self.visit(node.iter)
        var_name = node.var.val

        for item in iterable:
            self.set_var(var_name, item)
            self.visit(node.body)

    def visit_ProcDef(self, node: ProcDef):
        """
        Visit a procedure definition node and store it in the current scope.
        """
        self.set_var(node.name.val, node)

    def visit_ProcCall(self, node: ProcCall):
        """
        Visit a procedure call node and execute the procedure with the provided arguments.
        """
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
        """
        Visit a return node and raise a ReturnException with the return value.
        """
        val = self.visit(node.val) if node.val else None
        raise ReturnException(val)

    def _builtin_display(self, args: list[Any]):
        """
        Built-in DISPLAY procedure to print values to the console.
        """
        print(*args)
        return None

    def _builtin_input(self, args: list[Any]):
        """
        Built-in INPUT procedure to read input from the user.
        """
        prompt = args[0] if args else ""
        try:
            val = input(prompt)
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            return val

    def _builtin_random(self, args: list[Any]):
        """
        Built-in RANDOM procedure to generate a random integer between two bounds.
        """
        import random

        if len(args) != 2:
            raise Exception("RANDOM takes exactly 2 arguments")
        return random.randint(args[0], args[1])

    def _builtin_append(self, args: list[Any]):
        """
        Built-in APPEND procedure to append an element to a list.
        """
        if len(args) != 2:
            raise Exception("APPEND takes exactly 2 arguments")
        lst, val = args
        if not isinstance(lst, list):
            raise Exception("First argument to APPEND must be a list")
        lst.append(val)
        return None

    def _builtin_insert(self, args: list[Any]):
        """
        Built-in INSERT procedure to insert an element at a specific index in a list.
        """
        if len(args) != 3:
            raise Exception("INSERT takes exactly 3 arguments")
        lst, idx, val = args
        if not isinstance(lst, list):
            raise Exception("First argument to INSERT must be a list")
        if not isinstance(idx, int):
            raise Exception("Second argument to INSERT must be an integer")
        lst.insert(idx - 1, val)
        return None

    def _builtin_remove(self, args: list[Any]):
        """
        Built-in REMOVE procedure to remove an element at a specific index from a list.
        """
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

    def _builtin_length(self, args: list[Any]):
        """
        Built-in LENGTH procedure to get the length of a list or string.
        """
        if len(args) != 1:
            raise Exception("LENGTH takes exactly 1 argument")
        return len(args[0])


if __name__ == "__main__":
    if len(argv) > 1:
        filename = argv[1]
        if not path.exists(filename):
            print(f"Error: File '{filename}' not found.")
            exit(1)
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()
    try:
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        interpreter = Interpreter()
        interpreter.visit(ast)
    except Exception as e:
        print(f"Runtime Error: {e}")
