from dataclasses import dataclass
from typing import Callable, Any

from .tokens import Token

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
