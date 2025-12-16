from dataclasses import dataclass
from enum import Enum

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
