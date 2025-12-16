from .tokens import TokenType, Token

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
