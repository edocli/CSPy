from .tokens import TokenType, Token
from .nodes import (
    AST,
    Num,
    Str,
    Bool,
    ListLit,
    ListAccess,
    Var,
    BinOp,
    UnaryOp,
    Assign,
    Block,
    IfStmt,
    RepeatTimes,
    RepeatUntil,
    ForEach,
    ProcDef,
    ProcCall,
    Return,
)

class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.curr = self.tokens[self.pos]

    def advance(self) -> None:
        self.pos += 1
        self.curr = self.tokens[self.pos] if self.pos < len(self.tokens) else Token(TokenType.EOF)

    def peek(self) -> Token:
        if self.pos + 1 < len(self.tokens):
            return self.tokens[self.pos + 1]
        return Token(TokenType.EOF)

    def skip_nl(self) -> None:
        while self.curr.type == TokenType.NEWLINE:
            self.eat(TokenType.NEWLINE)

    def eat(self, tp: TokenType) -> Token:
        if self.curr.type == tp:
            tok = self.curr
            self.advance()
            return tok
        raise Exception(f"Unexpected: {self.curr}, expected: {tp}")

    def block(self) -> AST:
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
        stmts = []
        while self.curr.type != TokenType.EOF:
            if self.curr.type == TokenType.NEWLINE:
                self.eat(TokenType.NEWLINE)
            else:
                stmts.append(self.stmt())
        return Block(stmts)

    def stmt(self) -> AST:
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
        self.eat(TokenType.FOR)
        self.eat(TokenType.EACH)
        var = Var(self.eat(TokenType.ID).val)
        self.eat(TokenType.IN)
        iter_ = self.expr()
        return ForEach(var, iter_, self.block())

    def proc_def(self) -> AST:
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
        self.eat(TokenType.RET)
        self.eat(TokenType.LPAREN)
        self.skip_nl()
        val = self.expr()
        self.skip_nl()
        self.eat(TokenType.RPAREN)
        return Return(val)

    def assign(self) -> AST:
        node = Var(self.eat(TokenType.ID).val)
        if self.curr.type == TokenType.LBRACKET:
            self.eat(TokenType.LBRACKET)
            idx = self.expr()
            self.eat(TokenType.RBRACKET)
            node = ListAccess(node, idx)
        self.eat(TokenType.ASSIGN)
        return Assign(node, self.expr())

    def expr(self) -> AST:
        node = self.comp()
        while self.curr.type in (TokenType.AND, TokenType.OR):
            op = self.eat(self.curr.type)
            node = BinOp(node, op, self.comp())
        return node

    def comp(self) -> AST:
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
        node = self.term()
        while self.curr.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.eat(self.curr.type)
            node = BinOp(node, op, self.term())
        return node

    def term(self) -> AST:
        node = self.factor()
        while self.curr.type in (TokenType.MUL, TokenType.DIV, TokenType.MOD):
            op = self.eat(self.curr.type)
            node = BinOp(node, op, self.factor())
        return node

    def factor(self) -> AST:
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
        if self.peek().type == TokenType.LPAREN:
            return self.proc_call()
        node = Var(self.eat(TokenType.ID).val)
        if self.curr.type == TokenType.LBRACKET:
            self.eat(TokenType.LBRACKET)
            idx = self.expr()
            self.eat(TokenType.RBRACKET)
            return ListAccess(node, idx)
        return node
