from os import path
from sys import argv

from .lexer import Lexer
from .parser import Parser
from .interpreter import Interpreter


def main():
    code = ""
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


if __name__ == "__main__":
    main()
