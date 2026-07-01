import argparse


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--model", default="lstm", choices=["lstm", "rnn"])
    parse.add_argument("--epochs", type=int, default=20)

    parse.add_argument("--vocab", default="vocab.txt")


if __name__ == '__main__':
    main()
