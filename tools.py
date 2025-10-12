import subprocess

import typer


app = typer.Typer()


def run(cmd: str):
    subprocess.run(cmd, shell=True, check=True)


@app.command()
def lint():
    run("uv run ruff check --fix")


@app.command()
def format():
    run("uv run ruff format")


@app.command("fl")
def format_and_lint():
    format()
    lint()


@app.command()
def mypy():
    run("uv run mypy .")


@app.command("l")
def all_tasks():
    lint()
    format()
    mypy()


if __name__ == "__main__":
    app()
