"""Console script for keep_gpu."""
import keep_gpu

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for keep_gpu."""
    console.print("Replace this message by putting your code into "
               "keep_gpu.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
