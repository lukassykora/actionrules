"""Console script for action_rules."""
import action_rules

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for action_rules."""
    console.print("Replace this message by putting your code into "
               "action_rules.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
