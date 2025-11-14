from pathlib import Path
from typing import Annotated

import typer

from parsing.alg_memory_profiling import profile_memory_for_algs
from parsing.analyze import parse_analyze
from parsing.eval_online import parse_eval_online
from parsing.run_online import parse_run_online

app = typer.Typer()

@app.command()
def analyze_signal(config: Annotated[Path, typer.Option()]):
    parse_analyze(config)

@app.command()
def run(config: Annotated[Path, typer.Option()]):
    parse_run_online(config)

@app.command()
def eval(config: Annotated[Path, typer.Option()]):
    parse_eval_online(config)

@app.command()
def memory_profile():
    profile_memory_for_algs()

if __name__ == "__main__":
    app()