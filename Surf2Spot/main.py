import logging

import typer

from Surf2Spot.run import run_NB_preprocess, run_NB_craft, run_NB_predict, run_NB_draw, run_HS_preprocess, run_HS_craft, run_HS_predict, run_HS_draw

logging.basicConfig(level=logging.INFO)

CITATION = """
@article{Surf2Spot-Technical-Report,
	title        = {Surf2Spot: A Geometric Model for Prediction of Epitope and Binding Sites on Target Protein},
	author       = {{awzhao Discovery}},
	year         = 2025,
	journal      = {bioRxiv},
	publisher    = {},
	doi          = {},
	url          = {},
	elocation-id = {},
	eprint       = {}
}
""".strip()


def citation():
    """Print citation information"""
    typer.echo(CITATION)


def cli():
    app = typer.Typer()
    app.command("NB-preprocess", help="Run surfdiff to analyse single protein.")(run_NB_preprocess)
    app.command("NB-craft",help="Run surfdiff to analyse protein complex.")(run_NB_craft)
    app.command("NB-predict",help="Run surfdiff to analyse protein-ligand complex.")(run_NB_predict)
    app.command("NB-draw",help="Run surfdiff to analyse protein-ligand complex.")(run_NB_draw)
    
    app.command("HS-preprocess", help="Run surfdiff to analyse single protein.")(run_HS_preprocess)
    app.command("HS-craft",help="Run surfdiff to analyse protein complex.")(run_HS_craft)
    app.command("HS-predict",help="Run surfdiff to analyse protein-ligand complex.")(run_HS_predict)
    app.command("HS-draw",help="Run surfdiff to analyse protein-ligand complex.")(run_HS_draw)
    
    app.command("citation", help="Print citation information")(citation)
    app()


if __name__ == "__main__":
    cli()