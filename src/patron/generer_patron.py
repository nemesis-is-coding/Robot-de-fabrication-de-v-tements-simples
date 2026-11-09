"""Génération d'un patron de tote bag à partir de mesures (fonction F1).

Toutes les dimensions sont en millimètres. Chaque pièce est un rectangle
dont le contour inclut les marges de couture.
"""
import argparse
import json
import time
from dataclasses import dataclass, asdict


@dataclass
class Piece:
    nom: str
    quantite: int
    largeur: float  # dimension de coupe, marges incluses
    hauteur: float
    contour: list   # sommets (x, y), sens antihoraire, origine en bas à gauche


def _rectangle(largeur, hauteur):
    return [(0.0, 0.0), (largeur, 0.0), (largeur, hauteur), (0.0, hauteur)]


def generer_patron(largeur, hauteur, anse_longueur, anse_largeur,
                   marge=10.0, ourlet=30.0):
    """Retourne la liste des pièces à découper pour un tote bag.

    largeur, hauteur : taille finie du sac (mm)
    anse_longueur, anse_largeur : taille finie d'une anse (mm)
    marge : marge de couture (mm), ourlet : repli du haut du sac (mm)
    """
    for nom, valeur in [("largeur", largeur), ("hauteur", hauteur),
                        ("anse_longueur", anse_longueur),
                        ("anse_largeur", anse_largeur)]:
        if valeur <= 0:
            raise ValueError(f"{nom} doit être positif (reçu : {valeur})")
    if marge < 0 or ourlet < 0:
        raise ValueError("marge et ourlet doivent être positifs ou nuls")

    corps_l = largeur + 2 * marge
    corps_h = hauteur + marge + ourlet          # bas : marge, haut : ourlet
    anse_l = anse_longueur + 2 * marge
    anse_h = 2 * anse_largeur + 2 * marge       # bande pliée en deux

    return [
        Piece("corps", 2, corps_l, corps_h, _rectangle(corps_l, corps_h)),
        Piece("anse", 2, anse_l, anse_h, _rectangle(anse_l, anse_h)),
    ]


def exporter_svg(pieces, chemin, espace=20.0):
    """Écrit un SVG qui dispose une copie de chaque pièce côte à côte."""
    elements, x = [], espace
    hauteur_max = max(p.hauteur for p in pieces)
    for p in pieces:
        for _ in range(p.quantite):
            elements.append(
                f'<rect x="{x}" y="{espace}" width="{p.largeur}" '
                f'height="{p.hauteur}" fill="none" stroke="black"/>'
                f'<text x="{x + 5}" y="{espace + 15}" font-size="12">'
                f'{p.nom} {p.largeur:.0f}x{p.hauteur:.0f} mm</text>')
            x += p.largeur + espace
    largeur_svg, hauteur_svg = x, hauteur_max + 2 * espace
    with open(chemin, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{largeur_svg}mm" height="{hauteur_svg}mm" '
                f'viewBox="0 0 {largeur_svg} {hauteur_svg}">'
                + "".join(elements) + "</svg>")


def main():
    ap = argparse.ArgumentParser(description="Patron de tote bag (mm)")
    ap.add_argument("--largeur", type=float, default=380)
    ap.add_argument("--hauteur", type=float, default=420)
    ap.add_argument("--anse-longueur", type=float, default=600)
    ap.add_argument("--anse-largeur", type=float, default=30)
    ap.add_argument("--sortie", default="patron")
    a = ap.parse_args()

    t0 = time.perf_counter()
    pieces = generer_patron(a.largeur, a.hauteur, a.anse_longueur, a.anse_largeur)
    duree = time.perf_counter() - t0

    with open(a.sortie + ".json", "w") as f:
        json.dump([asdict(p) for p in pieces], f, indent=2)
    exporter_svg(pieces, a.sortie + ".svg")
    for p in pieces:
        print(f"{p.nom} x{p.quantite} : {p.largeur:.0f} x {p.hauteur:.0f} mm")
    print(f"Généré en {duree * 1000:.2f} ms -> {a.sortie}.json, {a.sortie}.svg")


if __name__ == "__main__":
    main()
