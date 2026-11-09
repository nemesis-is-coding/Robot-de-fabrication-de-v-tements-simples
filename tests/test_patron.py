import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "patron"))
from generer_patron import generer_patron


class TestPatron(unittest.TestCase):
    def test_dimensions_corps(self):
        corps = generer_patron(380, 420, 600, 30, marge=10, ourlet=30)[0]
        self.assertEqual(corps.nom, "corps")
        self.assertAlmostEqual(corps.largeur, 400.0, delta=1.0)   # 380 + 2*10
        self.assertAlmostEqual(corps.hauteur, 460.0, delta=1.0)   # 420 + 10 + 30

    def test_dimensions_anse(self):
        anse = generer_patron(380, 420, 600, 30, marge=10)[1]
        self.assertAlmostEqual(anse.largeur, 620.0, delta=1.0)    # 600 + 2*10
        self.assertAlmostEqual(anse.hauteur, 80.0, delta=1.0)     # 2*30 + 2*10

    def test_quantites(self):
        pieces = generer_patron(380, 420, 600, 30)
        self.assertEqual([p.quantite for p in pieces], [2, 2])

    def test_mesure_invalide(self):
        with self.assertRaises(ValueError):
            generer_patron(-1, 420, 600, 30)


if __name__ == "__main__":
    unittest.main()
