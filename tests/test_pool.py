"""Pool build: masses, official keys, membership flags and exclusions on real molecules."""

from __future__ import annotations

from pathlib import Path
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rdkit import Chem

from scripts.build_pool import build
from src.pool import load_pool

ETHANOL, ACETONE, ENOL = "CCO", "CC(C)=O", "CC(O)=C"  # acetone/enol: one official key


def _key(smiles: str) -> str:
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def test_pool_keys_masses_flags_and_exclusions(tmp_path: Path) -> None:
    train = tmp_path / "train.parquet"
    pq.write_table(pa.table({
        "inchikey14": [_key(ETHANOL)[:14], _key(ACETONE)[:14]],
        "normalized_smiles": ["OCC", ACETONE],  # training SMILES wins over COCONUT's
    }), train)
    coconut = tmp_path / "coconut.zip"
    rows = [(_key(ETHANOL), ETHANOL), (_key(ENOL), ENOL), (_key("CCN"), "CCN.Cl"), ("BADBADBADBADBA-X-N", "not-a-smiles")]
    with zipfile.ZipFile(coconut, "w") as archive:
        archive.writestr("c.csv", "standard_inchi_key,canonical_smiles\n" + "".join(f"{k},{s}\n" for k, s in rows))
    output = tmp_path / "data" / "pool"
    manifest = build(train, coconut, output, workers=1)
    assert {key: manifest["counts"][key] for key in ("union", "multi_component_dropped", "unparseable_dropped", "rows")} == {
        "union": 5, "multi_component_dropped": 1, "unparseable_dropped": 1, "rows": 3}
    pool = load_pool(output / "pool.parquet")
    by_key = {key: i for i, key in enumerate(pool.key14.tolist())}
    ethanol = by_key[_key(ETHANOL)[:14]]
    assert pool.mass[ethanol] == pytest.approx(46.041865, abs=1e-5)
    assert pool.smiles[ethanol] == "OCC" and pool.in_train[ethanol] and pool.coconut[ethanol]
    acetone, enol = by_key[_key(ACETONE)[:14]], by_key[_key(ENOL)[:14]]
    assert pool.key14[acetone] != pool.key14[enol]
    assert pool.canonical14[acetone] == pool.canonical14[enol]  # one metric slot
    assert pool.in_train[acetone] and not pool.coconut[acetone]
    assert pool.coconut[enol] and not pool.in_train[enol]
    assert list(pool.mass) == sorted(pool.mass)
    assert pool.canonical_of_key() == {pool.key14[i]: pool.canonical14[i] for i in (ethanol, acetone)}
    with pytest.raises(ValueError, match="data/"):
        build(train, coconut, tmp_path / "elsewhere", workers=1)
