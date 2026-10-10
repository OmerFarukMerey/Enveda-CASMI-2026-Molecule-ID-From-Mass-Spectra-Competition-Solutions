"""Rule table and enumeration primitives of the generation study (train/gen/rules.py) and its evaluator (train/gen/gen_eval.py).

No competition data: structures are written here. What is pinned down:
  * every rule parses; an addition rule changes the formula of a product by exactly its declared delta (a removal rule may also match a
    substituted leaving group: those products are dropped by the formula check);
  * formula arithmetic (parse / fstr / minus) round-trips RDKit's formula strings; a negative count gives no parent formula;
  * one step: O-methylation of a phenol with the target formula of anisole yields anisole; products of another formula are dropped;
  * a positional move turns 2-methylphenol into 3- and 4-methylphenol and never returns the parent;
  * two steps: phenol reaches methoxyphenols through hydroxylation + methylation in either order;
  * gen_eval ranks distinct metric keys, takes the best score of a key, ranks missing candidates last, and reports value = share x MRR.
usage: python tests/test_gen_rules.py   (or pytest)
"""
import os, sys, tempfile

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'train', 'gen'))
import gen_eval, rules as R
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

can = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
PROBES = ['COc1cc(C=CC(=O)OC2C(O)C(O)C(CO)OC2Oc2ccc(O)c(O)c2)ccc1O', 'CC(=O)NC1CCC(O)C(C)C1OS(=O)(=O)O', 'CC(C)=CCc1c(O)cc2OC(C)(C)C=Cc2c1OC', 'CCCCCCCC=CCCCCCCCC(=O)NCC(=O)O',
          'OCC1OC(c2c(O)cc(O)c3c(=O)cc(-c4ccccc4)oc23)C(O)C(O)C1O', 'CC1OC(OC2C(O)C(O)COC2OC(=O)CC(=O)O)C(O)C(O)C1O', 'OC1C=CC2OC2C1N(C)C',
          'OCC1OC(Oc2ccccc2)C(O)C(O)C1O', 'CC(=O)OC1CCCCC1', 'O=C(OC1CCCCC1)c1ccccc1', 'O=C(C=Cc1ccc(O)c(O)c1)OC1CCCCC1', 'O=C(C=Cc1ccc(O)cc1)OC1CCCCC1',
          'O=C(OC1CCCCC1)c1cc(O)c(O)c(O)c1', 'O=C(O)C1OC(Oc2ccccc2)C(O)C(O)C1O', 'COC(=O)c1ccccc1C', 'OC1COC(Oc2ccccc2)C(O)C1O', 'OCC1OC(Oc2ccccc2)C(O)C1O',
          'CC(C)=CCOc1ccccc1', 'CC(=O)CCNC', 'CC1OC(c2ccccc2O)C(O)C(O)C1O', 'OC1COC(c2ccccc2O)C(O)C1O']


def test_every_rule_changes_the_formula_by_its_delta():
    used = set()
    for smi in PROBES:
        m = Chem.MolFromSmiles(smi); base = R.parse(rdMolDescriptors.CalcMolFormula(m))
        for name, (_, delta, _) in R.RULES.items():
            for pm, s in R.apply(m, name, cap=8):
                ok = R.fstr(R.minus(R.parse(rdMolDescriptors.CalcMolFormula(pm)), delta)) == R.fstr(base)
                # a removal rule may also match a substituted leaving group (decoumaroyl on a feruloyl ester): such products have another
                # formula and are dropped by the formula check of products_one; every other rule must be exact
                assert ok or min(delta.values()) < 0, (name, smi, s)
                if ok:
                    used.add(name)
    assert used == set(R.RULES), sorted(set(R.RULES) - used)    # the probes exercise the whole table


def test_formula_arithmetic():
    assert R.fstr(R.parse('C15H10O7')) == 'C15H10O7' and R.fstr(R.parse('C6H5ClO')) == 'C6H5ClO'
    assert R.fstr(R.minus(R.parse('C7H8O'), dict(C=1, H=2))) == 'C6H6O'
    assert R.fstr(R.minus(R.parse('C6H6O'), dict(S=1, O=3))) is None
    assert 'C6H6O' in R.one_step_formulas(R.parse('C7H8O')) and 'methyl_O' in R.one_step_formulas(R.parse('C7H8O'))['C6H6O']


def test_one_step_keeps_only_the_target_formula():
    f = R.one_step_formulas(R.parse('C7H8O'))['C6H6O']; prods = dict(R.products_one('Oc1ccccc1', f, 'C7H8O'))
    assert can('COc1ccccc1') in prods and prods[can('COc1ccccc1')] == 'methyl_O'
    assert all(rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(s)) == 'C7H8O' for s in prods)
    assert can('Cc1ccccc1O') in prods                          # C-methylation is a product of the same formula


def test_move_gives_positional_isomers_not_the_parent():
    prods = {s for s, _ in R.products_move('Cc1ccccc1O', 'C7H8O')}
    assert can('Cc1cccc(O)c1') in prods and can('Cc1ccc(O)cc1') in prods and can('Cc1ccccc1O') not in prods


def test_two_steps_either_order():
    pairs = R.two_step_formulas(R.parse('C7H8O2'))['C6H6O']; prods = {s for s, _ in R.products_pairs('Oc1ccccc1', pairs, 'C7H8O2')}
    assert can('COc1ccccc1O') in prods and can('COc1ccc(O)cc1') in prods
    assert ('methyl_O', 'demethyl_O') not in R.two_step_formulas(R.parse('C6H6O')).get('C6H6O', [])   # net-zero pairs are moves, not two steps


def test_gen_eval_ranks_keys_and_reports_value():
    C = pd.DataFrame(dict(molecule_id=['a'] * 4 + ['b'] * 2, candidate_key=['K1', 'K2', 'K2', 'K3', 'K4', 'K5'], smiles=list('uvwxyz'), source_rule='r', step='1',
                          ord=[1, 2, 3, 4, 1, 2], parent_pid=0, parent_rank=1, n_support=1, is_answer=[False, False, False, True, False, False], eset='N'))
    with tempfile.TemporaryDirectory() as d:
        c = d + '/gen_cands_x.parquet'; C.to_parquet(c, index=False)
        t = gen_eval.evaluate(None, c, budgets=(4,)).iloc[0]                 # the order: the answer is the third distinct key of 'a'; 'b' has no answer
        assert t.molecules == 2 and t.answer_in_B == 1 and abs(t.mrr_given - 1 / 3) < 1e-9 and abs(t.value_per_molecule - 1 / 6) < 1e-9
        s = pd.DataFrame(dict(molecule_id=['a', 'a'], candidate_key=['K3', 'K1'], score=[2.0, 1.0]))   # K2 is not scored: it ranks last
        t = gen_eval.evaluate(s, c, budgets=(4,)).iloc[0]; assert t.top1 == 1.0 and t.value_per_molecule == 0.5
        assert len(gen_eval.evaluate(None, c, budgets=(3,))) == 0               # the answer is outside the first three structures
        assert len(gen_eval.pairs(c)) == 5


if __name__ == '__main__':
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f):
            f()
    print('gen rules OK: formula deltas of every rule, one step, moves, two steps, the evaluator')
