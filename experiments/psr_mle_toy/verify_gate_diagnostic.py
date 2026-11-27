"""Check the saved horizon and diagnostic mechanism results."""
import csv
from math import log
from pathlib import Path


def read(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def main():
    root=Path(__file__).parent/'results'
    base=root/'gated_horizon'
    rounds=read(base/'rounds.csv')
    finals=read(base/'finals.csv')
    assert len(finals)==18 and len(rounds)==165
    for row in rounds:
        assert row['certificate_valid']=='True'
        assert row['stage_ranks']=='1'+(';2'*(int(row['horizon'])-1))
        assert int(row['mle_calls'])<=int(row['true_rank_cap'])
        if row['stopped']=='False':
            assert row['progress_pass']=='True'
            assert row['cap_inequality_pass']=='True'
            assert (float(row['training_increment'])+
                    float(row['diagnostic_increment']))>=log(2)-1e-9
    for row in finals:
        assert row['stopped']=='True' and row['truth_fitted']=='True'
        assert row['sup_tv']=='0' and int(row['mle_calls'])==int(row['horizon'])+1
    for name in ('eps_0p1','eps_0p05'):
        for row in read(base/name/'finals.csv'):
            assert row['stopped']=='True' and int(row['mle_calls'])==int(row['horizon'])+1
    witness=read(root/'diagnostic_witness/witness.csv')
    assert len(witness)==3
    for row in witness:
        assert float(row['true_support_upper'])<float(row['training_threshold'])
        assert float(row['diagnostic_tv'])>=float(row['diagnostic_threshold'])
        assert float(row['diagnostic_logdet_increment'])>=log(65)
        assert row['candidate_excluded']=='True'
    print('18/18 gated runs and 3/3 diagnostic witnesses passed')


if __name__=='__main__':main()
