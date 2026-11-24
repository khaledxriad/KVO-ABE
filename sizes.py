"""Size model behind Figs. 11-13 and Table 10: pairing-based sizes (BLS12-381 element
sizes) against the lattice ciphertext of parameter set P-I.  Writes data/e5_sizes.csv.
Run params.py first (it writes data/params_sets.csv)."""
import csv

G1, GT, TAG, LAM = 48, 576, 32, 128

with open('data/params_sets.csv') as fh:
    rows = list(csv.DictReader(fh))
pI = [r for r in rows if r['set'] == 'P-I'][0]
m = int(float(pI['m']))
kappa = int(float(pI['K']))
elem = kappa / 8  # bytes per element of Z_q for q = 2^kappa

out = []
for L in (4, 8, 16, 32, 64):
    bsw = (2 * L + 1) * G1 + GT            # CP-ABE ciphertext: 2L+1 elements of G1, one of GT
    lai = 2 * bsw                          # verifiable variant: ciphertext and its redundant copy
    ghw = 2 * GT                           # outsourced download: two elements of GT
    ours = int((L + 1) * m * elem + LAM * elem + TAG)   # c_0, c_1..c_L, c_U and the tag
    dl = int(2 * m * elem + LAM * elem + TAG)           # c_0, psi_f, c_U and the tag
    out.append(dict(L=L, bsw07_ct=bsw, lai13_ct=lai, ghw11_download=ghw, ours_ct=ours,
                    ours_download=dl, lai_overhead_pct=100.0 * (lai - bsw) / bsw,
                    ours_overhead_pct=100.0 * TAG / ours))
with open('data/e5_sizes.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print('wrote data/e5_sizes.csv')
