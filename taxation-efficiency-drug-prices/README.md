# Taxes, efficiency and the price of medicines

Public reproducibility companion for a series on Medium by Diogo Ribeiro. The editorial manuscripts are kept in the private [research-articles-medium](https://github.com/DiogoRibeiro7/research-articles-medium/tree/main/series/taxation-efficiency-drug-prices) repository; this folder holds the code, the printed numbers and the figures. The series starts from the slogan "taxation is theft", separates the premises of that argument that are normative from the one that can be measured, and then measures the efficiency claim where it is most concrete: what a single public buyer such as Portugal's SNS pays for medicines, and what that costs in other currencies (time to access, leakage through parallel export). Every number quoted in the articles is printed by the script in the corresponding folder, and the committed `output.txt` is what the script prints.

- **Part 1 — the premises of "taxation is theft", and the one that data can test** ([`part1/`](part1)). A linear public-goods economy. Voluntary contributions with imperfect conditional cooperation (Fischbacher and Gächter 2010), calibrated to the first-round and tenth-round contribution levels summarised by Ledyard (1995) and by Andreoni (1995) from Isaac and Walker (1988), with a sensitivity analysis over the parts the calibration does not identify; costly peer punishment calibrated to two results of Fehr and Gächter (2000) and checked out of sample against the 50-round result of Gächter, Renner and Sefton (2008); antisocial punishment (Herrmann, Thöni and Gächter 2008); a tax with a marginal cost of public funds from Browning (1976) and Ballard, Shoven and Whalley (1985) and a share of revenue that never reaches the good; the break-even frontier between voluntary provision and the tax; and a map of the group sizes and returns that laboratory studies have covered against a national public good.

## Running

Python 3.13 with the pinned versions in `requirements.txt`.

```
pip install -r requirements.txt
cd part1
python premises_numbers.py > output.txt   # about two minutes, single-threaded; writes figure_data.json
python figures.py
cd ..
python -m unittest discover -s tests      # tests of the shared tools in publicgoods.py
```

`publicgoods.py` holds the tools the parts share: the conditional-cooperation model, peer punishment with prosocial and antisocial sanctions, and the welfare algebra of voluntary provision against a tax (the break-even share `s_star` and the break-even marginal group return `g_star`).

Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` to reproduce the committed output exactly.

A GitHub Actions workflow (`.github/workflows/taxation-efficiency-drug-prices.yml`) reruns each part on every change and checks that the output matches the committed `output.txt`.

## Data

- **Part 1** downloads nothing. The behavioural model is calibrated to published summary figures, each quoted with its source in `part1/premises_numbers.py`: first-round contributions of 40–60 percent of the endowment (Ledyard 1995); 15–25 percent by the tenth round (Andreoni 1995, citing Isaac and Walker 1988); 82.5 percent of subjects giving their whole endowment in the final period of the partner treatment with punishment, and a total cost of punishment about as large as the benefit from the extra cooperation in that experiment (Fehr and Williams 2017, footnote 3). The laboratory designs in Section 6 are the cells reported by Isaac, Walker and Williams (1994, Table 1), Fehr and Gächter (2000) and Weimann et al. (2019). The model is not a reanalysis of any experiment's microdata.

## References

Andreoni, J. (1995). Cooperation in public-goods experiments: kindness or confusion? *American Economic Review*, 85(4), 891–904.

Ballard, C. L., Shoven, J. B., & Whalley, J. (1985). General equilibrium computations of the marginal welfare costs of taxes in the United States. *American Economic Review*, 75(1), 128–138.

Browning, E. K. (1976). The marginal cost of public funds. *Journal of Political Economy*, 84(2), 283–298.

Fehr, E., & Gächter, S. (2000). Cooperation and punishment in public goods experiments. *American Economic Review*, 90(4), 980–994.

Fehr, E., & Williams, T. (2017). Creating an efficient culture of cooperation. IZA Discussion Paper No. 11131.

Fischbacher, U., & Gächter, S. (2010). Social preferences, beliefs, and the dynamics of free riding in public goods experiments. *American Economic Review*, 100(1), 541–556.

Gächter, S., Renner, E., & Sefton, M. (2008). The long-run benefits of punishment. *Science*, 322(5907), 1510.

Herrmann, B., Thöni, C., & Gächter, S. (2008). Antisocial punishment across societies. *Science*, 319(5868), 1362–1367.

Isaac, R. M., & Walker, J. M. (1988). Group size effects in public goods provision: the voluntary contributions mechanism. *Quarterly Journal of Economics*, 103(1), 179–199.

Isaac, R. M., Walker, J. M., & Williams, A. W. (1994). Group size and the voluntary provision of public goods: experimental evidence utilizing large groups. *Journal of Public Economics*, 54(1), 1–36.

Ledyard, J. O. (1995). Public goods: a survey of experimental research. In J. H. Kagel & A. E. Roth (Eds.), *The Handbook of Experimental Economics* (pp. 111–194). Princeton University Press.

Weimann, J., Brosig-Koch, J., Heinrich, T., Hennig-Schmidt, H., & Keser, C. (2019). Public good provision by large groups – the logic of collective action revisited. *European Economic Review*, 118, 348–363.
