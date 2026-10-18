"""Print what the models actually write at a few points of the loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import bigram as B
import experiment as E


def render(st: E.Setup, doc: np.ndarray, k: int = 45) -> str:
    return " ".join(st.c.vocab[i].split("|", 1)[1] for i in doc[:k])


def main(data_dir: Path, out: Path, seed: int = 11):
    st = E.Setup(data_dir)
    lines = []
    for tau, show in [(1.0, (0, 10, 50)), (0.8, (0, 3, 6, 10))]:
        rng = np.random.default_rng([seed, int(tau * 10)])
        model = st.real
        for t in range(max(show) + 1):
            s = B.make_sampler(model, tau=tau)
            gen = B.generate(s, st.n_docs, E.DOC_LEN, rng)
            if t in show:
                lines.append(f"--- replace, temperature {tau}, text written by model {t}")
                for d in range(3):
                    lines.append(render(st, gen[d]))
            model = B.count_matrix(gen, st.V)
    text = "\n".join(lines)
    out.write_text(text + "\n")
    return text


if __name__ == "__main__":
    import sys
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    print(main(data, Path("results") / "samples.txt"))
