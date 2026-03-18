import io

import numpy as np
import pandas as pd

from xrd_io import read_xrd_table_from_upload


def test_read_csv_with_headers():
    df = pd.DataFrame({"2Theta": np.linspace(10, 70, 50), "Intensity": np.linspace(100, 200, 50)})
    raw = df.to_csv(index=False).encode("utf-8")

    x, y = read_xrd_table_from_upload("sample.csv", raw)
    assert len(x) == len(y) == 50
    assert x[0] < x[-1]


def test_read_txt_without_headers_heuristic():
    x0 = np.linspace(5, 80, 60)
    y0 = np.sin(x0) * 10 + 100
    raw = "\n".join(f"{a:.4f} {b:.4f}" for a, b in zip(x0, y0)).encode("utf-8")

    x, y = read_xrd_table_from_upload("sample.txt", raw)
    assert len(x) == len(y) == 60
    assert np.isfinite(y).all()


def test_read_excel_xlsx():
    df = pd.DataFrame({"角度": np.linspace(10, 50, 30), "强度": np.linspace(10, 300, 30)})
    bio = io.BytesIO()
    df.to_excel(bio, index=False)

    x, y = read_xrd_table_from_upload("sample.xlsx", bio.getvalue())
    assert len(x) == len(y) == 30
    assert x[0] < x[-1]
