import os
from typing import Iterable, Optional, Tuple, Union, Any, Dict

import pandas as pd

ENC_LIST = ("utf-8-sig", "cp932", "shift_jis")
CSV_CHUNK_ROWS = 200_000


def _build_enc_list(encoding: Optional[str]) -> Tuple[str, ...]:
    if encoding:
        seen = []
        for enc in (encoding,) + ENC_LIST:
            if enc not in seen:
                seen.append(enc)
        return tuple(seen)
    return ENC_LIST


def detect_csv_encoding(
    path: Union[str, os.PathLike],
    *,
    nrows: int = 0,
    usecols: Optional[Iterable[int]] = None,
    **kwargs: Any,
) -> str:
    # kwargs: dtype, engine, sep, etc.
    for enc in ENC_LIST:
        try:
            pd.read_csv(path, encoding=enc, nrows=nrows, usecols=usecols, **kwargs)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def _read_csv_with_enc_fallback(
    path: Union[str, os.PathLike],
    *,
    encoding: Optional[str] = None,
    errors_replace: bool = True,
    **kwargs: Any,
):
    """
    Internal helper: try encodings; for the final fallback optionally use errors="replace".
    Note: some pandas versions may not accept errors= for certain engines; we only pass it on fallback.
    """
    for enc in _build_enc_list(encoding):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue

    fallback = encoding or "utf-8"
    if errors_replace:
        return pd.read_csv(path, encoding=fallback, errors="replace", **kwargs)
    return pd.read_csv(path, encoding=fallback, **kwargs)


def read_csv_path(
    path: Union[str, os.PathLike],
    encoding: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    return _read_csv_with_enc_fallback(path, encoding=encoding, **kwargs)


def read_csv_head(
    path: Union[str, os.PathLike],
    nrows: int = 0,
    encoding: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    return _read_csv_with_enc_fallback(path, encoding=encoding, nrows=nrows, **kwargs)


def read_csv_first_n_cols(
    path: Union[str, os.PathLike],
    ncols: int,
    encoding: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    usecols = list(range(ncols))
    return _read_csv_with_enc_fallback(path, encoding=encoding, usecols=usecols, **kwargs)


def read_csv_first_n_cols_chunks(
    path: Union[str, os.PathLike],
    ncols: int,
    chunksize: int = CSV_CHUNK_ROWS,
    encoding: Optional[str] = None,
    **kwargs: Any,
) -> pd.io.parsers.TextFileReader:
    usecols = list(range(ncols))
    return _read_csv_with_enc_fallback(
        path,
        encoding=encoding,
        usecols=usecols,
        chunksize=chunksize,
        **kwargs,
    )
