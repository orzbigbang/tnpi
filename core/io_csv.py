import os
from typing import Iterable, Optional, Tuple, Union

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
) -> str:
    for enc in ENC_LIST:
        try:
            pd.read_csv(path, encoding=enc, nrows=nrows, usecols=usecols)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def read_csv_path(path: Union[str, os.PathLike], encoding: Optional[str] = None) -> pd.DataFrame:
    for enc in _build_enc_list(encoding):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    fallback = encoding or "utf-8"
    return pd.read_csv(path, encoding=fallback, errors="replace")


def read_csv_head(
    path: Union[str, os.PathLike],
    nrows: int = 0,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    for enc in _build_enc_list(encoding):
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows)
        except UnicodeDecodeError:
            continue
    fallback = encoding or "utf-8"
    return pd.read_csv(path, encoding=fallback, errors="replace", nrows=nrows)


def read_csv_first_n_cols(
    path: Union[str, os.PathLike],
    ncols: int,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    usecols = list(range(ncols))
    for enc in _build_enc_list(encoding):
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols)
        except UnicodeDecodeError:
            continue
    fallback = encoding or "utf-8"
    return pd.read_csv(path, encoding=fallback, errors="replace", usecols=usecols)


def read_csv_first_n_cols_chunks(
    path: Union[str, os.PathLike],
    ncols: int,
    chunksize: int = CSV_CHUNK_ROWS,
    encoding: Optional[str] = None,
) -> pd.io.parsers.TextFileReader:
    usecols = list(range(ncols))
    for enc in _build_enc_list(encoding):
        try:
            return pd.read_csv(path, encoding=enc, usecols=usecols, chunksize=chunksize)
        except UnicodeDecodeError:
            continue
    fallback = encoding or "utf-8"
    return pd.read_csv(
        path,
        encoding=fallback,
        errors="replace",
        usecols=usecols,
        chunksize=chunksize,
    )
