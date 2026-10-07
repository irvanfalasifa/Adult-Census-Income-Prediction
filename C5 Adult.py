"""
C5.0 dari Python (rpy2) menggunakan DATA ONLINE tanpa download manual.
Revisi: Fokus 100% pada dataset Adult Income, loader robust, dan visualisasi PDF.
"""

import argparse
import io
import numpy as np
import pandas as pd
import requests

# ==== rpy2 (modern; tanpa pandas2ri.activate) ====
import rpy2.robjects as ro
from rpy2.robjects import r as R
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter

R('options(warn=-1)')
try:
    R('suppressPackageStartupMessages(library(C50))')
    R('if(!require("partykit", quietly=TRUE)) install.packages("partykit")')
except Exception:
    raise SystemExit("Paket R 'C50' atau 'partykit' bermasalah. Buka R lalu jalankan: install.packages(c('C50', 'partykit'))")

def to_r_df(pdf: pd.DataFrame):
    with localconverter(default_converter + pandas2ri.converter):
        return pandas2ri.py2rpy(pdf)

def coerce_obj_to_str(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str)
    return df

def fetch_text(url: str, timeout=20) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python/requests"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def load_adult():
    cols = [
        "age","workclass","fnlwgt","education","education_num","marital_status",
        "occupation","relationship","race","sex","capital_gain","capital_loss",
        "hours_per_week","native_country","income"
    ]

    train_urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "https://raw.githubusercontent.com/selva86/datasets/master/adult.csv",
    ]
    test_urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    ]

    tr_text, src_train = None, None
    last_err = None
    for u in train_urls:
        try:
            tr_text = fetch_text(u)
            src_train = u
            break
        except Exception as e:
            last_err = e
    if tr_text is None:
        raise SystemExit(f"Gagal mengambil adult.data dari semua URL. Terakhir: {last_err}")

    if src_train.endswith("adult.csv"):
        df = pd.read_csv(io.StringIO(tr_text))
        rename_map = {
            "age":"age","workclass":"workclass","fnlwgt":"fnlwgt","education":"education",
            "education.num":"education_num","marital.status":"marital_status","occupation":"occupation",
            "relationship":"relationship","race":"race","sex":"sex","capital.gain":"capital_gain",
            "capital.loss":"capital_loss","hours.per.week":"hours_per_week","native.country":"native_country",
            "income":"income"
        }
        df = df.rename(columns=rename_map)
        df = df[[c for c in rename_map.values() if c in df.columns]]
    else:
        tr = pd.read_csv(io.StringIO(tr_text), header=None, names=cols, na_values="?", skipinitialspace=True)
        te_text, src_test = None, None
        for u in test_urls:
            try:
                te_text = fetch_text(u)
                src_test = u
                break
            except Exception:
                pass

        if te_text:
            te = pd.read_csv(io.StringIO(te_text), header=0, names=cols, na_values="?", skipinitialspace=True, comment=None)
            df = pd.concat([tr, te], axis=0, ignore_index=True)
        else:
            df = tr

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    if "income" in df.columns:
        df["income"] = df["income"].str.replace(".", "", regex=False)

    df = df.replace({"nan": np.nan})
    before = len(df)
    df = df.dropna(axis=0).reset_index(drop=True)
    after = len(df)
    print(f"[Adult] drop NaN: {before - after} baris; sisa {after} baris. Sumber train: {src_train}")

    num_cols = ["age","fnlwgt","education_num","capital_gain","capital_loss","hours_per_week"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=0).reset_index(drop=True)

    TARGET = "income"
    return df, TARGET

def is_binary(series: pd.Series) -> bool:
    return len(series.dropna().unique()) == 2

def train_c50(df: pd.DataFrame, target: str, trials=25, minCases=10, CF=0.1, winnow=True, test_size=0.25, seed=42):
    from sklearn.model_selection import train_test_split

    df = df.dropna(axis=0).reset_index(drop=True)
    df = coerce_obj_to_str(df)

    X = df.drop(columns=[target])
    y = df[target]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    tr_pdf = Xtr.copy(); tr_pdf["__label__"] = ytr.values
    tr_r  = to_r_df(tr_pdf)
    te_r  = to_r_df(Xte.copy())

    ro.globalenv["tr"] = tr_r
    ro.globalenv["te"] = te_r
    ro.globalenv["winnow_flag"] = ro.BoolVector([bool(winnow)])
    ro.globalenv["trials_val"]  = ro.IntVector([int(trials)])
    ro.globalenv["minCases_val"]= ro.IntVector([int(minCases)])
    ro.globalenv["CF_val"]      = ro.FloatVector([float(CF)])

    R('''
        tr_y <- as.factor(tr$`__label__`)
        tr_x <- tr[, setdiff(names(tr), "__label__"), drop=FALSE]
        ctrl <- C5.0Control(minCases=minCases_val[1], CF=CF_val[1])
        model <- C5.0(x=tr_x, y=tr_y, trials=trials_val[1], winnow=winnow_flag[1], control=ctrl)
        p_class <- predict(model, te, type="class")
        p_prob  <- tryCatch(predict(model, te, type="prob"), error=function(e) NULL)
        s <- capture.output(summary(model))
        
        pdf("c50_tree_visualization.pdf", width=30, height=20)
        plot(model)
        dev.off()
    ''')
    
    y_pred = np.array(list(R('as.character(p_class)')))
    print("\n=== Ringkasan Model (C5.0) ===")
    print("\n".join(list(R('s'))))
    print("\n[INFO] Visualisasi pohon keputusan telah disimpan sebagai 'c50_tree_visualization.pdf' di folder direktori Anda.")

    print("\n=== Evaluasi Test ===")
    print("Accuracy:", __import__("sklearn.metrics").metrics.accuracy_score(yte.astype(str), y_pred.astype(str)))
    print("\nClassification report:\n", __import__("sklearn.metrics").metrics.classification_report(yte.astype(str), y_pred.astype(str), digits=4))
    print("Confusion matrix:\n", __import__("sklearn.metrics").metrics.confusion_matrix(yte.astype(str), y_pred.astype(str)))

    prob_df = None
    try:
        if is_binary(yte) and R('!is.null(p_prob)')[0]:
            prob_df = pd.DataFrame(np.array(R('as.matrix(p_prob)')),
                                   columns=list(R('colnames(p_prob)')))
            pos = sorted(prob_df.columns)[-1]
            y_true_bin = (yte.astype(str).values == pos).astype(int)
            auc = __import__("sklearn.metrics").metrics.roc_auc_score(y_true_bin, prob_df[pos].astype(float).values)
            print(f"ROC-AUC (pos='{pos}'):", round(float(auc), 4))
    except Exception:
        pass

    return yte, y_pred, prob_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1) 
    ap.add_argument("--mincases", type=int, default=10)
    ap.add_argument("--cf", type=float, default=0.1)
    ap.add_argument("--winnow", type=str, default="true")
    args = ap.parse_args()

    winnow = args.winnow.lower() in ("1","true","yes","y")

    print("[INFO] Memuat UCI Adult Income (online)…")
    df, target = load_adult()

    print(f"[INFO] Data shape: {df.shape}, target='{target}', kelas: {df[target].nunique()}")
    _y, _yp, _prob = train_c50(
        df, target,
        trials=args.trials, minCases=args.mincases, CF=args.cf, winnow=winnow
    )
    print("\nSelesai")

if __name__ == "__main__":
    main()