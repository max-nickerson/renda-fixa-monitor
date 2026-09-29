import zipfile, pandas as pd
z = zipfile.ZipFile("data/history/nightly/alt_signals/raw/oferta_distribuicao.zip")
print(z.namelist())
a = pd.read_csv(z.open("oferta_distribuicao.csv"), sep=";", encoding="latin-1", low_memory=False)
print(a.columns.tolist())
a = a[a["Tipo_Ativo"].str.contains("DEB", na=False)]
print(len(a)); print(a["Rito_Oferta"].value_counts() if "Rito_Oferta" in a else "")
dcols=[c for c in a.columns if c.startswith("Data")]
print(a[dcols].head(5).to_string())
for c in dcols:
    d=pd.to_datetime(a[c],errors="coerce"); print(c, d.notna().mean().round(3), d.min(), d.max())
st=pd.to_datetime(a["Data_Inicio_Oferta"],errors="coerce")
for c in dcols:
    d=pd.to_datetime(a[c],errors="coerce"); x=(d-st).dt.days.dropna(); 
    if len(x): print("lag",c, x.describe().round(0).to_dict())
b = pd.read_csv(z.open("oferta_resolucao_160.csv"), sep=";", encoding="latin-1", low_memory=False)
print(b.columns.tolist())
b=b[b["Valor_Mobiliario"].str.startswith("Deb",na=False)]
print(len(b)); print(b["Status_Requerimento"].value_counts()); print(b["Rito_Requerimento"].value_counts() if "Rito_Requerimento" in b else "")
dcols=[c for c in b.columns if c.startswith("Data")]
rq=pd.to_datetime(b["Data_requerimento"],errors="coerce")
for c in dcols:
    d=pd.to_datetime(b[c],errors="coerce"); x=(d-rq).dt.days.dropna()
    print(c, d.notna().mean().round(3), d.min(), d.max(), x.describe().round(0).to_dict() if len(x) else "")
print(pd.to_datetime(a["Data_Inicio_Oferta"],errors="coerce").dt.year.value_counts().sort_index())
print(rq.dt.year.value_counts().sort_index())
