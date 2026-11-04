"""
Dashboard Analisis Sentimen - Cancel Culture Raffi Ahmad, RANS Carnival & Le Minerale
Jalankan:  streamlit run app.py
Data    :  Dataset_Final_143_Komentar_PDF_Cancel_Culture_Raffi_RANS_LeMinerale.xlsx
           (letakkan satu folder dengan app.py, atau unggah lewat sidebar)
"""
import difflib
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
DEFAULT_FILE = BASE / "data_komentar.xlsx"  # dibaca otomatis, tidak perlu unggah

st.set_page_config(page_title="Cancel Culture Raffi - RANS - Le Minerale", layout="wide")

# ---------------------------------------------------------------- data
COL = {
    "no": "No",
    "akun": "Akun Kreator TikTok",
    "link": "Link Postingan / Akun TikTok",
    "hal": "Halaman PDF & Sumber",
    "tahap": "Tahap Kausalitas Cancel Culture",
    "user": "Kode Responden / Username (Anonim)",
    "isi": "Isi Komentar (Verbatim sesuai PDF)",
    "waktu": "Waktu",
    "likes": "Jumlah Likes",
    "replies": "Jumlah Replies",
    "sent": "Sentimen",
    "sub": "Sub-Kategori Narasi",
    "indikasi": "Indikasi Akun",
}


@st.cache_data
def load(src) -> pd.DataFrame:
    df = pd.read_excel(src, sheet_name=0, header=3)
    df = df[pd.to_numeric(df[COL["no"]], errors="coerce").notna()].copy()  # buang baris TOTAL
    df[COL["likes"]] = pd.to_numeric(df[COL["likes"]], errors="coerce").fillna(0)
    df[COL["replies"]] = pd.to_numeric(df[COL["replies"]], errors="coerce").fillna(0)
    df[COL["sent"]] = df[COL["sent"]].replace({"Positif (Defensif)": "Pro-Raffi / Defensif"})
    df["Tahap"] = df[COL["tahap"]].str.extract(r"^(Tahap \w+)")[0]
    df["Tahap (lengkap)"] = df[COL["tahap"]]
    df["Akun"] = df[COL["akun"]].str.replace(r"\s*\(.*\)$", "", regex=True)  # "Klip Boikot Le Minerale"
    df["Kode"] = df[COL["user"]].astype(str).str.extract(r"^(RES-\d+)")[0]
    df["Kelompok"] = df[COL["indikasi"]].map(
        lambda x: "Organik" if x == "Organik" else "Defensif / Indikasi Buzzer"
    )
    return df


def _angka(txt, label):
    """'1.4M Views' -> 1.4e6 ; '3.502 Komentar' -> 3502 ; '79.6K Likes' -> 79600"""
    m = re.search(r"([\d.,]+)\s*([KM]?)\s*" + label, str(txt), flags=re.I)
    if not m:
        return None
    num, suf = m.group(1), m.group(2).upper()
    if suf:
        return float(num.replace(",", ".")) * (1e3 if suf == "K" else 1e6)
    return float(num.replace(".", "").replace(",", ""))


@st.cache_data
def load_video(src) -> pd.DataFrame:
    v = pd.read_excel(src, sheet_name=2, header=3)
    v = v[v["Halaman PDF"].notna()].copy()
    out = pd.DataFrame(
        {
            "Halaman PDF": v["Halaman PDF"],
            "Akun / sumber": v["Akun TikTok"],
            "Tahap": v["Tahap Kausalitas"],
            "Views": v["Metrik Video di PDF"].map(lambda t: _angka(t, "Views")),
            "Likes video": v["Metrik Video di PDF"].map(lambda t: _angka(t, "Likes")),
            "Komentar di platform": v["Metrik Video di PDF"].map(lambda t: _angka(t, "Komentar")),
            "Shares": v["Metrik Video di PDF"].map(lambda t: _angka(t, "Shares")),
            "Metrik asli": v["Metrik Video di PDF"],
            "Link": v["Link Postingan / Akun TikTok"].map(lambda u: u if str(u).startswith("http") else None),
        }
    )
    return out


STOP = set(
    """yang dan di ke dari ini itu untuk dengan ga gak nggak tidak aja saja sih deh dong kok lah nih
    udah sudah jadi kalau kalo karena juga atau pada ada buat bisa mau tapi tp yg dg sama banget bgt
    si ya nya the to of in is a aku gue gw lo lu kita kami dia mereka lebih sudah masih akan pun
    para oleh saat satu apa gimana kenapa kan lagi hanya cuma sampai sampe udh udah emang""".split()
)


def top_words(texts, n=20):
    c = Counter()
    for t in texts:
        for w in re.findall(r"[a-zA-Z]{3,}", str(t).lower()):
            if w not in STOP:
                c[w] += 1
    return pd.DataFrame(c.most_common(n), columns=["Kata", "Frekuensi"])


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * x)) / (n * x.sum()) - (n + 1) / n)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def norm_text(t):
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower()).strip()


def duplikat(frame, ambang=0.8):
    """Pasangan komentar yang hampir identik (indikasi template/copy-paste)."""
    rows = frame.reset_index(drop=True)
    teks = [norm_text(t) for t in rows[COL["isi"]]]
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if len(teks[i]) < 20 or len(teks[j]) < 20:
                continue
            r = difflib.SequenceMatcher(None, teks[i], teks[j]).ratio()
            if r >= ambang:
                out.append((round(r, 2), rows.loc[i, COL["isi"]], rows.loc[j, COL["isi"]]))
    return pd.DataFrame(out, columns=["Kemiripan", "Komentar A", "Komentar B"]).sort_values("Kemiripan", ascending=False)


LMN_RULES = [
    ("Mengaitkan atribut biru dengan sponsor", r"biru|bingkisan|totebag|tote bag|gapura|tenda"),
    ("Pindah / tidak beli lagi", r"pindah|ganti|cleo|pristine|aqua|aquviva|club|alternatif|rekomendasi|purelife|gak laku|g laku|air minum|korupsi"),
    ("Membantah klarifikasi / curiga", r"kerja ?sama|bohong|klarifikasi|gak mungkin|ga mungkin|nggak mungkin"),
    ("Kecewa / seruan boikot", r"boikot|boikut|kecewa|nyesel|rezim|sponsor"),
]


def kategori_lmn(t):
    for nama, pola in LMN_RULES:
        if re.search(pola, str(t), flags=re.I):
            return nama
    return "Lainnya"


def waktu_ke_hari(w):
    w = str(w)
    m = re.match(r"(\d+)d ago", w)
    if m:
        return int(m.group(1))
    if "h ago" in w:
        return 0
    return None  # format tanggal (mis. 10-3) tidak dipakai


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Sumber & Filter")
up = st.sidebar.file_uploader("Ganti dataset (opsional)", type="xlsx")
src = up if up else (DEFAULT_FILE if DEFAULT_FILE.exists() else None)
if src is None:
    st.error("File data_komentar.xlsx tidak ditemukan di folder aplikasi. Unggah lewat sidebar.")
    st.stop()

df = load(src)

tahap_opts = sorted(df["Tahap (lengkap)"].unique())
f_tahap = st.sidebar.multiselect("Tahap kausalitas", tahap_opts, default=tahap_opts)
akun_opts = sorted(df["Akun"].unique())
f_akun = st.sidebar.multiselect("Akun / sumber video", akun_opts, default=akun_opts)
f_kel = st.sidebar.multiselect(
    "Kelompok", ["Organik", "Defensif / Indikasi Buzzer"], default=["Organik", "Defensif / Indikasi Buzzer"]
)
st.sidebar.caption("Data komentator sudah dianonimkan (kode RES-xxx). Dashboard hanya menampilkan kode responden.")

d = df[df["Tahap (lengkap)"].isin(f_tahap) & df["Akun"].isin(f_akun) & df["Kelompok"].isin(f_kel)]
if d.empty:
    st.error("Tidak ada data untuk filter ini.")
    st.stop()

# ---------------------------------------------------------------- header
st.title("Cancel Culture Raffi Ahmad, RANS Carnival & Le Minerale")
st.caption(
    "Analisis dataset komentar TikTok hasil ekstraksi dari PDF tangkapan layar. "
    "Label sentimen dan indikasi akun berasal dari pelabelan dalam dataset (bukan model otomatis)."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Komentar", f"{len(d):,}")
c2.metric("Total likes", f"{int(d[COL['likes']].sum()):,}")
c3.metric("Total replies", f"{int(d[COL['replies']].sum()):,}")
c4.metric("% Negatif", f"{(d[COL['sent']].eq('Negatif').mean() * 100):.1f}%")
c5.metric("% Defensif / buzzer", f"{(d['Kelompok'].ne('Organik').mean() * 100):.1f}%")

tab1, tab2, tab3, tab4, tab6, tab7, tab5 = st.tabs(
    ["Sentimen", "Rantai kausalitas", "Sumber video", "Narasi & kata", "Validitas & Keterbatasan",
     "Daftar video & link", "Data mentah"]
)

# ---------------------------------------------------------------- tab 1
with tab1:
    a, b = st.columns(2)
    sent_n = d[COL["sent"]].value_counts().reset_index()
    sent_n.columns = ["Sentimen", "Jumlah"]
    a.plotly_chart(
        px.pie(sent_n, names="Sentimen", values="Jumlah", hole=0.45, title="Proporsi sentimen (jumlah komentar)"),
        use_container_width=True,
    )
    sent_l = d.groupby(COL["sent"])[COL["likes"]].sum().reset_index()
    b.plotly_chart(
        px.pie(sent_l, names=COL["sent"], values=COL["likes"], hole=0.45, title="Proporsi sentimen (bobot likes)"),
        use_container_width=True,
    )

    kel = d.groupby("Kelompok").agg(
        Komentar=(COL["no"], "count"),
        Total_Likes=(COL["likes"], "sum"),
        Rata_Likes=(COL["likes"], "mean"),
        Median_Likes=(COL["likes"], "median"),
    ).reset_index()
    st.subheader("Organik vs defensif / indikasi buzzer")
    st.dataframe(kel.round(1), use_container_width=True, hide_index=True)
    st.plotly_chart(
        px.bar(
            kel.melt(id_vars="Kelompok", value_vars=["Rata_Likes", "Median_Likes"]),
            x="Kelompok", y="value", color="variable", barmode="group",
            title="Rata-rata vs median likes per kelompok", labels={"value": "Likes", "variable": ""},
        ),
        use_container_width=True,
    )
    st.info(
        "Komentar defensif hanya sebagian kecil dari total likes. Namun jumlah komentar yang kecil dan "
        "pelabelan 'indikasi buzzer' bersifat dugaan, sehingga di makalah tulis sebagai indikasi, bukan bukti."
    )

# ---------------------------------------------------------------- tab 2
with tab2:
    ag = d.groupby("Tahap (lengkap)").agg(
        Komentar=(COL["no"], "count"),
        Total_Likes=(COL["likes"], "sum"),
        Median_Likes=(COL["likes"], "median"),
        Replies=(COL["replies"], "sum"),
    ).reset_index()
    ag["Porsi likes (%)"] = (ag["Total_Likes"] / ag["Total_Likes"].sum() * 100).round(1)
    ag["Tahap"] = ag["Tahap (lengkap)"].str.extract(r"^(Tahap \w+)")[0]
    ag = ag.sort_values("Tahap")

    a, b = st.columns(2)
    a.plotly_chart(
        px.bar(ag, x="Tahap", y="Komentar", text="Komentar", title="Jumlah komentar per tahap"),
        use_container_width=True,
    )
    b.plotly_chart(
        px.bar(ag, x="Tahap", y="Total_Likes", text="Porsi likes (%)", title="Total likes per tahap (label = % porsi)"),
        use_container_width=True,
    )

    fun = go.Figure(
        go.Funnel(y=ag["Tahap (lengkap)"], x=ag["Total_Likes"], textinfo="value+percent total")
    )
    fun.update_layout(title="Distribusi bobot likes sepanjang rantai kausalitas")
    st.plotly_chart(fun, use_container_width=True)

    sub = d.groupby(["Tahap", COL["sent"]]).size().reset_index(name="Jumlah")
    st.plotly_chart(
        px.bar(sub, x="Tahap", y="Jumlah", color=COL["sent"], barmode="stack", title="Sentimen per tahap"),
        use_container_width=True,
    )
    st.dataframe(ag.drop(columns="Tahap"), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- tab 3
with tab3:
    ak = d.groupby("Akun").agg(
        Komentar=(COL["no"], "count"),
        Total_Likes=(COL["likes"], "sum"),
        Median_Likes=(COL["likes"], "median"),
        Max_Likes=(COL["likes"], "max"),
    ).reset_index().sort_values("Total_Likes", ascending=True)
    a, b = st.columns(2)
    a.plotly_chart(
        px.bar(ak, x="Total_Likes", y="Akun", orientation="h", title="Total likes komentar per akun"),
        use_container_width=True,
    )
    b.plotly_chart(
        px.bar(ak.sort_values("Komentar"), x="Komentar", y="Akun", orientation="h", title="Jumlah komentar per akun"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.scatter(
            d, x=COL["replies"], y=COL["likes"], color="Akun", log_y=True,
            hover_data=[COL["isi"]], title="Likes vs replies per komentar (skala log pada likes)",
        ),
        use_container_width=True,
    )
    st.caption(
        "Distribusi likes sangat timpang: beberapa komentar viral menyumbang sebagian besar likes. "
        "Gunakan median selain rata-rata, dan sebutkan hal ini sebagai keterbatasan."
    )

    hari = d[COL["waktu"]].map(waktu_ke_hari).dropna()
    if not hari.empty:
        hh = hari.astype(int).map(lambda x: "<1 hari" if x == 0 else f"{x} hari lalu").value_counts().reset_index()
        hh.columns = ["Usia komentar saat ditangkap", "Jumlah"]
        st.plotly_chart(
            px.bar(hh, x="Usia komentar saat ditangkap", y="Jumlah", title="Usia komentar saat tangkapan layar"),
            use_container_width=True,
        )

# ---------------------------------------------------------------- tab 4
with tab4:
    a, b = st.columns(2)
    sub_n = d[COL["sub"]].value_counts().head(15).sort_values().reset_index()
    sub_n.columns = ["Sub-kategori", "Jumlah"]
    a.plotly_chart(
        px.bar(sub_n, x="Jumlah", y="Sub-kategori", orientation="h", title="15 sub-kategori narasi teratas"),
        use_container_width=True,
    )
    tw = top_words(d[COL["isi"]], 20).sort_values("Frekuensi")
    b.plotly_chart(
        px.bar(tw, x="Frekuensi", y="Kata", orientation="h", title="20 kata paling sering muncul"),
        use_container_width=True,
    )

    kw = {
        "boikot": r"boikot|boikut",
        "mahasewa / bayaran": r"mahasewa|sewa|bayar|gaji|50rb|50 ribu",
        "buzzer": r"buzzer",
        "Le Minerale": r"le ?min",
        "RANS / carnival": r"rans|carnival",
        "Willie Salim": r"willie|wili|willy",
        "Raffi": r"raffi|rapi",
    }
    rows = [
        (k, int(d[COL["isi"]].astype(str).str.contains(p, case=False, regex=True).sum())) for k, p in kw.items()
    ]
    kdf = pd.DataFrame(rows, columns=["Topik", "Komentar yang menyebut"]).sort_values("Komentar yang menyebut")
    st.plotly_chart(
        px.bar(kdf, x="Komentar yang menyebut", y="Topik", orientation="h", title="Frekuensi penyebutan topik kunci"),
        use_container_width=True,
    )

    st.subheader("10 komentar dengan likes tertinggi")
    cols = [COL["akun"], COL["isi"], COL["likes"], COL["replies"], COL["sent"]]
    cols.insert(1, "Kode")
    st.dataframe(d.nlargest(10, COL["likes"])[cols], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- tab 6 (validitas)
with tab6:
    L = COL["likes"]
    st.subheader("1. Ketimpangan likes (mengapa rata-rata menyesatkan)")
    g = gini(d[L])
    top5 = d[L].nlargest(5).sum() / d[L].sum() * 100
    top10p = d[L].nlargest(max(1, int(len(d) * 0.1))).sum() / d[L].sum() * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Koefisien Gini likes", f"{g:.2f}")
    m2.metric("Porsi 5 komentar teratas", f"{top5:.1f}%")
    m3.metric("Porsi 10% komentar teratas", f"{top10p:.1f}%")
    m4.metric("Median vs rata-rata", f"{d[L].median():,.0f} vs {d[L].mean():,.0f}")

    x = np.sort(d[L].values)
    lor = np.insert(np.cumsum(x) / x.sum(), 0, 0)
    pop = np.linspace(0, 1, len(lor))
    fig = go.Figure()
    fig.add_scatter(x=pop, y=lor, name="Distribusi likes aktual")
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Pemerataan sempurna", line=dict(dash="dash"))
    fig.update_layout(title="Kurva Lorenz likes", xaxis_title="Proporsi komentar (terendah ke tertinggi)",
                      yaxis_title="Proporsi kumulatif likes")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2. Uji ketahanan: apakah dominasi suatu tahap hanya karena beberapa komentar viral?")
    cap = d[L].quantile(0.95)
    base = d.copy()
    base["Asli"] = base[L]
    base["Dibatasi (persentil 95)"] = base[L].clip(upper=cap)
    base["Tanpa @kompascom"] = np.where(base["Akun"] == "@kompascom", 0, base[L])
    rob = base.groupby("Tahap")[["Asli", "Dibatasi (persentil 95)", "Tanpa @kompascom"]].sum()
    rob = (rob / rob.sum() * 100).round(1).reset_index().melt(id_vars="Tahap", var_name="Skenario", value_name="Porsi likes (%)")
    st.plotly_chart(
        px.bar(rob, x="Tahap", y="Porsi likes (%)", color="Skenario", barmode="group",
               title="Porsi likes per tahap pada tiga skenario pembobotan"),
        use_container_width=True,
    )
    st.caption("Jika urutan tahap berubah drastis antar skenario, kesimpulan berbasis likes tidak stabil. "
               "Laporkan jumlah komentar dan median, bukan hanya total likes.")

    st.subheader("3. Selang kepercayaan proporsi (sampel kecil, bukan acak)")
    n = len(d)
    k_neg = int((d[COL["sent"]] == "Negatif").sum())
    k_def = int((d["Kelompok"] != "Organik").sum())
    rows = []
    for nama, k in [("Negatif", k_neg), ("Defensif / indikasi buzzer", k_def)]:
        lo, hi = wilson(k, n)
        rows.append({"Proporsi": nama, "k": k, "n": n, "Estimasi (%)": round(k / n * 100, 1),
                     "CI 95% bawah (%)": round(lo * 100, 1), "CI 95% atas (%)": round(hi * 100, 1)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Interval Wilson. Hanya menggambarkan ketidakpastian pengambilan sampel; tidak memperbaiki bias pemilihan akun/komentar.")

    st.subheader("4. Seberapa kuat bukti 'buzzer'? (kelompok defensif)")
    dfn = d[d["Kelompok"] != "Organik"]
    if dfn.empty:
        st.write("Kelompok defensif tidak ada pada data terfilter.")
    else:
        st.write(
            f"Kelompok defensif berisi {len(dfn)} komentar, namun isinya heterogen: dukungan moral kepada Raffi, "
            "pembelaan terhadap Le Minerale, komentar ekonomi pedagang, hingga komentar politik. "
            "Tidak semuanya otomatis buzzer."
        )
        st.dataframe(
            dfn[[COL["isi"], COL["sub"], COL["indikasi"], COL["likes"]]],
            use_container_width=True, hide_index=True,
        )
        ambang = st.slider("Ambang kemiripan teks (0 = longgar, 1 = identik)", 0.3, 0.9, 0.4, 0.05)
        dup = duplikat(dfn, ambang)
        if dup.empty:
            st.write("Tidak ada pasangan komentar yang semirip itu.")
        else:
            st.dataframe(dup, use_container_width=True, hide_index=True)
        st.info(
            "Temuan penting: tidak ada komentar yang copy-paste verbatim. Kemiripannya hanya tematik "
            "(contoh: 'Semangat Raffi... biarin difitnah') dan bernilai sedang (sekitar 0,47). Pola ini konsisten "
            "dengan dugaan koordinasi, tetapi juga dengan dukungan penggemar yang tulus. Gunakan istilah "
            "'komentar pro-Raffi/defensif' sebagai istilah utama, dan sebut 'indikasi buzzer' hanya sebagai dugaan "
            "yang bersumber dari pelabelan dan pemberitaan, bukan dari analisis teks ini."
        )

    st.subheader("5. Tahap 5 (Le Minerale): apa yang sebenarnya terbukti?")
    t5 = d[d["Tahap"] == "Tahap 5"].copy()
    if t5.empty:
        st.write("Tahap 5 tidak ada pada data terfilter.")
    else:
        t5["Kategori (aturan kata kunci)"] = t5[COL["isi"]].map(kategori_lmn)
        k5 = t5.groupby("Kategori (aturan kata kunci)").agg(
            Komentar=(COL["no"], "count"), Total_Likes=(L, "sum")).reset_index()
        st.plotly_chart(px.bar(k5, x="Kategori (aturan kata kunci)", y="Komentar", text="Komentar",
                               title=f"Isi komentar Tahap 5 (n = {len(t5)})"), use_container_width=True)
        st.warning(
            "Data ini menunjukkan persepsi dan seruan boikot di kolom komentar. Data ini TIDAK menunjukkan "
            "penurunan penjualan atau kerugian finansial Le Minerale. Pihak Le Minerale juga membantah "
            "menjadi sponsor acara. Untuk klaim kerugian, tambahkan data pembanding (laporan keuangan, data "
            "penjualan, atau pemberitaan ekonomi). Kategori di atas berbasis kata kunci, jadi periksa manual."
        )

    st.subheader("6. Kualitas data")
    bad_time = int(d[COL["waktu"]].map(waktu_ke_hari).isna().sum())
    st.write(f"- Komentar dengan format waktu tidak terbaca (mis. '10-3'): **{bad_time}**. Cek ulang di PDF sebelum dipakai untuk analisis waktu.")
    st.write(f"- Komentar dari tangkapan layar, bukan hasil scraping resmi; jumlah dan urutan bergantung pada apa yang tampil di layar.")
    st.write("- Label 'Pro-Raffi / Defensif' menggantikan 'Positif (Defensif)' agar tidak terbaca sebagai sentimen positif umum.")

    st.subheader("Saran penulisan di makalah")
    st.markdown(
        """
1. **Metode:** sebut sebagai analisis konten kualitatif-deskriptif atas sampel purposif, bukan survei representatif.
2. **Angka:** laporkan N, median, dan CI selain total likes; hindari klaim "mayoritas publik".
3. **Buzzer:** gunakan frasa "diduga/indikasi", didukung pola template dan pemberitaan media.
4. **Dampak:** pisahkan *terbukti* (pembatalan musisi/tenant dan sepinya acara menurut media), *dugaan* (peserta/penonton bayaran), dan *belum terbukti* (kerugian Le Minerale).
5. **Etika:** samarkan username komentator dan cantumkan tanggal pengambilan data.
6. **Keterbatasan:** sampel 143 komentar dari 9 akun, bias algoritma TikTok, dan label ditentukan manual.
"""
    )


# ---------------------------------------------------------------- tab 7 (video & link)
with tab7:
    try:
        vv = load_video(src)
    except Exception:
        vv = pd.DataFrame()
    if vv.empty:
        st.info("Sheet pemetaan video tidak ditemukan pada dataset ini.")
    else:
        st.subheader("Video sumber data")
        st.caption("Metrik diambil dari tangkapan layar PDF saat pengambilan data; tidak semua video menampilkan views.")
        urut = vv.sort_values(["Views", "Komentar di platform"], ascending=False, na_position="last")
        st.dataframe(
            urut[["Halaman PDF", "Akun / sumber", "Tahap", "Views", "Likes video", "Komentar di platform", "Shares", "Link"]],
            use_container_width=True, hide_index=True,
            column_config={"Link": st.column_config.LinkColumn("Link TikTok", display_text="Buka"),
                           "Views": st.column_config.NumberColumn(format="%d"),
                           "Likes video": st.column_config.NumberColumn(format="%d"),
                           "Komentar di platform": st.column_config.NumberColumn(format="%d"),
                           "Shares": st.column_config.NumberColumn(format="%d")},
        )
        kb = vv.dropna(subset=["Komentar di platform"]).copy()
        kb["Akun / sumber"] = kb["Akun / sumber"].str.split(" / ").str[0]
        kb = kb.sort_values("Komentar di platform")
        st.plotly_chart(
            px.bar(kb, x="Komentar di platform", y="Akun / sumber", orientation="h",
                   title="Jumlah komentar pada tiap video (di platform)"),
            use_container_width=True,
        )
        st.caption("Dataset ini hanya memuat sampel 143 komentar dari total puluhan ribu komentar pada video-video tersebut.")


# ---------------------------------------------------------------- tab 5
with tab5:
    cols = [COL["no"], COL["akun"], "Tahap (lengkap)", COL["isi"], COL["waktu"], COL["likes"],
            COL["replies"], COL["sent"], COL["sub"], COL["indikasi"], COL["link"]]
    cols.insert(3, "Kode")
    view = d[cols]
    st.dataframe(view, use_container_width=True, hide_index=True,
                 column_config={COL["link"]: st.column_config.LinkColumn("Link TikTok")})
    st.download_button(
        "Unduh data terfilter (CSV)",
        view.to_csv(index=False).encode("utf-8-sig"),
        file_name="komentar_terfilter.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Keterbatasan: sampel kecil (143 komentar) dari 9 akun, bukan sampel acak; likes sangat timpang; "
    "label buzzer berupa dugaan; hubungan sebab-akibat antar tahap bersifat naratif, bukan terbukti secara statistik."
)
