"""Daily B3 equity history for debenture issuers (direct listings and listed parents).

Outputs
  research/data/equity_map.csv         cnpj8 -> ticker(s), mapping_type direct|parent|none
  data/history/equity_raw/<T>.json     raw brapi quote responses (range=10y, interval=1d)
  data/history/equity_raw/cotahist_<Y>.csv.gz   B3 COTAHIST extract (fallback for delisted names)
  data/history/equity_daily.pkl        long [ticker, date, close, adj_close, volume, source]

Re-runnable: brapi JSON cached (refreshed if older than --max-age-h), COTAHIST past years cached
forever, current year refreshed daily.  Run:
  PYTHONPATH=. PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe research/data_equity.py
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import time
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from rfmonitor.config import ROOT, settings
from rfmonitor.http import get
from rfmonitor.sources import brapi, cvm, snd

RAW = ROOT / "data" / "history" / "equity_raw"
OUT_PKL = ROOT / "data" / "history" / "equity_daily.pkl"
MAP_CSV = ROOT / "research" / "data" / "equity_map.csv"
LAB = ROOT / "data" / "history" / "lab_weekly.pkl"
START = pd.Timestamp("2020-06-01")
COTAHIST_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{y}.ZIP"

# ---------------------------------------------------------------------------------------------
# Curated mapping.  cnpj8 -> [(ticker, mapping_type, confidence, note)].  ticker None = reviewed,
# no listed equity (unlisted / foreign parent / fragmented JV).  Delisted tickers are kept: they
# are fetched from B3 COTAHIST up to the delisting date.
# ---------------------------------------------------------------------------------------------
D, P = "direct", "parent"
NEO = ("NEOE3", P, "high", "Neoenergia (Iberdrola) group; NEOE3 delisted after 2025/26 OPA -> COTAHIST")
ENGI = ("ENGI11", P, "high", "Energisa group subsidiary")
EQTL = ("EQTL3", P, "high", "Equatorial group subsidiary")
CPFE = ("CPFE3", P, "high", "CPFL Energia (State Grid) subsidiary")
CMIG = ("CMIG4", P, "high", "Cemig subsidiary")
CPLE = ("CPLE3", P, "high", "Copel subsidiary (CPLE6 unified into CPLE3 in 2025)")
AXIA = ("AXIA3", P, "high", "Eletrobras/AXIA subsidiary (ELET3 renamed AXIA3)")
ENBR = ("ENBR3", P, "med", "EDP Energias do Brasil; ENBR3 delisted Aug-2023 (EDP OPA) -> COTAHIST, no parent quote after")
ECOR = ("ECOR3", P, "high", "Ecorodovias concession")
MOTV = ("MOTV3", P, "high", "Motiva (ex-CCR, CCRO3) concession")
ARTR = ("ARTR3", P, "low", "Arteris concession; ARTR3 delisted (Brookfield/Abertis) -> COTAHIST if any")
EGIE = ("EGIE3", P, "med", "Engie Brasil Energia asset")
RENT = ("RENT3", P, "high", "Localiza group")
NONE = lambda note: (None, "none", "", note)  # noqa: E731

MAP: dict[str, list[tuple]] = {
    "16670085": [("RENT3", D, "high", "")],
    "00864214": [("ENGI11", D, "high", "units; most liquid class")],
    "43776517": [("SBSP3", D, "high", "")],
    "04423567": [("ENEV3", D, "high", "")],
    "06047087": [("RDOR3", D, "high", "")],
    "02474103": [("EGIE3", D, "high", "")],
    "07859971": [("TAEE11", D, "high", "units")],
    "02998611": [("ISAE4", D, "high", "ex-CTEEP TRPL4 (brapi continuous history)")],
    "21314559": [("MOVI3", D, "high", "")],
    "03467321": [("ENMT4", D, "low", "own listing, illiquid"), ENGI],
    "50746577": [("CSAN3", D, "high", "")],
    "15139629": [("CEEB3", D, "low", "own listing, illiquid"), NEO],
    "08827501": [NONE("Aegea: unlisted (Equipav/GIC/Itausa minority)")],
    "00001180": [("AXIA3", D, "high", "Eletrobras, ELET3 renamed AXIA3")],
    "10835932": [NEO],
    "17281106": [("CSMG3", D, "high", "")],
    "23373000": [("VAMO3", D, "high", "")],
    "04368898": [CPLE],
    "12091809": [("BRAV3", D, "med", "Brava (Enauta+3R 2024); BRAV3 history pre-merger = 3R RRRP3")],
    "02328280": [("EKTR4", D, "low", "own listing, illiquid"), NEO],
    "01543032": [EQTL],
    "09149503": [("SRNA3", P, "high", "Serena (ex-Omega OMGE3); delisted 2025 -> COTAHIST")],
    "00194724": [("AESB3", P, "med", "ex-AES Brasil Operacoes; AESB3 until 2024 merger into Auren -> COTAHIST"),
                 ("AURE3", P, "med", "Auren (parent after Oct-2024 AES merger; AURE3 listed 2022)")],
    "04370282": [CPLE],
    "04992714": [NONE("NTS: unlisted (Brookfield/Itausa)")],
    "71208516": [NONE("Algar Telecom: CVM-registered, no traded shares")],
    "08873873": [ECOR],
    "08467115": [("EQTL3", P, "high", "CEEE-D privatised to Equatorial Jul-2021")],
    "61486650": [("DASA3", D, "high", "")],
    "06981180": [CMIG],
    "05197443": [("HAPV3", D, "high", "")],
    "02932074": [("HYPE3", D, "high", "")],
    "06057223": [("ASAI3", D, "high", "")],
    "02016440": [CPFE],
    "02846056": [("MOTV3", D, "high", "ex-CCR CCRO3")],
    "08926302": [("PRIO3", P, "high", "PRIO subsidiary")],
    "01417222": [("MRSA3B", D, "low", "balcao listing, almost no trading; owners Vale/CSN/Rumo")],
    "02635522": [("JALL3", D, "high", "")],
    "02919555": [("ARTR3", D, "low", "Arteris; delisted -> COTAHIST if any")],
    "61856571": [("CGAS5", D, "low", "own listing, illiquid"),
                 ("CSAN3", P, "med", "Cosan via Compass (PASS3 only listed 2026)")],
    "15413826": [ENGI],
    "32681371": [NONE("Vix Logistica: unlisted (Aguia Branca)")],
    "33050196": [CPFE],
    "08324196": [NEO],
    "02502844": [("RAIL3", P, "high", "Rumo subsidiary")],
    "76484013": [("SAPR11", D, "high", "units")],
    "31748174": [NONE("Vero: unlisted")],
    "02387241": [("RAIL3", D, "high", "")],
    "04172213": [CPFE],
    "92802784": [NONE("Corsan: Aegea since 2023, unlisted")],
    "34274233": [("VBBR3", D, "high", "")],
    "16404287": [("SUZB3", D, "high", "")],
    "01616929": [NONE("Saneago: state-owned, unlisted")],
    "02286479": [RENT],
    "12648327": [("HBSA3", D, "high", "IPO Sep-2020")],
    "60894730": [("USIM5", D, "high", "")],
    "33958695": [("UNIP6", D, "high", "")],
    "00389481": [NONE("LM Transportes: unlisted")],
    "07415333": [("SIMH3", D, "high", "")],
    "60840055": [("FLRY3", D, "high", "")],
    "42276907": [NONE("VLI: unlisted (Vale/Brookfield/Mitsui)")],
    "33042730": [("CSNA3", D, "high", "")],
    "09095183": [ENGI],
    "04739720": [("EGIE3", P, "high", "Engie Brasil plant (Pampa Sul)")],
    "07040108": [NONE("Cagece: state-owned")],
    "33000167": [("PETR4", D, "high", "")],
    "28201130": [ENGI],
    "61156113": [("MYPK3", D, "high", "")],
    "19699063": [("CPLE3", P, "low", "Copel 50.1% JV with Furnas")],
    "02302100": [ENBR],
    "33592510": [("VALE3", D, "high", "")],
    "07594978": [("SMFT3", D, "high", "IPO Jul-2021")],
    "07047251": [("COCE5", D, "low", "own listing, illiquid; parent Enel unlisted in BR")],
    "51218147": [("IGTI11", P, "high", "Iguatemi S.A. holding (IGTA3 pre-2021)")],
    "75609123": [("LCAM3", D, "med", "Unidas/Locamerica; merged into Localiza Jul-2022 -> COTAHIST"),
                 ("RENT3", P, "med", "Localiza after Jul-2022 merger")],
    "10647979": [NONE("Rota das Bandeiras: unlisted (Odebrecht Transport)")],
    "89086144": [("RAPT4", D, "high", "")],
    "92715812": [CPFE],
    "61695227": [NONE("Enel SP (ex-Eletropaulo): Enel, not listed in BR")],
    "24743678": [("EQTL3", P, "med", "Echoenergia acquired by Equatorial 2022")],
    "00242184": [("ARML3", D, "high", "IPO Jun-2021")],
    "09346601": [("B3SA3", D, "high", "")],
    "08336783": [("CLSC4", P, "high", "Celesc subsidiary")],
    "61532644": [("ITSA4", D, "high", "")],
    "24396489": [NONE("BRK Ambiental: unlisted (Brookfield)")],
    "02451848": [MOTV],
    "03279285": [("ORVR3", P, "high", "Orizon Valorizacao subsidiary")],
    "02800026": [("COGN3", D, "high", "")],
    "33541368": [AXIA],
    "08402943": [("RIAA3", D, "high", "Guararapes GUAR3 renamed Riachuelo RIAA3")],
    "03220438": [("EQTL3", D, "high", "")],
    "04601397": [("BRST3", D, "high", "IPO Jul-2021")],
    "02016507": [AXIA],
    "12009135": [NONE("Alianca Geracao: Vale/Cemig JV, no clear single parent")],
    "05965546": [("EQTL3", P, "high", "CEA acquired by Equatorial 2021")],
    "42310775": [NONE("Aguas do Rio 1: Aegea, unlisted")],
    "04895728": [("EQPA3", D, "low", "own listing, illiquid"), EQTL],
    "82508433": [NONE("Casan: state-owned")],
    "42644220": [NONE("Aguas do Rio 4: Aegea, unlisted")],
    "64904295": [("CAML3", D, "high", "")],
    "09114805": [("OPCT3", D, "high", "IPO Feb-2021")],
    "45242914": [("CEAB3", D, "high", "")],
    "07816890": [("MULT3", D, "high", "")],
    "07282377": [ENGI],
    "00357038": [AXIA],
    "60444437": [("LIGT3", P, "high", "Light S.A. subsidiary")],
    "40432544": [NONE("Claro: America Movil, not listed in BR")],
    "06248349": [NONE("TAG: Engie SA/CDPQ, not EGIE3")],
    "25369840": [("VBBR3", P, "med", "Comerc controlled by Vibra (50% 2023, control 2025)")],
    "08170849": [("DESK3", D, "high", "")],
    "27157474": [NONE("Aguas de Teresina: Aegea")],
    "27093558": [("MILS3", D, "high", "")],
    "08294224": [("JHSF3", D, "high", "")],
    "61190096": [NONE("Eurofarma: unlisted")],
    "25086034": [ENGI],
    "06981176": [CMIG],
    "06272793": [("EQMA3B", D, "low", "own listing, illiquid"), EQTL],
    "42150391": [("BRKM5", D, "high", "")],
    "21389501": [("PASS3", D, "med", "listed only May-2026"), ("CSAN3", P, "high", "Cosan subsidiary")],
    "08343492": [("MRVE3", D, "high", "")],
    "06347409": [("SBFG3", D, "high", "Grupo SBF")],
    "09313969": [ARTR],
    "03342704": [("RECV3", D, "high", "")],
    "08902291": [("CMIN3", D, "high", "IPO Feb-2021")],
    "02762121": [("STBP3", D, "high", "Santos Brasil; delisted 2025 (CMA CGM) -> COTAHIST")],
    "03207703": [ARTR],
    "03497792": [MOTV],
    "06626253": [("PGMN3", D, "high", "IPO Sep-2020")],
    "71673990": [("NATU3", D, "med", "Natura; brapi NATU3 continuous with NTCO3 (holding) pre-2024")],
    "04416935": [("ALUP11", P, "med", "Alupar-controlled transmission")],
    "12648266": [("AMBP3", D, "high", "IPO Jul-2020")],
    "15440708": [("MOTV3", P, "med", "ViaRio, CCR/Motiva majority")],
    "02509491": [ECOR],
    "28152650": [ENBR],
    "16676520": [("MATD3", D, "high", "IPO Apr-2021")],
    "51466860": [("SMTO3", D, "high", "")],
    "10841050": [ECOR],
    "08807432": [("YDUQ3", D, "high", "")],
    "05914650": [ENGI],
    "01083200": [("NEOE3", D, "high", "delisted after OPA -> COTAHIST")],
    "13017462": [ENGI],
    "05878397": [("ALOS3", D, "high", "Allos (ex-Aliansce Sonae ALSO3)")],
    "07522669": [NEO],
    "61585865": [("RADL3", D, "high", "")],
    "09288252": [("ANIM3", D, "high", "")],
    "03853896": [("MBRF3", D, "high", "Marfrig MRFG3 renamed MBRF3 after BRF merger 2025")],
    "12420164": [("VVEO3", D, "high", "IPO Jan-2021")],
    "60933603": [("CESP6", D, "med", "CESP; merged into Auren 2022 -> COTAHIST"),
                 ("AURE3", P, "med", "Auren after 2022 reorganisation")],
    "19527639": [ENGI],
    "01027058": [("CIEL3", D, "high", "Cielo; delisted 2024 -> COTAHIST")],
    "11992680": [("QUAL3", D, "high", "")],
    "17314329": [("MEAL3", D, "high", "")],
    "12272084": [EQTL],
    "13574594": [("ZAMP3", D, "high", "Zamp (ex-BK Brasil BKBR3); delisted 2024 -> COTAHIST")],
    "28594234": [("AURE3", D, "high", "listed Mar-2022")],
    "47508411": [("PCAR3", D, "high", "")],
    "84683374": [("TUPY3", D, "high", "")],
    "02558157": [("VIVT3", D, "high", "")],
    "22261473": [("CMIG4", P, "med", "Gasmig, Cemig 99%")],
    "33376989": [("IRBR3", D, "high", "reverse split 2023")],
    "33113309": [("VLID3", D, "high", "")],
    "02255187": [("FIQE3", D, "high", "IPO Jul-2021")],
    "09325109": [ARTR],
    "08364948": [("ALUP11", D, "high", "units")],
    "04626426": [("BPAC11", P, "med", "BTG Pactual group")],
    "04973790": [CPFE],
    "10979076": [("CPLE3", P, "med", "Copel wind complex (Cutia)")],
    "14797436": [("SRNA3", P, "med", "Omega/Serena asset (Delta 3); delisted 2025 -> COTAHIST")],
    "08070508": [("RAIZ4", P, "high", "Raizen S.A. listed Aug-2021")],
    "08811643": [("TRIS3", D, "high", "")],
    "09336431": [ARTR],
    "02639850": [("LOGN3", P, "med", "Log-In terminal (TVV)")],
    "52548435": [("JSLG3", D, "high", "")],
    "09041168": [("LOGG3", D, "high", "")],
    "42278473": [("WIZC3", D, "high", "")],
    "02600854": [("TIMS3", P, "med", "TIM Brasil holding of TIM S.A.")],
    "06977745": [("BRML3", D, "med", "BR Malls; merged into Allos 2023 -> COTAHIST"),
                 ("ALOS3", P, "med", "Allos after 2023 merger")],
    "33611500": [("GGBR4", D, "high", "")],
    "10324624": [MOTV],
    "26845460": [("EQTL3", P, "med", "Equatorial Transmissao SPE")],
    "26845497": [("EQTL3", P, "med", "Equatorial Transmissao SPE")],
    "12104241": [("ONCO3", D, "high", "IPO Aug-2021")],
    "16614075": [("DIRR3", D, "high", "")],
    "23274194": [AXIA],
    "36030967": [("EGIE3", P, "low", "Engie-branded SPE (city lighting)")],
    "53113791": [("TOTS3", D, "high", "")],
    "12528708": [("AERI3", D, "high", "")],
    "67620377": [("BEEF3", D, "high", "")],
    "47960950": [("MGLU3", D, "high", "")],
    "04986320": [("SEER3", D, "high", "")],
    "07779299": [("ENBR3", P, "low", "EDP transmission SPE")],
    "27831352": [("ENBR3", P, "low", "EDP transmission SPE")],
    "46989951": [("ENBR3", P, "low", "EDP transmission SPE")],
    "55078938": [("ENBR3", P, "low", "EDP transmission SPE")],
    "04426411": [("ENBR3", P, "low", "Enerpeixe, EDP 60%")],
    "03983431": [("ENBR3", D, "high", "EDP Brasil; delisted Aug-2023 -> COTAHIST")],
    "71476527": [("TEND3", D, "high", "")],
    "61079117": [("ALPA4", D, "high", "")],
    "09326342": [ARTR],
    "06840748": [EQTL],
    "26617923": [("TAEE11", P, "med", "Taesa transmission SPE")],
    "02421421": [("TIMS3", D, "high", "")],
    "20512706": [("VTRU3", D, "med", "B3 listing since 2024 (was Nasdaq)")],
    "02998301": [("GEPA4", D, "low", "own listing, illiquid (CTG Brasil)")],
    "49314049": [("ECOR3", P, "med", "EcoNoroeste")],
    "35593905": [ECOR],
    "04591168": [("CPFE3", P, "low", "Foz do Chapeco, CPFL 51%")],
    "39881421": [("CSNA3", P, "med", "CEEE-G acquired by CSN 2022")],
    "60537263": [("ALPK3", D, "high", "")],
    "19208022": [ECOR],
    "02511048": [ECOR],
    "30265100": [ECOR],
    "29884545": [ECOR],
    "15484093": [ECOR],
    "58607200": [ECOR],
    "04149454": [("ECOR3", D, "high", "")],
    "07823262": [("ALUP11", P, "med", "Alupar hydro (Foz do Rio Claro)")],
    "05321987": [("ALUP11", P, "low", "Alupar transmission (ENTE)")],
    "20626892": [("ALUP11", P, "low", "Alupar transmission (ELTE)")],
    "81243735": [("POSI3", D, "high", "")],
    "27665207": [("BBSE3", P, "med", "Brasilprev, BB Seguridade")],
    "83475913": [("PTBL3", D, "high", "")],
    "75315333": [("CRFB3", D, "high", "Carrefour Brasil/Atacadao; delisted 2025 -> COTAHIST")],
    "07682638": [MOTV],
    "09387725": [MOTV],
    "42288184": [("MOTV3", P, "med", "ViaMobilidade 8/9, Motiva majority")],
    "44140908": [("MOTV3", P, "med", "Pampulha airport, Motiva")],
    "92754738": [("LREN3", D, "high", "")],
    "13270520": [("KRSA3", D, "high", "Kora Saude; delisted -> COTAHIST")],
    "73178600": [("CYRE3", D, "high", "")],
    "35822503": [("ANIM3", P, "med", "Inspirali, Anima medical unit")],
    "01838723": [("BRFS3", D, "high", "BRF; merged into MBRF 2025 -> COTAHIST"),
                 ("MBRF3", P, "med", "MBRF after 2025 merger")],
    "14578002": [("ENEV3", P, "high", "Eneva Parnaiba complex")],
    "94813102": [("TTEN3", D, "high", "IPO Jul-2021")],
    "41934221": [("SIMH3", P, "med", "CS Brasil, Simpar group")],
    "42385499": [("SRNA3", P, "med", "Serena group; delisted 2025 -> COTAHIST")],
    "33041260": [("BHIA3", D, "high", "ex-Via VIIA3")],
    "28438834": [NEO],
    "10923227": [("BPAC11", P, "high", "BTG Pactual controlling holding")],
    "89637490": [("KLBN11", D, "high", "units")],
    "08312229": [("EZTC3", D, "high", "")],
    "92791243": [("RANI3", D, "high", "")],
    "08797760": [("CURY3", D, "high", "IPO Sep-2020")],
    "28942127": [("EGIE3", P, "med", "Miranda HPP, Engie since 2017 auction")],
    "28925264": [("EGIE3", P, "med", "Jaguara HPP, Engie since 2017 auction")],
    "18593815": [("PRNR3", D, "high", "IPO Aug-2021")],
    "42771949": [("AALR3", D, "med", "")],
    "42278291": [("LOGN3", D, "med", "illiquid; OPA by MSC 2021-22")],
    "00776574": [("AMER3", D, "high", "judicial recovery 2023; reverse split")],
    "62984091": [("CSED3", D, "high", "IPO Feb-2021")],
    "33453598": [("RAIZ4", D, "high", "IPO Aug-2021")],
    "61602199": [("UGPA3", P, "high", "Ultragaz, Ultrapar")],
    "33337122": [("UGPA3", P, "high", "Ipiranga, Ultrapar")],
    "34130063": [("UGPA3", P, "high", "Ultracargo, Ultrapar")],
    "16590234": [("AZZA3", D, "high", "ex-Arezzo ARZZ3")],
    "10285590": [("SOMA3", D, "high", "Grupo Soma; merged into Azzas 2024 -> COTAHIST"),
                 ("AZZA3", P, "med", "Azzas after 2024 merger")],
    "37663076": [("AURE3", P, "high", "Auren holding")],
    "08439659": [CPFE],
    "26659061": [("ESPA3", D, "med", "")],
    "61088894": [("CAMB3", D, "med", "")],
    "03307926": [("RAIL3", P, "med", "Brado, Rumo")],
    "88610126": [("FRAS3", D, "high", "")],
    "92665611": [("PNVL3", D, "high", "")],
    "97837181": [("DXCO3", D, "high", "")],
    "04065033": [ENGI],
    "35067262": [ENGI],
    "36521478": [ENGI],
    "11421994": [("ORVR3", D, "high", "")],
    "01599101": [("SEQL3", D, "high", "")],
    "08336804": [("CLSC4", P, "high", "Celesc subsidiary")],
    "44649812": [("HAPV3", P, "med", "NotreDame Intermedica, Hapvida since 2022")],
    "02149205": [("PSSA3", D, "high", "")],
    "19126003": [CPLE],
    "61409892": [("CBAV3", D, "high", "IPO Jul-2021")],
    "20247322": [("ALLD3", D, "high", "")],
    "33839910": [("VIVA3", D, "high", "")],
    "01425787": [("ITUB4", P, "high", "Redecard, Itau Unibanco")],
    "00535681": [("CSAN3", P, "low", "Compagas, Compass (Cosan) 51% since 2022")],
    "00954394": [("VULC3", P, "med", "Vulcabras subsidiary")],
    "35654688": [("AMOB3", D, "low", "Automob (ex-Vamos Linha Amarela)")],
}

# Top-weight issuers reviewed and found unlisted (no clear listed parent), documented as 'none'.
NONE_REVIEWED = {
    "35764708": "unlisted", "03025305": "unlisted", "10531501": "unlisted", "13642699": "unlisted",
    "60855574": "unlisted", "12647827": "unlisted", "25176391": "Enel, not listed in BR",
    "40263170": "unlisted (Solvi)", "23438929": "Alares: unlisted", "32021201": "unlisted",
    "23096269": "CTG Brasil, unlisted", "50376938": "unlisted", "42353180": "Igua, unlisted",
    "26664057": "unlisted (Patria)", "06220197": "unlisted", "02041460": "V.tal, unlisted",
    "84684455": "unlisted", "01637895": "Votorantim, unlisted", "44330975": "unlisted",
    "01317277": "unlisted", "09584854": "JV, no clear parent", "15286382": "JV, no clear parent",
    "15286437": "JV, no clear parent", "15578569": "GRU Airport, Invepar (unlisted)",
    "20223016": "State Grid, unlisted", "18494537": "JV", "08213823": "Aura Minerals (TSX/BDR)",
    "47080619": "Tereos, unlisted", "19527586": "JV", "01691945": "Igua, unlisted",
    "16838421": "XP Inc. Nasdaq (BDR not in brapi)", "08159965": "Igua, unlisted",
    "15385166": "Aegea group", "02382073": "Aegea group", "03264927": "Aegea group",
    "04089570": "Aegea group", "12300288": "Norte Energia JV", "09391823": "Santo Antonio JV",
    "24396489": "BRK Ambiental (Brookfield)", "09257877": "VLI group", "00924429": "VLI group",
    "23399329": "Afya, Nasdaq", "04065791": "Sinqia (delisted 2023, not fetched)",
}

WEIGHT_COVER = 0.85  # auto direct-match everything, curate at least the issuers up to this weight


def _norm_words(s: str) -> set[str]:
    return {w for w in cvm.core_name(s or "").split() if len(w) > 2}


# ---------------------------------------------------------------------------------------------
def universe() -> pd.DataFrame:
    lab = pd.read_pickle(LAB)
    e = lab[(lab["eligible"] == True) & (pd.to_datetime(lab["week"]) >= "2022-01-01")]  # noqa: E712
    w = e.groupby("cnpj8").size().rename("w")
    allc = pd.Series(0, index=pd.Index(lab["cnpj8"].dropna().unique(), name="cnpj8"), name="w")
    w = w.reindex(allc.index).fillna(0).astype(int).sort_values(ascending=False)
    t = snd.table().copy()
    t["cnpj8"] = t["CNPJ"].str.replace(r"\D", "", regex=True).str[:8]
    name = t.drop_duplicates("cnpj8").set_index("cnpj8")["Empresa"]
    codes = t.assign(c=t["ISIN"].str[2:6]).groupby("cnpj8")["c"].agg(lambda s: sorted(set(s.dropna())))
    u = pd.DataFrame({"w": w})
    u["issuer_name"] = name.reindex(u.index)
    u["codes"] = codes.reindex(u.index)
    u["cum"] = u["w"].cumsum() / u["w"].sum()
    return u


def stock_list(max_age_h: float = 24 * 7) -> list[dict]:
    p = RAW / "_brapi_stock_list.json"
    if p.exists() and time.time() - p.stat().st_mtime < max_age_h * 3600:
        return json.loads(p.read_text())
    out, page = [], 1
    while True:
        r = brapi._get("/quote/list", limit=2000, page=page)
        out += r.get("stocks", [])
        if not r.get("hasNextPage"):
            break
        page += 1
    out = [x for x in out if x.get("type") == "stock"]
    p.write_text(json.dumps(out))
    return out


def build_map(u: pd.DataFrame, stocks: list[dict]) -> pd.DataFrame:
    roots: dict[str, list[dict]] = {}
    for s in stocks:
        if not s["stock"].endswith("F"):
            roots.setdefault(s["stock"][:4], []).append(s)
    rows = []
    for c8, r in u.iterrows():
        nm = r["issuer_name"] if isinstance(r["issuer_name"], str) else ""
        if c8 in MAP:
            for tk, typ, conf, note in MAP[c8]:
                rows.append((c8, nm, tk, typ, conf, note))
            continue
        # automatic direct match: ISIN issuer code == B3 ticker root AND names overlap
        cands = []
        for code in (r["codes"] if isinstance(r["codes"], list) else []):
            for s in roots.get(code, []):
                if _norm_words(nm) & _norm_words(s.get("name", "")):
                    cands.append(s)
        if cands:
            cands.sort(key=lambda s: -(s.get("volume") or 0))
            top = cands[0]
            rows.append((c8, nm, top["stock"], D, "med", "auto: ISIN code + name match"))
            for s in cands[1:2]:  # second share class only if reasonably liquid
                if (s.get("volume") or 0) >= 0.2 * (top.get("volume") or 0) and s["stock"][:4] == top["stock"][:4]:
                    rows.append((c8, nm, s["stock"], D, "med", "auto: second share class"))
            continue
        if c8 in NONE_REVIEWED:
            rows.append((c8, nm, None, "none", "", NONE_REVIEWED[c8]))
        elif r["cum"] <= WEIGHT_COVER + 0.001:
            rows.append((c8, nm, None, "none", "", "reviewed: SPE/private, no clear listed parent"))
    m = pd.DataFrame(rows, columns=["cnpj8", "issuer_name", "ticker", "mapping_type", "confidence", "note"])
    return m


# ---------------------------------------------------------------------------------------------
def fetch_brapi(ticker: str, max_age_h: float) -> dict | None:
    """Raw brapi quote JSON (cached).  Returns None on 404 or on a redirect to another symbol."""
    p = RAW / f"{ticker}.json"
    if p.exists() and time.time() - p.stat().st_mtime < max_age_h * 3600:
        j = json.loads(p.read_text())
    else:
        h = {"Authorization": f"Bearer {settings.brapi_token}"} if settings.brapi_token else {}
        r = get(f"{brapi.BASE}/quote/{ticker}", params={"range": "10y", "interval": "1d"}, headers=h)
        if r.status_code == 404:
            j = {"error": 404}
        else:
            r.raise_for_status()
            j = r.json()
        p.write_text(json.dumps(j))
        time.sleep(0.2)
    res = (j.get("results") or [None])[0]
    if not res or not res.get("historicalDataPrice"):
        return None
    if res.get("symbol") != ticker:  # e.g. BRFS3 -> MBRF3: a different company's history
        return None
    return res


def brapi_frame(ticker: str, res: dict) -> pd.DataFrame:
    h = pd.DataFrame(res["historicalDataPrice"])
    h = h.dropna(subset=["close"])
    d = pd.to_datetime(h["date"], unit="s", utc=True).dt.tz_convert("America/Sao_Paulo")
    out = pd.DataFrame({
        "ticker": ticker, "date": d.dt.tz_localize(None).dt.normalize(),
        "close": h["close"].astype(float),
        "adj_close": (h["adjustedClose"] if "adjustedClose" in h else h["close"]).astype(float),
        "volume": h["volume"].astype(float), "source": "brapi",
    })
    out["adj_close"] = out["adj_close"].fillna(out["close"])
    return out.drop_duplicates("date", keep="last")


def cotahist_year(y: int) -> pd.DataFrame:
    """B3 COTAHIST extract (spot market, all tickers).  Past years cached forever, current daily."""
    p = RAW / f"cotahist_{y}.csv.gz"
    fresh = y < date.today().year or (p.exists() and time.time() - p.stat().st_mtime < 20 * 3600)
    if p.exists() and fresh:
        return pd.read_csv(p, dtype={"ticker": str}, parse_dates=["date"])
    print(f"  downloading COTAHIST {y} ...", flush=True)
    r = get(COTAHIST_URL.format(y=y))
    r.raise_for_status()
    rows = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            for raw in f:
                ln = raw.decode("latin1")
                if not ln.startswith("01") or ln[24:27] != "010":  # 010 = mercado a vista
                    continue
                rows.append((ln[12:24].strip(), ln[2:10], int(ln[108:121]) / 100,
                             int(ln[152:170]), int(ln[170:188]) / 100, int(ln[210:217])))
    df = pd.DataFrame(rows, columns=["ticker", "date", "close", "volume", "fin_volume", "fatcot"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["close"] = df["close"] / df["fatcot"]
    df = df.drop(columns="fatcot")
    df.to_csv(p, index=False, compression="gzip")
    return df


# Same-share ticker renames inside COTAHIST (old code spliced under the final ticker).
ALIASES = {"OMGE3": "SRNA3", "BKBR3": "ZAMP3"}

# brapi splices predecessor histories under the current ticker.  These predecessors were verified
# (daily-return agreement with B3 COTAHIST = 100%); brapi rows older than the first B3 trade of
# the ticker or of a verified predecessor are DROPPED (e.g. RDOR3 pre-IPO, AURE3 pre-2022 = CESP6,
# ALOS3 pre-Oct-2023 which is not Aliansce).
VERIFIED_PRED = {"AXIA3": ["ELET3"], "AZZA3": ["ARZZ3"], "BHIA3": ["VIIA3"], "BRAV3": ["RRRP3"],
                 "BRST3": ["BRIT3"], "DXCO3": ["DTEX3"], "IGTI11": ["IGTA3"], "ISAE4": ["TRPL4"],
                 "MBRF3": ["MRFG3"], "MOTV3": ["CCRO3"], "NATU3": ["NTCO3"], "RIAA3": ["GUAR3"],
                 "SBFG3": ["CNTO3"], "TIMS3": ["TIMP3"], "VBBR3": ["BRDT3"], "WIZC3": ["WIZS3"],
                 "AMER3": ["BTOW3"]}
# Where brapi's pre-rename rows are wrong, fill from the B3 predecessor (scaled to join smoothly).
FILL_PRED = {"ALOS3": "ALSO3"}


def validate_brapi(frames: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], dict]:
    """Truncate brapi series to B3-verified dates; return per-ticker return agreement with B3."""
    codes = {f["ticker"].iat[0] for f in frames}
    codes |= {p for t in codes for p in VERIFIED_PRED.get(t, [])} | set(FILL_PRED.values())
    b3 = cotahist(codes)
    first = b3.groupby("ticker")["date"].min()
    out, stats = [], {}
    for f in frames:
        t = f["ticker"].iat[0]
        start = min([first.get(c, pd.Timestamp.max) for c in [t] + VERIFIED_PRED.get(t, [])])
        g = f[f["date"] >= start].copy()
        if t in FILL_PRED and not g.empty:
            p = b3[(b3["ticker"] == FILL_PRED[t]) & (b3["date"] < g["date"].min())].copy()
            if not p.empty:
                k = g["adj_close"].iat[0] / g["close"].iat[0]
                p["ticker"], p["adj_close"], p["source"] = t, p["close"] * k, f"cotahist:{FILL_PRED[t]}"
                g = pd.concat([p, g], ignore_index=True)
        own = b3[b3["ticker"] == t].set_index("date")["close"]
        s = g.set_index("date")["close"]
        rb, rr = np.log(s).diff(), np.log(own.reindex(s.index)).diff()
        m = rb.notna() & rr.notna()
        stats[t] = float(((rb - rr).abs() < 0.005)[m].mean()) if m.any() else np.nan
        out.append(g)
    return out, stats


def cotahist(tickers: set[str]) -> pd.DataFrame:
    want = set(tickers) | {a for a, t in ALIASES.items() if t in tickers}
    parts = []
    for y in range(START.year, date.today().year + 1):
        c = cotahist_year(y)
        parts.append(c[c["ticker"].isin(want)])
    c = pd.concat(parts, ignore_index=True)
    c["ticker"] = c["ticker"].replace(ALIASES)
    c = c.sort_values("volume").drop_duplicates(["ticker", "date"], keep="last")
    return pd.DataFrame({"ticker": c["ticker"], "date": c["date"], "close": c["close"],
                         "adj_close": c["close"], "volume": c["volume"].astype(float), "source": "cotahist"})


# ---------------------------------------------------------------------------------------------
def coverage(m: pd.DataFrame, eq: pd.DataFrame) -> dict:
    lab = pd.read_pickle(LAB)
    e = lab[(lab["eligible"] == True) & (pd.to_datetime(lab["week"]) >= "2022-01-01")][["cnpj8", "week"]].copy()  # noqa: E712
    e["week"] = pd.to_datetime(e["week"])
    mm = m.dropna(subset=["ticker"])
    span = eq.groupby("ticker")["date"].agg(["min", "max"])
    out = {"rows": len(e)}
    for typ in ("direct", "parent"):
        tk = mm[mm["mapping_type"] == typ][["cnpj8", "ticker"]].merge(span, left_on="ticker", right_index=True)
        j = e.reset_index().merge(tk, on="cnpj8")
        ok = j[(j["week"] >= j["min"]) & (j["week"] <= j["max"] + pd.Timedelta(days=7))]
        out[f"{typ}_mapped"] = e["cnpj8"].isin(mm.loc[mm["mapping_type"] == typ, "cnpj8"])
        out[f"{typ}_priced"] = e.index.isin(ok["index"])
    out["any_mapped"] = (out["direct_mapped"] | out["parent_mapped"]).mean()
    out["any_priced"] = (out["direct_priced"] | out["parent_priced"]).mean()
    out["direct_priced_share"] = out["direct_priced"].mean()
    out["parent_only_priced_share"] = (out["parent_priced"] & ~out["direct_priced"]).mean()
    hi = mm[mm["confidence"] == "high"]
    out["high_conf_mapped"] = e["cnpj8"].isin(hi["cnpj8"]).mean()
    return {k: v for k, v in out.items() if not isinstance(v, (pd.Series, np.ndarray))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-age-h", type=float, default=20, help="brapi JSON cache age before refetch")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    MAP_CSV.parent.mkdir(parents=True, exist_ok=True)

    u = universe()
    stocks = stock_list()
    m = build_map(u, stocks)
    m.to_csv(MAP_CSV, index=False)
    tickers = sorted(m["ticker"].dropna().unique())
    print(f"map: {m['cnpj8'].nunique()} issuers, {len(tickers)} tickers", flush=True)

    frames, missing = [], []
    for i, t in enumerate(tickers):
        res = fetch_brapi(t, a.max_age_h)
        if res is None:
            missing.append(t)
        else:
            frames.append(brapi_frame(t, res))
    print(f"brapi ok: {len(frames)}; COTAHIST fallback: {missing}", flush=True)
    frames, agree = validate_brapi(frames)
    bad = {k: round(v, 3) for k, v in agree.items() if not v >= 0.95}
    print(f"brapi vs B3 daily-return agreement: median {np.nanmedian(list(agree.values())):.3f}; <95%: {bad}")
    if missing:
        frames.append(cotahist(set(missing)))
    eq = pd.concat(frames, ignore_index=True)
    eq = eq[(eq["date"] >= START) & (eq["date"] < pd.Timestamp(date.today()))]  # drop today's partial bar
    eq = eq.sort_values(["ticker", "date"]).reset_index(drop=True)
    eq.to_pickle(OUT_PKL)
    print(f"equity_daily: {len(eq)} rows, {eq['ticker'].nunique()} tickers, "
          f"{eq['date'].min().date()} -> {eq['date'].max().date()}")
    empty = sorted(set(tickers) - set(eq["ticker"]))
    print("no data at all:", empty)

    span = eq.groupby("ticker").agg(first=("date", "min"), last=("date", "max"), n=("date", "size"),
                                    src=("source", "first"),
                                    med_vol=("volume", "median"))
    late = span[span["first"] > START + pd.Timedelta(days=10)]
    early = span[span["last"] < eq["date"].max() - pd.Timedelta(days=10)]
    print("starts late:\n", late[["first", "n", "src"]].to_string())
    print("ends early:\n", early[["last", "n", "src"]].to_string())
    illiq = span[span["med_vol"] < 5000]
    print("illiquid (median vol < 5k shares):", list(illiq.index))
    cov = coverage(m, eq)
    print("coverage:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in cov.items()}))


if __name__ == "__main__":
    main()
