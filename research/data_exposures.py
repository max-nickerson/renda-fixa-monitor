"""Production-chain exposure map for debenture issuers.

Builds (re-runnable, deterministic):
  research/data/issuer_sectors.csv    cnpj8 -> sector (+ source, confidence, weight share)
  research/data/sector_exposures.csv  sector -> signed weights on canonical macro/commodity variables
  research/data/issuer_overrides.csv  cnpj8 -> signed weights that REPLACE the sector row for that issuer
  research/data/EXPOSURES_README.md   coverage summary

Classification priority: manual (curated, below) > cvm (unambiguous CVM SETOR_ATIV) > name_rule
(regex on issuer name) > cvm (ambiguous CVM sector default) > other.

Sign convention: + = variable UP helps margins/revenue (revenue driver); - = variable UP hurts (cost
driver). |weight| ~ importance to margins; sum |w| <= ~2 per sector. Sectors without a clear link
carry a single placeholder row (variable empty, weight 0).
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent.parent))  # run from anywhere

import re
import unicodedata
from pathlib import Path

import pandas as pd

from rfmonitor.config import DATA_DIR
from rfmonitor.ml.selection import reference
from rfmonitor.sources import cvm

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

VARIABLES = ["gas_hh", "gas_eu", "brent", "wti", "naphtha", "urea", "dap", "potash", "sugar", "ethanol_br",
             "soy", "corn", "wheat", "cattle_br", "iron_ore", "coal", "steel", "pld", "reservoir", "diesel_br",
             "gasoline_br", "usdbrl", "selic", "ipca", "ibc_br", "used_cars"]

SECTORS = ["utilities_transmission", "utilities_distribution", "utilities_generation_hydro",
           "utilities_generation_renewables", "utilities_generation_thermal", "oil_gas", "petrochemicals",
           "fertilizers", "sugar_ethanol", "agribusiness_grains", "agribusiness_protein", "mining", "steel_metals",
           "pulp_paper", "cement_construction_materials", "real_estate", "toll_roads", "airports_ports_logistics",
           "railways", "car_rental_fleet", "trucking_equipment_rental", "airlines", "retail", "healthcare",
           "education", "telecom", "sanitation", "banks_financials", "holding_diversified", "fuel_distribution",
           "food_beverages", "textiles_apparel", "technology", "other"]

# --------------------------------------------------------------------------------------------------
# 1. Sector exposure table: sector -> ({variable: weight}, rationale)
# --------------------------------------------------------------------------------------------------
SECTOR_EXPOSURES = {
    "utilities_transmission": ({"ipca": 0.3, "selic": -0.2},
        "RAP revenue IPCA/IGP-M indexed, no volume/commodity risk; highly levered"),
    "utilities_distribution": ({"ipca": 0.2, "ibc_br": 0.2, "selic": -0.2, "pld": -0.1},
        "Parcela B tariff inflation-indexed, load follows activity; energy cost pass-through with CVA/overcontracting lag"),
    "utilities_generation_hydro": ({"pld": 0.6, "reservoir": 0.5, "selic": -0.1},
        "Uncontracted energy sold at PLD; low reservoirs -> GSF deficit forces spot purchases"),
    "utilities_generation_renewables": ({"pld": 0.3, "ipca": 0.2, "selic": -0.2},
        "Mostly IPCA-indexed PPAs; merchant/curtailment tail at PLD; project-finance leverage"),
    "utilities_generation_thermal": ({"reservoir": -0.4, "pld": 0.3, "gas_eu": -0.2, "brent": -0.1},
        "Dispatched (variable revenue) when reservoirs low; fuel (LNG/oil-indexed gas) only partly passed through"),
    "oil_gas": ({"brent": 0.8, "usdbrl": 0.3},
        "Revenue = oil (and oil-indexed gas) priced in USD"),
    "petrochemicals": ({"naphtha": -0.6, "brent": 0.4, "usdbrl": 0.3, "ibc_br": 0.2},
        "Resin/chemical prices track crude & USD; naphtha is main feedstock cost; domestic demand"),
    "fertilizers": ({"urea": 0.8, "dap": 0.2, "gas_eu": -0.5, "gas_hh": -0.3, "usdbrl": 0.2},
        "Nitrogen revenue at import-parity urea; natural gas is dominant cost"),
    "sugar_ethanol": ({"sugar": 0.6, "ethanol_br": 0.4, "diesel_br": -0.2, "usdbrl": 0.2, "potash": -0.1},
        "Revenue sugar (USD export) + hydrous/anhydrous ethanol; diesel & fertilizer in CCT cost"),
    "agribusiness_grains": ({"soy": 0.5, "corn": 0.3, "usdbrl": 0.3, "diesel_br": -0.2, "potash": -0.2, "urea": -0.1},
        "Revenue soy/corn in USD; fertilizer and diesel are main input costs"),
    "agribusiness_protein": ({"corn": -0.4, "soy": -0.3, "cattle_br": -0.3, "usdbrl": 0.3},
        "Feed grains (poultry/pork) and live cattle are main costs; export revenue in USD"),
    "mining": ({"iron_ore": 0.8, "usdbrl": 0.4, "diesel_br": -0.2},
        "Iron-ore revenue in USD; diesel for mine fleet/rail"),
    "steel_metals": ({"steel": 0.7, "iron_ore": -0.3, "coal": -0.3, "usdbrl": 0.2, "ibc_br": 0.2},
        "Steel price vs ore + met-coal spread; import parity via FX; domestic demand"),
    "pulp_paper": ({"usdbrl": 0.6, "diesel_br": -0.1},
        "Pulp exported in USD with BRL cost base (no pulp price in variable set); fuel/logistics cost"),
    "cement_construction_materials": ({"ibc_br": 0.4, "coal": -0.2, "pld": -0.2, "diesel_br": -0.1, "selic": -0.2},
        "Volume follows activity/construction; pet-coke (coal proxy) and power are energy-intensive costs"),
    "real_estate": ({"selic": -0.5, "ipca": 0.2, "ibc_br": 0.1},
        "Mortgage affordability & funding cost; INCC/IGP-M-indexed receivables and leases"),
    "toll_roads": ({"ipca": 0.3, "ibc_br": 0.2, "selic": -0.2},
        "Tariffs IPCA-indexed by contract; heavy-vehicle traffic tracks activity; high leverage"),
    "airports_ports_logistics": ({"ibc_br": 0.4, "diesel_br": -0.2, "ipca": 0.1},
        "Volume-driven (cargo/passengers); fuel cost for trucking/cabotage; tariffs partly inflation-indexed"),
    "railways": ({"soy": 0.3, "corn": 0.2, "iron_ore": 0.2, "diesel_br": -0.3, "ipca": 0.1},
        "Freight volumes are grain/ore exports; diesel is largest variable cost"),
    "car_rental_fleet": ({"selic": -0.6, "used_cars": 0.4, "ibc_br": 0.1},
        "Fleet funded with CDI debt; used-car resale prices drive depreciation/capital gains"),
    "trucking_equipment_rental": ({"selic": -0.5, "used_cars": 0.2, "ibc_br": 0.2},
        "Asset-heavy CDI-funded rental; resale value of trucks/machinery (used-car proxy); activity"),
    "airlines": ({"brent": -0.7, "usdbrl": -0.5, "ibc_br": 0.2},
        "Jet fuel ~ crude; USD leases/maintenance; demand follows activity"),
    "retail": ({"ibc_br": 0.3, "selic": -0.3, "usdbrl": -0.1},
        "Consumer demand and credit cost; imported goods"),
    "healthcare": ({"ibc_br": 0.1},
        "Weak link: formal employment drives health-plan lives; no commodity driver"),
    "education": ({"selic": -0.2, "ibc_br": 0.1},
        "Student affordability/financing and employment; no commodity driver"),
    "telecom": ({"ipca": 0.1, "usdbrl": -0.1},
        "Contracts loosely inflation-indexed; USD-linked network capex; no commodity driver"),
    "sanitation": ({"ipca": 0.3, "selic": -0.2, "pld": -0.1},
        "Regulated tariffs inflation-indexed; leverage; electricity is 2nd-largest opex"),
    "banks_financials": ({}, "No production-chain commodity link (Selic effect ambiguous: NII vs credit)"),
    "holding_diversified": ({"ibc_br": 0.2, "selic": -0.2},
        "Generic: diversified portfolio, holding-level leverage; use issuer override where known"),
    "fuel_distribution": ({"diesel_br": 0.2, "gasoline_br": 0.2, "ethanol_br": 0.1, "ibc_br": 0.2},
        "Volume follows activity; inventory gains when pump prices rise"),
    "food_beverages": ({"wheat": -0.2, "sugar": -0.1, "corn": -0.1, "usdbrl": -0.2, "ibc_br": 0.1},
        "Grain/sugar/imported inputs are costs; domestic demand"),
    "textiles_apparel": ({"ibc_br": 0.2, "selic": -0.2, "usdbrl": -0.1},
        "Discretionary demand & store-card credit; imported inputs (cotton not in variable set)"),
    "technology": ({"usdbrl": -0.1}, "Weak link: USD-priced hardware/cloud costs"),
    "other": ({}, "Unclassified / no clear commodity link"),
}

# --------------------------------------------------------------------------------------------------
# 2. Issuer overrides: cnpj8 -> ({variable: weight}, rationale). REPLACE the sector row entirely.
# --------------------------------------------------------------------------------------------------
_PHARMA = ({"usdbrl": -0.2, "ibc_br": 0.1}, "Pharma manufacturer: imported APIs priced in USD")
_REST = ({"ibc_br": 0.3, "cattle_br": -0.2, "selic": -0.1}, "Restaurant chain: beef input cost, discretionary demand")
_OILSVC = ({"brent": 0.4, "usdbrl": 0.2}, "Oilfield/offshore services: activity follows oil price; USD contracts")
_EP = ({"brent": 0.9, "usdbrl": 0.4}, "Pure E&P: USD oil revenue, BRL-denominated debt service")
_WHEAT = ({"wheat": -0.6, "usdbrl": -0.2}, "Wheat miller: imported wheat is main cost")
OVERRIDES = {
    "50746577": ({"sugar": 0.2, "ethanol_br": 0.2, "soy": 0.1, "corn": 0.1, "diesel_br": -0.1, "ipca": 0.1, "selic": -0.3},
                 "Cosan: holding of Raizen (sugar/ethanol/fuel), Rumo (grain rail), Compass (gas), Moove; high holdco leverage"),
    "33453598": ({"sugar": 0.3, "ethanol_br": 0.3, "diesel_br": 0.1, "gasoline_br": 0.1, "ibc_br": 0.1, "usdbrl": 0.1},
                 "Raizen S.A.: integrated sugar-ethanol + fuel distribution"),
    "07415333": ({"selic": -0.5, "used_cars": 0.3, "ibc_br": 0.2, "diesel_br": -0.1},
                 "Simpar: holding of Movida, Vamos, JSL; CDI-funded fleets"),
    "12648327": ({"soy": 0.3, "corn": 0.2, "diesel_br": -0.3, "usdbrl": 0.2},
                 "Hidrovias do Brasil: grain barging (Arco Norte), USD-linked contracts, fuel cost"),
    "01417222": ({"iron_ore": 0.5, "steel": 0.1, "diesel_br": -0.3, "ipca": 0.1},
                 "MRS: rail volumes dominated by iron ore of Vale/CSN/Usiminas"),
    "04739720": ({"coal": -0.3, "reservoir": -0.3, "pld": 0.3},
                 "Pampa Sul: coal-fired thermal, dispatched when reservoirs low"),
    "01838723": ({"corn": -0.5, "soy": -0.3, "usdbrl": 0.2}, "BRF: poultry/pork, feed-grain cost"),
    "03853896": ({"cattle_br": -0.6, "usdbrl": 0.4}, "Marfrig: beef packer, cattle is cost, USD export/US ops"),
    "67620377": ({"cattle_br": -0.6, "usdbrl": 0.5}, "Minerva: beef packer, ~60%+ export revenue"),
    "33958695": ({"pld": -0.4, "usdbrl": 0.3, "naphtha": -0.2, "ibc_br": 0.2},
                 "Unipar: electricity-intensive chlor-alkali/PVC; dollarized prices; ethylene cost"),
    "05303439": ({"naphtha": -0.3, "brent": 0.2, "urea": 0.3, "gas_eu": -0.2, "usdbrl": 0.2},
                 "Unigel: styrenics/acrylics + leased nitrogen plants (gas -> ammonia/urea)"),
    "33042730": ({"steel": 0.5, "iron_ore": 0.4, "coal": -0.3, "usdbrl": 0.2},
                 "CSN: steel + large iron-ore mining (CSN Mineracao) + cement"),
    "33611500": ({"steel": 0.6, "iron_ore": -0.1, "coal": -0.1, "usdbrl": 0.3, "ibc_br": 0.2},
                 "Gerdau: scrap-based long steel, large US operations (USD)"),
    "60894730": ({"steel": 0.7, "coal": -0.3, "usdbrl": 0.2, "ibc_br": 0.2},
                 "Usiminas: flat steel with captive iron-ore mine (ore roughly hedged)"),
    "08213823": ({"usdbrl": 0.5, "diesel_br": -0.2}, "Aura Almas: gold miner (gold not in variable set), USD revenue"),
    "61409892": ({"usdbrl": 0.4, "pld": -0.2}, "CBA: aluminium (LME not in variable set), USD revenue, power-intensive"),
    "14998371": _WHEAT, "61065199": _WHEAT,
    "64904295": ({"sugar": -0.2, "usdbrl": -0.1, "ibc_br": 0.1}, "Camil: rice/beans/pasta + refined sugar (buys crystal sugar)"),
    "07526557": ({"usdbrl": -0.3, "corn": -0.1, "sugar": -0.1, "ibc_br": 0.2}, "Ambev: USD barley/aluminium, corn/sugar inputs"),
    "17314329": _REST, "13574594": _REST, "13783221": _REST,
    "91830836": ({"soy": -0.5, "diesel_br": 0.4}, "Olfar: soy crushing + biodiesel (priced vs diesel)"),
    "46710597": ({"ethanol_br": 0.7, "corn": -0.5}, "FS: corn-ethanol producer"),
    "08322396": ({"ethanol_br": 0.6, "sugar": 0.1, "corn": -0.2, "diesel_br": -0.1}, "Cerradinho: cane + corn ethanol"),
    "81243735": ({"usdbrl": -0.4, "ibc_br": 0.1}, "Positivo: PC/hardware assembler, USD components"),
    "20247322": ({"usdbrl": -0.3, "ibc_br": 0.1}, "Allied: imported electronics distributor"),
    "05917486": ({"usdbrl": -0.3, "ibc_br": 0.1}, "Livetech: electronics maker, USD components"),
    "02932074": _PHARMA, "61190096": _PHARMA, "60665981": _PHARMA, "60659463": _PHARMA,
    "48344725": _PHARMA, "02814497": _PHARMA, "16619378": _PHARMA,
    "84684455": ({"naphtha": -0.3, "ibc_br": 0.4, "selic": -0.1}, "Tigre: PVC pipes, resin cost, construction demand"),
    "42278291": ({"brent": -0.3, "ibc_br": 0.3}, "Log-In: cabotage shipping, bunker fuel cost"),
    "02762121": ({"ibc_br": 0.4, "usdbrl": 0.1}, "Santos Brasil: container terminal, trade volumes"),
    "12091809": _EP, "08926302": _EP, "03342704": _EP, "32021201": _EP,
    "09114805": _OILSVC, "29980141": _OILSVC, "08091102": _OILSVC, "03670763": _OILSVC, "11198242": _OILSVC,
    "94845674": ({"diesel_br": 0.4, "gasoline_br": 0.3, "brent": -0.6}, "Refinaria Riograndense: refining crack spread"),
    "14031191": ({"soy": 0.2, "corn": 0.1, "usdbrl": -0.1}, "Hinove: specialty fertilizers/biologicals, demand from grain farmers"),
    "02016440": None,  # placeholder (no override); keeps dict literal readable
}
OVERRIDES = {k: v for k, v in OVERRIDES.items() if v is not None}

# --------------------------------------------------------------------------------------------------
# 3. Manual sector assignments: cnpj8 -> (sector, confidence, note)
# --------------------------------------------------------------------------------------------------
H, M, L = "high", "medium", "low"
MANUAL = {
    # car rental / fleet
    "16670085": ("car_rental_fleet", H, "Localiza"), "02286479": ("car_rental_fleet", H, "Localiza Fleet"),
    "21314559": ("car_rental_fleet", H, "Movida"), "75609123": ("car_rental_fleet", H, "Unidas (Localiza)"),
    "45736131": ("car_rental_fleet", H, "Unidas locadora"), "00389481": ("car_rental_fleet", M, "LM Frotas fleet outsourcing"),
    "41934221": ("car_rental_fleet", M, "CS Brasil (Simpar) public fleet"), "00873894": ("car_rental_fleet", H, "Lets rent a car"),
    "08795211": ("car_rental_fleet", M, "Maestro"), "19091996": ("car_rental_fleet", M, "Mobitech"),
    "26982634": ("car_rental_fleet", M, "Turbi car sharing"), "41335131": ("car_rental_fleet", M, "Mottu motorcycle rental"),
    "20611180": ("car_rental_fleet", L, "Seteloc vehicle rental (assumed)"),
    # equipment / truck rental
    "23373000": ("trucking_equipment_rental", H, "Vamos truck/machinery rental"),
    "00242184": ("trucking_equipment_rental", H, "Armac"), "27093558": ("trucking_equipment_rental", H, "Mills"),
    "41570356": ("trucking_equipment_rental", H, "Vrental"), "08100057": ("trucking_equipment_rental", M, "Tecnogera generators"),
    "35654688": ("trucking_equipment_rental", M, "Vamos maquinas"), "45754044": ("trucking_equipment_rental", M, "Simak"),
    "62233407": ("trucking_equipment_rental", M, "Ineer equipment rental"), "08259544": ("trucking_equipment_rental", L, "Topico warehouse/equipment rental"),
    "07415333": ("holding_diversified", H, "Simpar (Movida/Vamos/JSL) - override"),
    # logistics / ports / airports
    "52548435": ("airports_ports_logistics", H, "JSL road logistics"), "32681371": ("airports_ports_logistics", M, "VIX logistics/fleet"),
    "12648327": ("airports_ports_logistics", H, "Hidrovias do Brasil - override"),
    "02762121": ("airports_ports_logistics", H, "Santos Brasil container terminal"),
    "15578569": ("airports_ports_logistics", H, "GRU Airport"), "43514079": ("airports_ports_logistics", M, "CLI Sul grain terminal"),
    "15114494": ("airports_ports_logistics", M, "Corredor Logistica grain terminal"),
    "01317277": ("airports_ports_logistics", H, "Itapoa port"), "01115535": ("airports_ports_logistics", H, "TESC port"),
    "02639850": ("airports_ports_logistics", H, "TVV port"), "27316538": ("airports_ports_logistics", H, "Vports (Vitoria port)"),
    "48710127": ("airports_ports_logistics", H, "Novo Norte airports"), "48725405": ("airports_ports_logistics", H, "Aena airports block"),
    "48533969": ("airports_ports_logistics", M, "airports"), "48534024": ("airports_ports_logistics", M, "airports"),
    "42206269": ("airports_ports_logistics", M, "CCR Bloco Central airports"), "42130537": ("airports_ports_logistics", M, "CCR Bloco Sul airports"),
    "15494541": ("airports_ports_logistics", M, "Salus port infra"), "42278291": ("airports_ports_logistics", H, "Log-In cabotage - override"),
    "20854869": ("airports_ports_logistics", L, "Marlin navegacao (shipping, assumed)"),
    "27486182": ("airports_ports_logistics", M, "Aguia Branca bus transport"),
    "01413969": ("airports_ports_logistics", M, "Comfrio cold-chain logistics"), "34130063": ("airports_ports_logistics", H, "Ultracargo liquid terminals"),
    "47548779": ("airports_ports_logistics", L, "GNL Brasil LNG logistics"), "43217280": ("other", M, "Socicam bus terminals"),
    # railways
    "01417222": ("railways", H, "MRS - override"), "02502844": ("railways", H, "Rumo Malha Paulista"),
    "02387241": ("railways", H, "Rumo"), "42276907": ("railways", H, "VLI"), "09257877": ("railways", H, "Ferrovia Norte Sul (Rumo)"),
    "00924429": ("railways", H, "FCA (VLI)"), "03307926": ("railways", M, "Brado (Rumo containers)"),
    # urban transit -> toll_roads (tariff-indexed concession economics)
    "10324624": ("toll_roads", M, "MetroRio - urban transit concession"), "62070362": ("toll_roads", M, "Metro SP - urban transit"),
    "07682638": ("toll_roads", M, "ViaQuatro - urban transit"), "42288184": ("toll_roads", M, "ViaMobilidade 8/9 - urban transit"),
    "46574475": ("toll_roads", M, "Metro BH - urban transit"), "35588161": ("toll_roads", M, "Linha Universidade - urban transit"),
    "02846056": ("toll_roads", H, "Motiva (ex-CCR): roads + airports + transit"), "08873873": ("toll_roads", H, "Ecorodovias"),
    "02919555": ("toll_roads", H, "Arteris"), "18903785": ("toll_roads", M, "Holding do Araguaia (BR-153)"),
    "09367702": ("toll_roads", M, "CCR concessions holding"), "41508382": ("toll_roads", H, "Rodovias do Brasil holding"),
    "65329869": ("toll_roads", L, "Acquavias (transport concession, assumed)"), "43277147": ("toll_roads", L, "Sul Concessoes (assumed roads)"),
    # oil & gas / fuel distribution
    "04992714": ("utilities_transmission", H, "NTS gas pipeline, ship-or-pay IGP-M-indexed"),
    "06248349": ("utilities_transmission", H, "TAG gas pipeline, ship-or-pay"),
    "34840096": ("utilities_transmission", M, "TRSP LNG regas terminal, contracted"),
    "33000167": ("oil_gas", H, "Petrobras"), "12091809": ("oil_gas", H, "Brava E&P"), "08926302": ("oil_gas", H, "PRIO"),
    "03342704": ("oil_gas", H, "PetroReconcavo"), "32021201": ("oil_gas", H, "Origem gas E&P"),
    "09114805": ("oil_gas", H, "OceanPact offshore services"), "29980141": ("oil_gas", M, "Oceanica subsea services"),
    "08091102": ("oil_gas", M, "Ocyan offshore"), "03670763": ("oil_gas", M, "Omni offshore helicopters"),
    "11198242": ("oil_gas", L, "OSX shipbuilding"), "94845674": ("oil_gas", M, "Refinaria Riograndense - override"),
    "34274233": ("fuel_distribution", H, "Vibra"), "33453598": ("fuel_distribution", H, "Raizen S.A. (fuel + sugar) - override"),
    "33337122": ("fuel_distribution", H, "Ipiranga"), "61602199": ("fuel_distribution", H, "Ultragaz LPG"),
    "03237583": ("fuel_distribution", H, "Copa Energia LPG"), "03987364": ("fuel_distribution", H, "Atem"),
    # gas distribution
    "61856571": ("utilities_distribution", H, "Comgas gas distribution"), "21389501": ("utilities_distribution", H, "Compass gas distribution"),
    "22261473": ("utilities_distribution", H, "Gasmig"), "00535681": ("utilities_distribution", H, "Compagas"),
    "34432153": ("utilities_distribution", H, "Bahiagas"), "03024705": ("utilities_distribution", H, "Necta gas"),
    # power: integrated / holdings / generation
    "00864214": ("utilities_distribution", H, "Energisa (distribution-dominant)"),
    "03220438": ("utilities_distribution", H, "Equatorial (distribution-dominant)"),
    "01083200": ("utilities_distribution", H, "Neoenergia (distribution-dominant)"),
    "03983431": ("utilities_distribution", H, "EDP Brasil (distribution-dominant)"),
    "60444437": ("utilities_distribution", H, "Light"), "61695227": ("utilities_distribution", H, "Enel SP"),
    "08467115": ("utilities_distribution", H, "CEEE-D"), "05965546": ("utilities_distribution", H, "CEA Amapa"),
    "08336783": ("utilities_distribution", H, "Celesc D"), "33050196": ("utilities_distribution", H, "CPFL Paulista"),
    "04172213": ("utilities_distribution", H, "CPFL Piratininga"), "15139629": ("utilities_distribution", H, "Coelba"),
    "10835932": ("utilities_distribution", H, "Celpe"), "08324196": ("utilities_distribution", H, "Cosern"),
    "07047251": ("utilities_distribution", H, "Coelce"), "02328280": ("utilities_distribution", H, "Elektro"),
    "53859112": ("utilities_distribution", M, "CPFL Jaguari"), "83855973": ("utilities_distribution", M, "DCELT"),
    "02016440": ("utilities_distribution", H, "RGE Sul"),
    "07859971": ("utilities_transmission", H, "Taesa"), "02998611": ("utilities_transmission", H, "ISA Energia"),
    "08364948": ("utilities_transmission", H, "Alupar"), "28201130": ("utilities_transmission", H, "Energisa Transmissao"),
    "92715812": ("utilities_transmission", H, "CPFL Transmissao"), "34395916": ("utilities_transmission", L, "V2I (Vinci) transmission"),
    "23520790": ("utilities_transmission", M, "Verene transmission"), "31001230": ("utilities_transmission", H, "Celeo Redes"),
    "24624551": ("utilities_transmission", M, "Argo transmission"), "40215231": ("utilities_transmission", M, "MEZ transmission"),
    "31231893": ("utilities_transmission", M, "MEZ transmission"), "33950678": ("utilities_transmission", M, "MEZ transmission"),
    "31231479": ("utilities_transmission", M, "MEZ transmission"), "60408635": ("utilities_transmission", L, "Regera grid (assumed)"),
    "04423567": ("utilities_generation_thermal", H, "Eneva gas-to-wire thermal"),
    "14578002": ("utilities_generation_thermal", H, "Parnaiba II (Eneva)"), "15743303": ("utilities_generation_thermal", M, "Parnaiba (Eneva)"),
    "04739720": ("utilities_generation_thermal", H, "Pampa Sul coal - override"),
    "23449511": ("utilities_generation_thermal", H, "GNA I LNG thermal"), "23514652": ("utilities_generation_thermal", H, "GNA II LNG thermal"),
    "27241084": ("utilities_generation_thermal", H, "Portocem LNG thermal"),
    "02474103": ("utilities_generation_hydro", H, "Engie Brasil (hydro-dominant)"),
    "00001180": ("utilities_generation_hydro", H, "Axia (ex-Eletrobras)"), "02016507": ("utilities_generation_hydro", M, "Axia Sul (ex-Eletrosul)"),
    "33541368": ("utilities_generation_hydro", H, "Chesf"), "00357038": ("utilities_generation_hydro", H, "Eletronorte"),
    "23274194": ("utilities_generation_hydro", H, "Furnas"), "04370282": ("utilities_generation_hydro", H, "Copel GeT"),
    "06981176": ("utilities_generation_hydro", H, "Cemig GT"), "60933603": ("utilities_generation_hydro", H, "CESP"),
    "00194724": ("utilities_generation_hydro", M, "Auren operations (hydro + wind)"),
    "28594234": ("utilities_generation_hydro", M, "Auren (hydro + wind)"), "37663076": ("utilities_generation_hydro", M, "Auren"),
    "23096269": ("utilities_generation_hydro", H, "Rio Parana (CTG)"), "02998301": ("utilities_generation_hydro", H, "Rio Paranapanema (CTG)"),
    "19014221": ("utilities_generation_hydro", H, "CTG Brasil"), "12300288": ("utilities_generation_hydro", H, "Norte Energia (Belo Monte)"),
    "09391823": ("utilities_generation_hydro", H, "Santo Antonio"), "19527586": ("utilities_generation_hydro", H, "Sinop"),
    "18494537": ("utilities_generation_hydro", H, "Sao Manoel"), "12009135": ("utilities_generation_hydro", H, "Alianca Geracao (hydro)"),
    "25176391": ("utilities_generation_hydro", H, "Volta Grande hydro"), "39881421": ("utilities_generation_hydro", H, "CEEE-G"),
    "08336804": ("utilities_generation_hydro", H, "Celesc G"), "16575828": ("utilities_generation_hydro", M, "Belo Monte stake holder"),
    "04591168": ("utilities_generation_hydro", H, "Foz do Chapeco"), "04426411": ("utilities_generation_hydro", H, "Enerpeixe"),
    "03460864": ("utilities_generation_hydro", H, "Lajeado"), "07727966": ("utilities_generation_hydro", H, "Serra do Facao"),
    "23080281": ("utilities_generation_hydro", M, "Tibagi hydro"), "07802794": ("utilities_generation_hydro", H, "Essentia PCHs"),
    "02397080": ("utilities_generation_hydro", H, "Itapebi"), "07823262": ("utilities_generation_hydro", H, "Foz do Rio Claro"),
    "05104205": ("utilities_generation_hydro", L, "Confluencia (small hydro, assumed)"),
    "09149503": ("utilities_generation_renewables", H, "Serena (ex-Omega) wind/solar"),
    "42500384": ("utilities_generation_renewables", H, "Serena"), "42385499": ("utilities_generation_renewables", H, "Serena"),
    "24743678": ("utilities_generation_renewables", H, "Echoenergia wind"), "25369840": ("utilities_generation_renewables", M, "Comerc (DG solar + trading)"),
    "08439659": ("utilities_generation_renewables", H, "CPFL Renovaveis"), "09334083": ("utilities_generation_renewables", H, "EDP Renovaveis"),
    "35714529": ("utilities_generation_renewables", M, "Tucano wind (AES)"), "34623550": ("utilities_generation_renewables", M, "Tucano wind (AES)"),
    "45024644": ("utilities_generation_renewables", M, "Cajuina wind (AES)"), "14797436": ("utilities_generation_renewables", M, "Delta wind (Serena)"),
    "15190480": ("utilities_generation_renewables", M, "Delta wind"), "13787764": ("utilities_generation_renewables", M, "Delta wind"),
    "24274124": ("utilities_generation_renewables", M, "Assurua wind"), "23778492": ("utilities_generation_renewables", M, "Assurua wind"),
    "38286323": ("utilities_generation_renewables", M, "Assurua wind"), "20829557": ("utilities_generation_renewables", M, "Morrinhos renewables"),
    "09359927": ("utilities_generation_renewables", M, "Asa Branca wind"), "13234214": ("utilities_generation_renewables", M, "Itarema wind"),
    "43386975": ("utilities_generation_renewables", M, "Serra do Serido wind"), "42165941": ("utilities_generation_renewables", L, "Potengi (wind, assumed)"),
    "50258089": ("utilities_generation_renewables", L, "Barreiras holding (renewables, assumed)"),
    "34714313": ("utilities_generation_renewables", L, "solar SPE"), "34714322": ("utilities_generation_renewables", L, "solar SPE"),
    "34745410": ("utilities_generation_renewables", L, "solar SPE"), "34714305": ("utilities_generation_renewables", L, "solar SPE"),
    "08773135": ("utilities_generation_renewables", M, "2W Ecobank renewables"),
    "04023261": ("other", L, "energy trader"), "04973790": ("other", M, "CPFL energy trader"),
    "34475373": ("other", M, "XP energy trader"), "49037416": ("other", M, "energy trader"), "17858631": ("other", M, "energy trader"),
    "19046324": ("other", M, "energy trader"), "36030967": ("other", M, "Engie smart-city/public-lighting PPP"),
    "19126003": ("other", M, "Copel services"),
    # sanitation / waste
    "12648266": ("other", M, "Ambipar environmental/emergency services"),
    "09527023": ("other", L, "Ambipar-related holding"),
    # agri / food / fertilizers
    "50746577": ("holding_diversified", H, "Cosan - override"),
    "08070508": ("sugar_ethanol", H, "Raizen Energia (sugar/ethanol arm)"),
    "13642699": ("sugar_ethanol", H, "Agrovale sugar/ethanol"), "47902283": ("sugar_ethanol", M, "Sonora sugar/ethanol mill"),
    "46710597": ("sugar_ethanol", H, "FS corn ethanol - override"), "08322396": ("sugar_ethanol", H, "Cerradinho - override"),
    "94813102": ("agribusiness_grains", H, "3tentos grains/inputs/crushing"),
    "04626426": ("agribusiness_grains", L, "BTG commodities trading"), "91830836": ("agribusiness_grains", M, "Olfar crushing/biodiesel - override"),
    "10209063": ("agribusiness_grains", L, "Rech agricola"), "04854422": ("agribusiness_grains", L, "Agricola Alvorada"),
    "27664414": ("agribusiness_grains", L, "Agro Talent"), "07483401": ("fertilizers", L, "Total Biotecnologia biological inputs"),
    "14031191": ("fertilizers", L, "Hinove agro inputs - override"), "02476026": ("fertilizers", H, "Ultrafertil nitrogen"),
    "03853896": ("agribusiness_protein", H, "Marfrig - override"), "67620377": ("agribusiness_protein", H, "Minerva - override"),
    "01838723": ("agribusiness_protein", H, "BRF - override"),
    "05303439": ("petrochemicals", H, "Unigel - override"), "33958695": ("petrochemicals", H, "Unipar - override"),
    "86445822": ("petrochemicals", L, "Copobras plastic packaging"),
    # mining / metals / industrials
    "61409892": ("steel_metals", M, "CBA aluminium - override"), "08213823": ("mining", H, "Aura gold - override"),
    "18540906": ("mining", L, "UEM mineracao"), "61082988": ("steel_metals", M, "ferro-alloys"),
    "61156113": ("other", M, "Iochpe-Maxion auto/truck wheels (capital goods)"),
    "89086144": ("other", M, "Randoncorp truck trailers/parts (capital goods)"),
    "88610126": ("other", M, "Fras-le auto parts (capital goods)"),
    "84683374": ("steel_metals", L, "Tupy castings (pig iron/scrap pass-through)"),
    "12528708": ("other", M, "Aeris wind blades (capital goods)"), "78958717": ("other", L, "Romagnole electrical products"),
    "92038108": ("other", L, "Brinox housewares"), "24682682": ("other", L, "GJA industries"),
    "59981829": ("retail", M, "Rodobens dealership/consortium"),
    # construction materials
    "01637895": ("cement_construction_materials", H, "Votorantim Cimentos"), "84684455": ("cement_construction_materials", H, "Tigre - override"),
    "97837181": ("cement_construction_materials", H, "Dexco"), "83475913": ("cement_construction_materials", H, "Portobello"),
    "18593815": ("other", M, "Priner industrial services"), "17185786": ("other", L, "Barbosa Mello construction/engineering"),
    "02949016": ("other", L, "construction services"),
    # real estate / malls
    "51218147": ("real_estate", H, "Iguatemi malls"), "07816890": ("real_estate", H, "Multiplan malls"),
    "05878397": ("real_estate", H, "Allos malls"), "06977745": ("real_estate", H, "BR Malls"),
    "06977751": ("real_estate", H, "BR Properties"), "08294224": ("real_estate", H, "JHSF"),
    "02578564": ("real_estate", M, "MRL (MRV group)"), "24013278": ("real_estate", L, "Barao locacao e empreendimentos"),
    "04409762": ("real_estate", L, "Dix empreendimentos"), "16990436": ("real_estate", L, "Campo Largo patrimonial"),
    # retail / consumer
    "06057223": ("retail", H, "Assai"), "47508411": ("retail", H, "GPA"), "75315333": ("retail", H, "Atacadao"),
    "47960950": ("retail", H, "Magalu"), "33041260": ("retail", H, "Casas Bahia"), "00776574": ("retail", H, "Americanas"),
    "06347409": ("retail", H, "SBF/Centauro"), "59546515": ("retail", H, "Fisia (Nike distributor)"),
    "79379491": ("retail", H, "Havan"), "06626253": ("retail", H, "Pague Menos pharmacy"),
    "61585865": ("retail", H, "RD pharmacy"), "79430682": ("retail", H, "Nissei pharmacy"), "92665611": ("retail", H, "Panvel pharmacy"),
    "06147451": ("retail", M, "Calamo beauty distribution"), "12979552": ("retail", L, "Skinstore"),
    "18328118": ("retail", H, "Petz"), "33839910": ("retail", H, "Vivara"), "04565289": ("retail", M, "Bemol"),
    "03995515": ("retail", H, "Mateus"), "01157555": ("retail", M, "Tenda atacado"), "17392519": ("retail", M, "Evino"),
    "15426874": ("retail", M, "tyre retail"), "22761584": ("retail", M, "Fortbras auto parts retail"),
    "10158356": ("retail", L, "CPX distribuidora"), "61067161": ("other", M, "Nadir Figueiredo glassware"),
    "92754738": ("textiles_apparel", H, "Renner"), "33200056": ("textiles_apparel", H, "Riachuelo"),
    "00954394": ("textiles_apparel", H, "Vulcabras footwear"), "82641325": ("healthcare", M, "Cremer medical supplies"),
    "71673990": ("other", M, "Natura cosmetics (consumer goods)"), "08505736": ("other", M, "Flora hygiene/cleaning products"),
    "07594978": ("other", M, "Smart Fit gyms"), "24921465": ("other", M, "Bluefit gyms"), "22902694": ("other", M, "Selfit gyms"),
    "07737623": ("other", M, "Bodytech gyms"), "10760260": ("other", M, "CVC travel"), "11805397": ("other", M, "Beach Park"),
    "52177416": ("other", M, "football club SAF"), "60537263": ("other", M, "Estapar parking"),
    "27865757": ("other", M, "Globo media"), "09309318": ("other", L, "Emive security services"),
    "00973749": ("other", L, "Top Service facility services"),
    # healthcare
    "12420164": ("healthcare", H, "CM Hospitalar distribution"), "11992680": ("healthcare", H, "Qualicorp health benefits"),
    "71476527": ("real_estate", H, "Tenda homebuilder"),
    # financials / tech
    "16838421": ("banks_financials", H, "XP"), "04233319": ("banks_financials", M, "Bradesco group holding"),
    "01425787": ("banks_financials", H, "Rede (Itau acquiring)"), "10923227": ("banks_financials", H, "BTG holding"),
    "51427102": ("technology", M, "TecBan ATM network"), "28042871": ("banks_financials", M, "Quod credit bureau"),
    "12592831": ("holding_diversified", L, "MNLT holding"),
    "81243735": ("technology", H, "Positivo - override"), "20247322": ("technology", M, "Allied - override"),
    "05917486": ("technology", M, "Livetech - override"),
    "13743550": ("technology", M, "Ascenty data centers"), "34562112": ("technology", M, "Scala data centers"),
    "05510654": ("technology", M, "Algar TI"),
    "35764708": ("telecom", H, "Brasil TecPar ISP"),
    "43945407": ("utilities_generation_renewables", M, "Helexia solar"),
    "52609844": ("utilities_generation_renewables", M, "Helexia solar"),
    "07522669": ("utilities_distribution", H, "Neoenergia Brasilia"),
    "02814497": ("healthcare", H, "Cimed pharma - override"), "16619378": ("healthcare", H, "Cimed pharma - override"),
    "23399329": ("education", H, "Afya medical education"),
    "10997565": ("utilities_transmission", M, "Transenergia SP (transmission)"),
    "14683671": ("utilities_transmission", M, "Transnorte (transmission)"),
    "10553895": ("utilities_transmission", L, "Transenergia (transmission, assumed)"),
    "40480481": ("utilities_generation_renewables", M, "Rio Alto solar"),
    "30342595": ("utilities_generation_renewables", M, "Serra do Mel wind"),
    "20512213": ("utilities_generation_renewables", M, "Chapada do Piaui wind"),
    "19943730": ("utilities_generation_renewables", M, "Voltalia wind"),
    "00905036": ("banks_financials", M, "Icatu"),
    "06013760": ("airports_ports_logistics", H, "Convicon container terminal"),
}

# CVM SETOR_ATIV (stripped of the "Emp. Adm. Part. - " prefix) -> sector. Unambiguous ones only.
CVM_UNAMBIGUOUS = {
    "Saneamento, Serv. Água e Gás": "sanitation", "Telecomunicações": "telecom", "Serviços Médicos": "healthcare",
    "Serviços médicos": "healthcare", "Farmacêutico e Higiene": "healthcare", "Educação": "education",
    "Bancos": "banks_financials", "Seguradoras e Corretoras": "banks_financials",
    "Intermediação Financeira": "banks_financials", "Securitização de Recebíveis": "banks_financials",
    "Arrendamento Mercantil": "banks_financials", "Bolsas de Valores/Mercadorias e Futuros": "banks_financials",
    "Crédito Imobiliário": "banks_financials", "Factoring": "banks_financials",
    "Papel e Celulose": "pulp_paper", "Embalagens": "pulp_paper", "Reflorestamento": "pulp_paper",
    "Têxtil e Vestuário": "textiles_apparel", "Petroquímicos e Borracha": "petrochemicals",
    "Extração Mineral": "mining", "Metalurgia e Siderurgia": "steel_metals",
    "Agricultura (Açúcar, Álcool e Cana)": "sugar_ethanol", "Petróleo e Gás": "oil_gas",
    "Alimentos": "food_beverages", "Bebidas e Fumo": "food_beverages", "Pesca": "food_beverages",
    "Comunicação e Informática": "technology", "Brinquedos e Lazer": "other", "Hospedagem e Turismo": "other",
    "Construção Civil, Mat. Constr. e Decoração": "real_estate", "Const. Civil, Mat. Const. e Decoração": "real_estate",
    "Gráficas e Editoras": "other",
}
# ambiguous CVM sectors: used only if no name rule matched
CVM_AMBIGUOUS = {
    "Energia Elétrica": "utilities_generation_renewables", "Serviços Transporte e Logística": "airports_ports_logistics",
    "Comércio (Atacado e Varejo)": "retail", "Máquinas, Equipamentos, Veículos e Peças": "other",
    "Máqs., Equip., Veíc. e Peças": "other", "Sem Setor Principal": "holding_diversified",
    "Outras Atividades Industriais": "other", "Serviços Diversos": "other", "Serviços em Geral": "other",
    "Comércio Exterior": "agribusiness_grains",
}

# ordered name rules (regex on accent-stripped upper-case name) -> (sector, confidence)
NAME_RULES = [
    (r"SECURITIZ", "banks_financials", M),
    (r"ILUMINACAO|ILUMINA |IL\. ?PUB|\bQLUZ\b", "other", M),  # public-lighting PPPs
    (r"AEROPORTO|VOE XAP", "airports_ports_logistics", H),
    (r"TRANSMISSOR|TRANSMISSAO|INTERLIGACAO ELETRICA|SUBESTACAO|TRANS DE E E|\bGRID\b", "utilities_transmission", H),
    (r"DISTRIBUI.*ENERGIA|ENERGIA.*DISTRIBUI|DISTRIBUICAO S", "utilities_distribution", H),
    (r"\bGAS\b|GAS NATURAL", "utilities_distribution", M),
    (r"SANEAMENTO|\bAGUAS?\b|ESGOTO|AMBIENTAL|RESIDUOS|REUSO", "sanitation", H),
    (r"\bUTE\b|TERMELETRICA|TERMOELETRICA", "utilities_generation_thermal", H),
    (r"EOLIC|VENTOS|WIND|SOLAR|FOTOVOLTAIC|\bUFV\b|RENOVAV|GERACAO DISTRIBUIDA|\bG\.?D\.?\b|THOPEN|\bAXS\b|ATHON|ALSOL",
     "utilities_generation_renewables", H),
    (r"ACUCAR|ALCOOL|ETANOL|BIOENERG|SUCROENERG|AGROINDUSTRI|\bUSINA\b", "sugar_ethanol", M),
    (r"\bUHE\b|HIDRELETRICA|HIDRO ELETRICA|\bPCHS?\b|ENERGETICA", "utilities_generation_hydro", M),
    (r"METRO|TRENS|MOBILIDADE|MOBI ", "toll_roads", L),
    (r"FERROVI|MALHA ", "railways", H),
    (r"RODOVI|AUTOPISTA|CONCESSIONARIA|\bVIA\b|\bECO\d|\bEPR\b|ROTA D|ENTREVIAS|VIARONDON|ECOVIAS|VIAPAULISTA|RODOANEL",
     "toll_roads", H),
    (r"PORTUARI|CONTEINERES|TERMINA(L|IS)|\bPORTOS?\b|PORTONAVE|LOGISTIC|NAVEGACAO|ARMAZENS", "airports_ports_logistics", M),
    (r"RENT A CAR|LOCADORA DE VEICULOS|LOCACAO DE VEICULOS", "car_rental_fleet", H),
    (r"LOCACAO|LOCADORA", "trucking_equipment_rental", M),
    (r"HOSPITA|SAUDE|MEDIC|ONCOLOG|FARMAC|LABORATORIO|DIAGNOST|HEALTH|VISAO|CORPOREOS", "healthcare", M),
    (r"TELECOM|FIBRA|TORRES|WINITY|TELEFONIA", "telecom", M),
    (r"EDUCA|ENSINO", "education", M),
    (r"SHOPPING|INCORPORA|IMOBILIARI|CONSTRUTORA|REALTY|PROPERTIES|PATRIMONIAL", "real_estate", M),
    (r"MINERACAO", "mining", M),
    (r"SIDERURG|METALURG|FERRO LIGA|\bACO\b", "steel_metals", M),
    (r"TECNOLOGIA|SISTEMAS|SOFTWARE", "technology", L),
    (r"SEGUR|INVESTIMENTOS|FINANCEIR|CREDITO|CARTOES|PAGAMENTO|\bBANK\b|BANCO|PREVIDENCI", "banks_financials", M),
    (r"ENERGIA|ENERGY|GERACAO|GERADORA", "utilities_generation_renewables", L),  # unknown power SPEs: mostly DG/wind/solar
    (r"AGRO|AGRICOLA", "agribusiness_grains", L),
]
NAME_RULES = [(re.compile(p), s, c) for p, s, c in NAME_RULES]
HOLDING_RULE = re.compile(r"PARTICIPACOES|HOLDING|SUBHOLDING|CAPITAL")  # applied after ambiguous CVM


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s.upper()).strip()


def _strip_prefix(setor):
    if not isinstance(setor, str):
        return None
    if setor.startswith("Emp. Adm. Participa"):
        return "Sem Setor Principal"
    return re.sub(r"^Emp\. ?Adm\. ?Part\.? ?-? ?", "", setor).strip() or "Sem Setor Principal"


def classify(cnpj8, name, setor):
    if cnpj8 in MANUAL:
        s, c, note = MANUAL[cnpj8]
        return s, "manual", c, note
    base = _strip_prefix(setor)
    if base in CVM_UNAMBIGUOUS:
        return CVM_UNAMBIGUOUS[base], "cvm", M, f"CVM: {setor}"
    n = _norm(name)
    for rx, s, c in NAME_RULES:
        if rx.search(n):
            # a CVM energy/transport registration raises confidence of a matching name rule
            return s, "name_rule", c, f"rule /{rx.pattern[:40]}/" + (f"; CVM: {setor}" if base else "")
    if base in CVM_AMBIGUOUS:
        return CVM_AMBIGUOUS[base], "cvm", L, f"CVM (ambiguous): {setor}"
    if HOLDING_RULE.search(n):
        return "holding_diversified", "name_rule", L, "generic holding/participacoes name"
    if base == "Emp. Adm. Part." or (isinstance(setor, str) and setor.startswith("Emp. Adm. Part")):
        return "holding_diversified", "cvm", L, f"CVM: {setor}"
    return "other", "name_rule", L, "no rule matched"


def build():
    lab = pd.read_pickle(DATA_DIR / "history" / "lab_weekly.pkl")
    lab["cnpj8"] = lab["cnpj8"].astype(str).str.zfill(8)
    elig = lab[(lab["eligible"] == True) & (pd.to_datetime(lab["week"]) >= "2022-01-01")]  # noqa: E712
    w = elig.groupby("cnpj8").size()
    uni = pd.DataFrame(index=pd.Index(sorted(lab["cnpj8"].unique()), name="cnpj8"))
    uni["n_eligible"] = w.reindex(uni.index).fillna(0).astype(int)
    uni["weight_share"] = uni["n_eligible"] / uni["n_eligible"].sum()

    ref = reference()
    ref["c8"] = ref["cnpj"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(14).str[:8]
    names = ref.dropna(subset=["issuer"]).groupby("c8")["issuer"].agg(lambda s: s.mode().iat[0])

    reg = cvm.registry().copy()
    reg["c8"] = reg["cnpj_digits"].astype(str).str.zfill(14).str[:8]
    reg["_active"] = (reg["SIT"].astype(str).str.upper() == "ATIVO").astype(int)
    reg = reg.sort_values(["c8", "_active"], ascending=[True, False]).drop_duplicates("c8").set_index("c8")

    uni["issuer_name"] = names.reindex(uni.index)
    uni["issuer_name"] = uni["issuer_name"].fillna(reg["DENOM_SOCIAL"].reindex(uni.index)).fillna("")
    uni["cvm_sector"] = reg["SETOR_ATIV"].reindex(uni.index)

    rows = [classify(c, r.issuer_name, r.cvm_sector) for c, r in uni.iterrows()]
    uni[["sector", "sector_source", "confidence", "note"]] = pd.DataFrame(rows, index=uni.index)
    has_ovr = uni.index.isin(list(OVERRIDES))
    uni.loc[has_ovr, "note"] = uni.loc[has_ovr, "note"].astype(str) + " [issuer override]"
    assert set(uni["sector"]) <= set(SECTORS), set(uni["sector"]) - set(SECTORS)

    out = uni.reset_index().sort_values("weight_share", ascending=False)
    out[["cnpj8", "issuer_name", "sector", "sector_source", "confidence", "weight_share", "note"]].to_csv(
        OUT / "issuer_sectors.csv", index=False, encoding="utf-8")

    # sector exposures
    srows = []
    for sec in SECTORS:
        wts, why = SECTOR_EXPOSURES[sec]
        assert set(wts) <= set(VARIABLES), (sec, set(wts) - set(VARIABLES))
        assert sum(abs(v) for v in wts.values()) <= 2.0 + 1e-9, sec
        if not wts:
            srows.append((sec, "", 0.0, why))
        for v, x in wts.items():
            srows.append((sec, v, x, why))
    sx = pd.DataFrame(srows, columns=["sector", "variable", "weight", "rationale"])
    sx.to_csv(OUT / "sector_exposures.csv", index=False, encoding="utf-8")

    orows = []
    for c8, (wts, why) in OVERRIDES.items():
        assert set(wts) <= set(VARIABLES), (c8, set(wts) - set(VARIABLES))
        for v, x in wts.items():
            orows.append((c8, v, x, why))
    ox = pd.DataFrame(orows, columns=["cnpj8", "variable", "weight", "rationale"])
    ox.to_csv(OUT / "issuer_overrides.csv", index=False, encoding="utf-8")

    write_readme(out, sx, ox)
    return out, sx, ox


def write_readme(out, sx, ox):
    cov = out.groupby("sector")["weight_share"].sum().sort_values(ascending=False)
    real_commod = {"gas_hh", "gas_eu", "brent", "wti", "naphtha", "urea", "dap", "potash", "sugar", "ethanol_br", "soy",
                   "corn", "wheat", "cattle_br", "iron_ore", "coal", "steel", "diesel_br", "gasoline_br"}
    power = {"pld", "reservoir"}

    def eff(c8, sec):  # effective per-issuer exposure: override replaces sector row
        return OVERRIDES[c8][0] if c8 in OVERRIDES else SECTOR_EXPOSURES[sec][0]

    effs = [eff(c, s) for c, s in zip(out["cnpj8"], out["sector"])]
    ws = out["weight_share"].to_numpy()
    commod_w = sum(w for e, w in zip(effs, ws) if sum(abs(x) for v, x in e.items() if v in real_commod) >= 0.5)
    power_w = sum(w for e, w in zip(effs, ws) if sum(abs(x) for v, x in e.items() if v in power) >= 0.5)
    src = out.groupby("sector_source").agg(n=("cnpj8", "size"), w=("weight_share", "sum"))
    conf = out.groupby("confidence")["weight_share"].sum()
    other_w = cov.get("other", 0.0)
    lines = [
        "# Issuer exposure map (built by `research/data_exposures.py`)", "",
        f"Universe: {len(out)} issuers (cnpj8) in `lab_weekly.pkl`; weight = eligible rows since 2022 "
        f"({(out.weight_share > 0).sum()} issuers with weight > 0). Every issuer is classified; "
        f"'other' holds {other_w:.1%} of weight.", "",
        "Classification source (issuers / weight): " + "; ".join(
            f"{k} {int(r.n)} / {r.w:.1%}" for k, r in src.iterrows()) + ". "
        "Confidence by weight: " + ", ".join(f"{k} {v:.1%}" for k, v in conf.items()) + ".", "",
        "| sector | weight | sector | weight | sector | weight |", "|---|---|---|---|---|---|",
        *["| " + " | ".join(f"{s} | {v:.1%}" for s, v in list(cov.items())[i:i + 3]) + " |"
          for i in range(0, len(cov), 3)], "",
        f"Issuers with >=0.5 total |weight| on traded commodities (energy/ags/metals/fuels) cover {commod_w:.1%} of "
        f"weight; with >=0.5 on power (pld/reservoir) {power_w:.1%}; the rest is macro-only (ipca/selic/ibc_br/usdbrl) "
        f"or none. "
        f"{ox['cnpj8'].nunique()} issuers carry overrides (`issuer_overrides.csv` REPLACES the sector row).", "",
        "Caveats:",
        "- Universe is dominated by regulated/concession infra (power, sanitation, roads) whose drivers are IPCA/Selic, "
        "not commodities; hydro PLD/reservoir is the main 'commodity' channel for them.",
        "- CVM SETOR_ATIV is coarse (no hydro/transmission/distribution split), so power issuers are manual/name-rule. "
        "Unregistered SPEs with generic names ('X ENERGIA S.A.') default to renewables at low confidence.",
        "- Weights are judgmental priors (sign reliable, magnitude rough); hydro PLD sign is ambiguous in dry years "
        "(GSF deficit). Pulp, gold, aluminium, cotton, rice lack a variable, so those issuers lean on usdbrl only.",
        "- Sectors with no link carry one row with empty `variable` and weight 0: drop NaN variables when pivoting.",
    ]
    (OUT / "EXPOSURES_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    out, sx, ox = build()
    cov = out.groupby("sector")["weight_share"].sum().sort_values(ascending=False)
    print(cov.to_string(float_format=lambda x: f"{x:.3f}"))
    print(out.groupby("sector_source").agg(n=("cnpj8", "size"), w=("weight_share", "sum")))
    print(f"wrote {OUT}")
