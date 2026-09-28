# Issuer exposure map (built by `research/data_exposures.py`)

Universe: 967 issuers (cnpj8) in `lab_weekly.pkl`; weight = eligible rows since 2022 (924 issuers with weight > 0). Every issuer is classified; 'other' holds 3.0% of weight.

Classification source (issuers / weight): cvm 150 / 25.3%; manual 330 / 52.9%; name_rule 487 / 21.7%. Confidence by weight: high 61.5%, low 3.0%, medium 35.5%.

| sector | weight | sector | weight | sector | weight |
|---|---|---|---|---|---|
| utilities_distribution | 16.6% | sanitation | 9.2% | utilities_transmission | 8.7% |
| utilities_generation_hydro | 7.2% | toll_roads | 6.8% | healthcare | 5.4% |
| utilities_generation_renewables | 5.0% | car_rental_fleet | 4.5% | telecom | 3.7% |
| other | 3.0% | airports_ports_logistics | 2.7% | banks_financials | 2.7% |
| sugar_ethanol | 2.5% | oil_gas | 2.4% | real_estate | 2.2% |
| holding_diversified | 2.1% | utilities_generation_thermal | 2.1% | railways | 2.1% |
| retail | 1.9% | trucking_equipment_rental | 1.5% | steel_metals | 1.2% |
| education | 1.0% | textiles_apparel | 0.8% | petrochemicals | 0.7% |
| mining | 0.7% | pulp_paper | 0.6% | fuel_distribution | 0.6% |
| food_beverages | 0.6% | technology | 0.6% | cement_construction_materials | 0.4% |
| agribusiness_protein | 0.3% | agribusiness_grains | 0.2% | fertilizers | 0.1% |

Issuers with >=0.5 total |weight| on traded commodities (energy/ags/metals/fuels) cover 11.4% of weight; with >=0.5 on power (pld/reservoir) 9.3%; the rest is macro-only (ipca/selic/ibc_br/usdbrl) or none. 50 issuers carry overrides (`issuer_overrides.csv` REPLACES the sector row).

Caveats:
- Universe is dominated by regulated/concession infra (power, sanitation, roads) whose drivers are IPCA/Selic, not commodities; hydro PLD/reservoir is the main 'commodity' channel for them.
- CVM SETOR_ATIV is coarse (no hydro/transmission/distribution split), so power issuers are manual/name-rule. Unregistered SPEs with generic names ('X ENERGIA S.A.') default to renewables at low confidence.
- Weights are judgmental priors (sign reliable, magnitude rough); hydro PLD sign is ambiguous in dry years (GSF deficit). Pulp, gold, aluminium, cotton, rice lack a variable, so those issuers lean on usdbrl only.
- Sectors with no link carry one row with empty `variable` and weight 0: drop NaN variables when pivoting.
