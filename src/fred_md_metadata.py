"""Metadata for the FRED-MD macro panel (McCracken and Ng, 2016).

Bianchi, Buchner, and Tamoni (2021) use the FRED-MD database as their set of
macroeconomic predictors. Their group-ensemble neural network
(``NN1LayerEnsemExog`` in the provided replication code) needs each series to be
assigned to one of the eight McCracken-Ng groups:

    1. Output and income
    2. Labor market
    3. Housing
    4. Consumption, orders, and inventories
    5. Money and credit
    6. Interest and exchange rates
    7. Prices
    8. Stock market

Each entry below gives

- ``tcode``: the FRED-MD transformation code (1 = level, 2 = first difference,
  3 = second difference, 4 = log, 5 = log first difference,
  6 = log second difference, 7 = first difference of percent change).
- ``group``: McCracken-Ng group number (1-8).
- ``fred_id``: the FRED series used to rebuild the series when the official
  FRED-MD CSV cannot be downloaded. ``None`` means the series is not available
  (with sufficient history) on FRED and is skipped in the fallback panel.

When the official FRED-MD CSV is available, its own transformation codes take
precedence; this table is then only used for the group assignment.
"""

import pandas as pd

GROUP_NAMES = {
    1: "Output and income",
    2: "Labor market",
    3: "Housing",
    4: "Consumption, orders, and inventories",
    5: "Money and credit",
    6: "Interest and exchange rates",
    7: "Prices",
    8: "Stock market",
}

# fmt: off
# (fred_md_id, tcode, group, fred_id)
_SERIES = [
    # Group 1: Output and income
    ("RPI", 5, 1, "RPI"),
    ("W875RX1", 5, 1, "W875RX1"),
    ("INDPRO", 5, 1, "INDPRO"),
    ("IPFPNSS", 5, 1, "IPFPNSS"),
    ("IPFINAL", 5, 1, "IPFINAL"),
    ("IPCONGD", 5, 1, "IPCONGD"),
    ("IPDCONGD", 5, 1, "IPDCONGD"),
    ("IPNCONGD", 5, 1, "IPNCONGD"),
    ("IPBUSEQ", 5, 1, "IPBUSEQ"),
    ("IPMAT", 5, 1, "IPMAT"),
    ("IPDMAT", 5, 1, "IPDMAT"),
    ("IPNMAT", 5, 1, "IPNMAT"),
    ("IPMANSICS", 5, 1, "IPMANSICS"),
    ("IPB51222S", 5, 1, "IPB51222S"),
    ("IPFUELS", 5, 1, "IPFUELS"),
    ("NAPMPI", 1, 1, None),
    ("CUMFNS", 2, 1, "CUMFNS"),
    # Group 2: Labor market
    ("HWI", 2, 2, None),
    ("HWIURATIO", 2, 2, None),
    ("CLF16OV", 5, 2, "CLF16OV"),
    ("CE16OV", 5, 2, "CE16OV"),
    ("UNRATE", 2, 2, "UNRATE"),
    ("UEMPMEAN", 2, 2, "UEMPMEAN"),
    ("UEMPLT5", 5, 2, "UEMPLT5"),
    ("UEMP5TO14", 5, 2, "UEMP5TO14"),
    ("UEMP15OV", 5, 2, "UEMP15OV"),
    ("UEMP15T26", 5, 2, "UEMP15T26"),
    ("UEMP27OV", 5, 2, "UEMP27OV"),
    ("CLAIMSx", 5, 2, "ICSA"),
    ("PAYEMS", 5, 2, "PAYEMS"),
    ("USGOOD", 5, 2, "USGOOD"),
    ("CES1021000001", 5, 2, "CES1021000001"),
    ("USCONS", 5, 2, "USCONS"),
    ("MANEMP", 5, 2, "MANEMP"),
    ("DMANEMP", 5, 2, "DMANEMP"),
    ("NDMANEMP", 5, 2, "NDMANEMP"),
    ("SRVPRD", 5, 2, "SRVPRD"),
    ("USTPU", 5, 2, "USTPU"),
    ("USWTRADE", 5, 2, "USWTRADE"),
    ("USTRADE", 5, 2, "USTRADE"),
    ("USFIRE", 5, 2, "USFIRE"),
    ("USGOVT", 5, 2, "USGOVT"),
    ("CES0600000007", 1, 2, "CES0600000007"),
    ("AWOTMAN", 2, 2, "AWOTMAN"),
    ("AWHMAN", 1, 2, "AWHMAN"),
    ("NAPMEI", 1, 2, None),
    ("CES0600000008", 6, 2, "CES0600000008"),
    ("CES2000000008", 6, 2, "CES2000000008"),
    ("CES3000000008", 6, 2, "CES3000000008"),
    # Group 3: Housing
    ("HOUST", 4, 3, "HOUST"),
    ("HOUSTNE", 4, 3, "HOUSTNE"),
    ("HOUSTMW", 4, 3, "HOUSTMW"),
    ("HOUSTS", 4, 3, "HOUSTS"),
    ("HOUSTW", 4, 3, "HOUSTW"),
    ("PERMIT", 4, 3, "PERMIT"),
    ("PERMITNE", 4, 3, "PERMITNE"),
    ("PERMITMW", 4, 3, "PERMITMW"),
    ("PERMITS", 4, 3, "PERMITS"),
    ("PERMITW", 4, 3, "PERMITW"),
    # Group 4: Consumption, orders, and inventories
    ("DPCERA3M086SBEA", 5, 4, "DPCERA3M086SBEA"),
    ("CMRMTSPLx", 5, 4, "CMRMTSPL"),
    ("RETAILx", 5, 4, "RSAFS"),
    ("NAPM", 1, 4, None),
    ("NAPMNOI", 1, 4, None),
    ("NAPMSDI", 1, 4, None),
    ("NAPMII", 1, 4, None),
    ("ACOGNO", 5, 4, "ACOGNO"),
    ("AMDMNOx", 5, 4, "AMDMNO"),
    ("ANDENOx", 5, 4, "ANDENO"),
    ("AMDMUOx", 5, 4, "AMDMUO"),
    ("BUSINVx", 5, 4, "BUSINV"),
    ("ISRATIOx", 2, 4, "ISRATIO"),
    ("UMCSENTx", 2, 4, "UMCSENT"),
    # Group 5: Money and credit
    ("M1SL", 6, 5, "M1SL"),
    ("M2SL", 6, 5, "M2SL"),
    ("M2REAL", 5, 5, "M2REAL"),
    ("AMBSL", 6, 5, None),  # older vintages; BOGMBASE is used in the fallback
    ("BOGMBASE", 6, 5, "BOGMBASE"),
    ("TOTRESNS", 6, 5, "TOTRESNS"),
    ("NONBORRES", 7, 5, "NONBORRES"),
    ("BUSLOANS", 6, 5, "BUSLOANS"),
    ("REALLN", 6, 5, "REALLN"),
    ("NONREVSL", 6, 5, "NONREVSL"),
    ("CONSPI", 2, 5, None),  # NONREVSL / PI, constructed in the fallback
    ("MZMSL", 6, 5, "MZMSL"),
    ("DTCOLNVHFNM", 6, 5, "DTCOLNVHFNM"),
    ("DTCTHFNM", 6, 5, "DTCTHFNM"),
    ("INVEST", 6, 5, "INVEST"),
    # Group 6: Interest and exchange rates
    ("FEDFUNDS", 2, 6, "FEDFUNDS"),
    ("CP3Mx", 2, 6, None),
    ("TB3MS", 2, 6, "TB3MS"),
    ("TB6MS", 2, 6, "TB6MS"),
    ("GS1", 2, 6, "GS1"),
    ("GS5", 2, 6, "GS5"),
    ("GS10", 2, 6, "GS10"),
    ("AAA", 2, 6, "AAA"),
    ("BAA", 2, 6, "BAA"),
    ("COMPAPFFx", 1, 6, None),
    ("TB3SMFFM", 1, 6, "TB3SMFFM"),
    ("TB6SMFFM", 1, 6, "TB6SMFFM"),
    ("T1YFFM", 1, 6, "T1YFFM"),
    ("T5YFFM", 1, 6, "T5YFFM"),
    ("T10YFFM", 1, 6, "T10YFFM"),
    ("AAAFFM", 1, 6, "AAAFFM"),
    ("BAAFFM", 1, 6, "BAAFFM"),
    ("TWEXMMTH", 5, 6, "TWEXMMTH"),
    ("TWEXAFEGSMTHx", 5, 6, None),
    ("EXSZUSx", 5, 6, "EXSZUS"),
    ("EXJPUSx", 5, 6, "EXJPUS"),
    ("EXUSUKx", 5, 6, "EXUSUK"),
    ("EXCAUSx", 5, 6, "EXCAUS"),
    # Group 7: Prices
    ("WPSFD49207", 6, 7, "WPSFD49207"),
    ("WPSFD49502", 6, 7, "WPSFD49502"),
    ("WPSID61", 6, 7, "WPSID61"),
    ("WPSID62", 6, 7, "WPSID62"),
    ("PPIFGS", 6, 7, None),
    ("PPIFCG", 6, 7, None),
    ("PPIITM", 6, 7, None),
    ("PPICRM", 6, 7, None),
    ("OILPRICEx", 6, 7, "WTISPLC"),
    ("PPICMM", 6, 7, "PPICMM"),
    ("NAPMPRI", 1, 7, None),
    ("CPIAUCSL", 6, 7, "CPIAUCSL"),
    ("CPIAPPSL", 6, 7, "CPIAPPSL"),
    ("CPITRNSL", 6, 7, "CPITRNSL"),
    ("CPIMEDSL", 6, 7, "CPIMEDSL"),
    ("CUSR0000SAC", 6, 7, "CUSR0000SAC"),
    ("CUUR0000SAD", 6, 7, None),
    ("CUSR0000SAD", 6, 7, "CUSR0000SAD"),
    ("CUSR0000SAS", 6, 7, "CUSR0000SAS"),
    ("CPIULFSL", 6, 7, "CPIULFSL"),
    ("CUUR0000SA0L2", 6, 7, None),
    ("CUSR0000SA0L2", 6, 7, "CUSR0000SA0L2"),
    ("CUSR0000SA0L5", 6, 7, "CUSR0000SA0L5"),
    ("PCEPI", 6, 7, "PCEPI"),
    ("DDURRG3M086SBEA", 6, 7, "DDURRG3M086SBEA"),
    ("DNDGRG3M086SBEA", 6, 7, "DNDGRG3M086SBEA"),
    ("DSERRG3M086SBEA", 6, 7, "DSERRG3M086SBEA"),
    # Group 8: Stock market
    ("S&P 500", 5, 8, None),
    ("S&P: indust", 5, 8, None),
    ("S&P div yield", 2, 8, None),
    ("S&P PE ratio", 5, 8, None),
    ("VXOCLSx", 1, 8, None),
    ("VIXCLSx", 1, 8, None),
]
# fmt: on

FRED_MD_METADATA = pd.DataFrame(
    _SERIES, columns=["fred_md_id", "tcode", "group", "fred_id"]
).set_index("fred_md_id")
FRED_MD_METADATA["group_name"] = FRED_MD_METADATA["group"].map(GROUP_NAMES)


def get_group(series_id):
    """Return the McCracken-Ng group number for a FRED-MD series id.

    >>> get_group("INDPRO")
    1
    >>> get_group("S&P 500")
    8
    >>> get_group("NOT_A_SERIES") is None
    True
    """
    if series_id in FRED_MD_METADATA.index:
        return int(FRED_MD_METADATA.loc[series_id, "group"])
    return None
