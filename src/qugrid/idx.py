"""Column indices for MATPOWER-style case arrays.

QuGrid stores networks in the MATPOWER case format (version 2), the de facto
standard exchange format in power system research. This module gives each
column a name so the rest of the code base never contains a bare column
number. See the MATPOWER manual, Appendix B, for the full format definition.
"""

# ---- bus columns -----------------------------------------------------------
BUS_I = 0  # bus number (external, 1-based in files)
BUS_TYPE = 1  # 1 = PQ, 2 = PV, 3 = reference, 4 = isolated
PD = 2  # active load [MW]
QD = 3  # reactive load [MVAr]
GS = 4  # shunt conductance [MW at V = 1 pu]
BS = 5  # shunt susceptance [MVAr at V = 1 pu]
BUS_AREA = 6
VM = 7  # voltage magnitude [pu]
VA = 8  # voltage angle [degree]
BASE_KV = 9
ZONE = 10
VMAX = 11
VMIN = 12

PQ = 1
PV = 2
REF = 3
NONE = 4

# ---- generator columns -----------------------------------------------------
GEN_BUS = 0
PG = 1  # active output [MW]
QG = 2  # reactive output [MVAr]
QMAX = 3
QMIN = 4
VG = 5  # voltage setpoint [pu]
MBASE = 6
GEN_STATUS = 7  # > 0 = in service
PMAX = 8  # [MW]
PMIN = 9  # [MW]

# ---- branch columns --------------------------------------------------------
F_BUS = 0
T_BUS = 1
BR_R = 2  # resistance [pu]
BR_X = 3  # reactance [pu]
BR_B = 4  # total line charging susceptance [pu]
RATE_A = 5  # MVA rating (0 = unlimited)
RATE_B = 6
RATE_C = 7
TAP = 8  # transformer off-nominal turns ratio (0 = line)
SHIFT = 9  # transformer phase shift [degree]
BR_STATUS = 10
ANGMIN = 11
ANGMAX = 12

# ---- generator cost columns ------------------------------------------------
MODEL = 0  # 1 = piecewise linear, 2 = polynomial
STARTUP = 1  # [$]
SHUTDOWN = 2  # [$]
NCOST = 3
COST = 4  # first coefficient; polynomial costs are highest order first
PW_LINEAR = 1
POLYNOMIAL = 2
