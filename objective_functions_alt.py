import numpy as np

# design_variables = (
#     ('x1', 'Distance from shore',           'm'),
#     ('x2', 'Island surface area relative to Storage area', '%'), New bounds from 5% to 500%? [0.05, 5]
#     ('x3', 'Storage depth',       'm'),
#     ('x4', 'Storage area',     'm²')
# )

b1 = [90, 1500]      # x1
b2 = [0.05, 5.]       # x2 should be bigger then x4
b3 = [0, 100]       # x3
b4 = [0, 3000 / 4 * (np.pi) ** 2]      # x4
bounds = [b1, b2, b3, b4]

def constraint_1(variables):
    """Distance from shore must be smaller than area taken by the island:

    :return: 1-D array (length n) with constraint values; arr>0 means violation for 'ineq' constraints.
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x4 = variables[:, 3]
    min_dis = 1500      # minimum distance from middle of lake to the shore [m]
    
    return -min_dis + x4 * (1 + x2) / (np.pi) ** 2 + x1 # < 0

def constraint_2(variables):
    """Storage area must be smaller than island area:

    :return: 1-D array (length n) with constraint values; arr>0 means violation for 'ineq' constraints.

    Deze vervalt eigenlijk aangezien x2 nu relatief is aan x4
    """
    x1 = variables[:, 0]
    x2 = variables[:, 1]
    x3 = variables[:, 2]
    x4 = variables[:, 3]
    
    return x4 - x2 # < 0     # Placeholder: no constraint, return zeros meaning no violation

cons = [['ineq', constraint_1]]

# Define objective functions

def objective_function_1(x1, x2, x3, x4):
    """
    Cost function.
    
    :return: float of construction cost in euros.
    """
    mon1 = 7.5 # euro/m3 --> source: https://www.debodemafsluiter.nl/kosten-zandblazen-kruipruimte-ophogen/
    mon2 = 60e6 # euro labour cost for project of 3 years: is half the total construction cost
    mon3 = 16000 # euro per meter for clay dike : https://publicwiki.deltares.nl/spaces/AST/pages/289702255/Tables+construction+and+maintenance+cost+Dutch+prices

    cons1 = mon1 * x2 * x4 * 2.5  # Soil, m3 island = m2_island*2.5 --> river is 2.5m deep
    cons2 = (mon2 / 2.35e6) * x4 * (1 + x2)  # Labour cost of project/ area project * area of the island variable
    cons3 = mon3 * (np.sqrt(x4 / np.pi) * 2 * np.pi) * 10  # Soil dike, m3 = sqrt(m2_island/pi)*2*pi --> assumed 10 meter necessary for the dike

    return cons1 + cons2 + cons3


def objective_function_2(x1, x2, x3, x4):
    """
    Storage capacity
    
    :return: float of capacity in m3
    """
    # We assumed it would have a cilindrical shape
    return x3 * x4


def objective_function_3(x1, x2, x3, x4):
    #TODO: Geeft hele strakke bounds
    """
    Width shipping channel
    
    :return: float of shipping width in m.
    """
    radius_island = np.sqrt(x2 / np.pi)
    br = 3000 # m the passable width of the river

    # return br - (radius_island * 2 + 500 + x1)
    return max([br - x1 - 2 * np.sqrt(x4 / np.pi) - 500, x1])
    # return 500

    # if x1 < br * 0.25:
    #     return br - (radius_island * 2 + 500 + x1)  # 500m measured in google maps, is an estimation of the current situation but it actually is dependent on basin area&island area

    # else: 
    #     return br - (radius_island * 2 + 500)


def objective_function_4(x1, x2, x3, x4):
    """
    Nature area
    
    :return: float of area in m2.
    """
    # We assumed 90% of the residual area is nature, 10% is miscellanous arae
    return x2 * x4 * 0.9


def objective_function_5(x1, x2, x3, x4):
    # TODO: Deze veranderd niet
    """
    Hinderance
    
    :return: float of hinderance in amount/year.
    """
    # For the current situation there is no info about the amount of hinderance (times/year), so we have assumed once every 3 years.
    # Based on this guess and the current values for x2 and x1, we made the formula
    return (np.sqrt(x4) / x1) / 9

objectives = [
    (objective_function_1, "Cost",              "$",            "Government"),
    (objective_function_2, "Storage capacity",    "m^3",          "Research Community"),
    (objective_function_3, "Width shipping channel", "m",     "Waterusers"),
    (objective_function_4, "Nature area",            "m^2", "Tourists"),
    (objective_function_5, "Hinderance",   "amount/year",            "Local residents"),
]

# Weights
weights = [0.2, 0.2, 0.2, 0.2, 0.2]

# Prefs zijn losjes gebaseerd op de uitkomsten van de minmax analyse.
prefs = [
    [[0., 50.7E6], [100, 0]],       # Objective 1  :    min =             0.0  $               max =    50,626,292.7  $
    [[0., 7.5E5], [0, 100]],       # Objective 2  :    min =             0.0  m^3             max =       740,220.3  m^3
    [[2495, 2500], [100, 0]],       # Objective 3  :    min =         2,497.5  m               max =         2,499.7  m
    [[0., 3.35E4], [0, 100]],       # Objective 4  :    min =             0.0  m^2             max =        33,309.9  m^2
    [[0, 3], [100, 0]],       # Objective 5  :    min =             0.0  amount/year     max =             0.0  amount/year
]
