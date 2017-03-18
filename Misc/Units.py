import numpy as np

# Fixed units on the cosmology natural units system
EPS0 = 1.0
MU0 = 1.0
C = 1.0
HBAR = 1.0
KB = 1.0
PI = np.pi

# Fundamental units conversions
CL = 1.0    # Length conversion
CT = 1.0    # Time conversion
CM = 1.0    # Mass conversion
CC = 1.0    # Charge conversion
CK = 1.0    # Temperature conversion


def scalingFactorToNatural(unitNames, unitPowers):
    """
    Function that calculates the scaling factor from SI to natural units

    :param unitNames:       List with the fundamental units used
    :param unitPowers:      List with the powers for each of the units in unitNames
    :return:                The scaling factor
    """
    if len(unitNames) != len(unitPowers) :
        raise Exception("Units.scalingFactorToNatural - Arguments size does not match")
    else:

        scalinFactor = 1.0

        for i in range(len(unitNames)):
            if unitNames[i] == "L" :
                scalinFactor *= CL**unitPowers[i]
            elif unitNames[i] == "T":
                scalinFactor *= CT ** unitPowers[i]
            elif unitNames[i] == "M":
                scalinFactor *= CM ** unitPowers[i]
            elif unitNames[i] == "C":
                scalinFactor *= CC ** unitPowers[i]
            elif unitNames[i] == "K":
                scalinFactor *= CK ** unitPowers[i]
            else:
                raise Exception("Units.scalingFactorToNatural - Unexpected unit name")

        return scalinFactor

def scalingFactorToSI(unitNames, unitPowers):
    """
    Function that calculates the scaling factor from natural units to SI

    :param unitNames:       List with the fundamental units used
    :param unitPowers:      List with the powers for each of the units in unitNames
    :return:                The scaling factor
    """
    return 1.0 / scalingFactorToNatural(unitNames, unitPowers)

def convertToSI(scalar, unitNames, unitPowers):
    """
    Function that converts on the natural units system
    to a SI value

    :param scalar:          Value to be converted to SI
    :param unitNames:       List with the fundamental units used
    :param unitPowers:      List with the powers for each of the units in unitNames
    :return:                The converted value
    """
    return scalar * scalingFactorToSI(unitNames, unitPowers)

def convertToNatural(scalar, unitNames, unitPowers):
    """
    Function that converts on the SI units system
    to a natural units value

    :param scalar:          Value to be converted to natural
    :param unitNames:       List with the fundamental units used
    :param unitPowers:      List with the powers for each of the units in unitNames
    :return:                The converted value
    """
    return scalar * scalingFactorToNatural(unitNames, unitPowers)
