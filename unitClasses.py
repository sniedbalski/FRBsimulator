import pint

class Units():
    """
    Designed to be a parent class. Initializes the pint unit registry and sets some constants.
    ...
    Attributes
    ----------
    unr : pint.UnitRegistry()
        initializes the unit registry across child classes
    quant : pint.UnitRegistry().Quantity
        used to set unit values throughotu child classes
    K : pint.UnitRegistry().Quantity
        value of the dispersion constant in units of DM / (MHz^2 * second)
    DM_UNIT : pint.UnitRegistry().Quantity
        sets the DM unit to parsec / (cm^3)
    """
    unr = pint.UnitRegistry()
    quant = unr.Quantity
    K = 2.41e-4 * (quant(1, 'parsec') / quant(1, 'cm')**3) * 1/(quant(1, 'MHz')**2 * quant(1, 's'))
    DM_UNIT = quant(1, 'parsec') / quant(1, 'cm')**3