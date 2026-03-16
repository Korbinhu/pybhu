import math
pi = math.pi

class Constant:

    def __init__(self, value: float, description: str) -> None:
        self._value = value
        self._description = description

    @property
    def value(self) -> float:
        """Numeric value of the constant."""
        return self._value

    @property
    def description(self) -> str:
        """Short description of the constant."""
        return self._description


hbar_unit_Js = Constant(value = 1.05457* 1e-34,
                        description = "hbar: Reduced Planck constant; unit: J.s")

hbar_unit_meVs = Constant(value = 6.582 * 1e-16  * 1e3,
                        description = "hbar: Reduced Planck constant; unit: meV.s")

m_e = Constant(value = 9.109 * 1e-31,
               description = "m_e: Electron mass; unit: kg")

u_B = Constant(value = 0.05788,
               description = "u_B: Bohr magneton ; unit: meV/T")





