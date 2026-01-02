from typing import Union
from llama_index.core.tools.tool_spec.base import BaseToolSpec


class UnitConvertionToolSpec(BaseToolSpec):
    """Unit Conversion tool spec."""

    spec_functions = ["convert_to", "convert_to_base_unit", "adjust_unit"]
        
    def convert_to(self, value: Union[float, int], from_unit: str, to_unit: str) -> str:
        """
        Convert a numeric value from one unit to another.

        Use this when the user asks to convert between specific units
        (e.g. meters to feet, kilograms to pounds).

        Args:
            value (float | int): The numeric value to convert
            from_unit (str): The source unit
            to_unit (str): The target unit
            
        """
        import pint
        
        try:
            quantity: pint.Quantity = pint.Quantity(value, from_unit)
            result: str = str(quantity.to(to_unit))
        except (pint.DimensionalityError, pint.UndefinedUnitError) as e:
            return str(e)
        except Exception as e:
            return "An unkown error has occured. Please try your request again or alter it"
        
        return result
    
    def convert_to_base_unit(self, value: Union[float, int], unit: str) -> str:
        """
        Convert a numeric value to its base SI units.


        Args:
            value (float | int): The numeric value
            unit (str): The unit to convert from
        """
        import pint
        
        try:
            quantity: pint.Quantity = pint.Quantity(value, unit)
        except pint.UndefinedUnitError as e:
            return str(e)
        except Exception as e:
            return "An unkown error has occured. Please try your request again or alter it"
        
        return quantity.to_base_units()        
       
    def adjust_unit(self, value: Union[float, int], unit: str) -> str:
        """
        Convert a numeric value to a more human-readable or compact unit.


        Args:
            value (float | int): The numeric value
            unit (str): The unit to adjust
        """        
        import pint
        
        try:
            quantity: pint.Quantity = pint.Quantity(value, unit)
        except (pint.UndefinedUnitError, pint.PintError) as e:
            return str(e)
        except Exception as e:
            return "An unkown error has occured. Please try your request again or alter it"
        
        return quantity.to_compact()
        
        