import numpy as np
from jax.scipy.special import erfc
from scipy.optimize import least_squares

q = 1.60217663e-19 # C
k = 1.380649e-23 # J/K

def RectMZI_thermPower(size: int, 
                       vendor_info: dict[str, float],
                       mean_ps: float = 0.5, # pi (default 0.5*pi)
                       ) -> float:
    '''Calculate the power consumption for maintaining the thermal controls of
        the thermo-optic phase shifters in a Rectangular MZI mesh using vendor_info.
    '''
    num_ps = size*(size-1)
    return num_ps*mean_ps*vendor_info['thermoptic phaseshifter power per pi']*1e-3 # W

def BER_NRZ(power: float,
            responsivity: float,
            rx_bandwidth: float,
            rx_in_impedance: float,
            dark_current: float = 0.0, # A (default 0.0 A)
            temperature: float = 300.0, # K (default 300 K)
            ) -> float:
    '''Calculate the bit error rate for a NRZ signal given the input power,
        responsivity, receiver bandwidth/impedance, and dark current.
    '''
    return 0.5*erfc((1/np.sqrt(2))*SNR_NRZ(power = power,
                                            responsivity = responsivity,
                                            rx_bandwidth = rx_bandwidth,
                                            rx_in_impedance = rx_in_impedance,
                                            dark_current = dark_current,
                                            temperature = temperature,
                                            ))

def BER_NRZ_from_SNR(SNR: float) -> float:
    '''Calculate the bit error rate for a NRZ signal given the signal to noise ratio.
    '''
    return 0.5*erfc((1/np.sqrt(2))*SNR)

def SNR_NRZ(power: float,
            responsivity: float,
            rx_bandwidth: float,
            rx_in_impedance: float,
            dark_current: float = 0.0, # A (default 0.0 A)
            temperature: float = 300.0, # K (default 300 K)
            ) -> float:
    '''Calculate the signal to noise ratio for a NRZ signal given the input power,
        receiver bandwidth/impedance, and dark current. Assumes a pin photodiode
        NOT avalanche photodiode.

        TODO: Add wavelegth sensitivity to responsivity
    '''
    signal = responsivity*power

    noise_squared = 2*q*power*responsivity*rx_bandwidth # shot noise
    noise_squared += 2*q*dark_current*rx_bandwidth # dark current noise
    noise_squared += 4*k*temperature*rx_bandwidth/np.real(rx_in_impedance) # thermal noise


    noise = np.sqrt(noise_squared)

    return signal/noise

def power_from_BER(BER: float,
                   responsivity: float,
                   rx_bandwidth: float,
                   rx_in_impedance: float,
                   dark_current: float = 0.0, # A (default 0.0 A)
                   temperature: float = 300.0, # K (default 300 K)
                   verbose: bool = False,
                   ) -> float:
    '''Calculate the power required to achieve a given BER for a NRZ signal given
        the responsivity, receiver bandwidth/impedance, and dark current.
    '''
    def error_function(power: float) -> float:
        ber = BER_NRZ(power, responsivity, rx_bandwidth, rx_in_impedance, dark_current, temperature)
        return ber - BER
    result = least_squares(error_function,
                           x0=[1e-6],
                           gtol=1e-15,
                           bounds=(1e-7, 1e-5),
                      )
    if not result.success:
        raise ValueError(f"Failed to converge: {result.message}")
    if verbose:
        print(f"MINIMIZE EXIT CONDITION: {result.status}")
    power = result.x[0]
    return power

def to_dBm(power: float) -> float:
    '''Convert power from Watts to dBm.
    '''
    return 10*np.log10(power*1e3)

def from_dBm(dBm: float) -> float:
    '''Convert power from dBm to Watts.
    '''
    return 1e-3*np.power(10, dBm/10)

def to_dB(power: float) -> float:
    '''Convert power from Watts to dB.
    '''
    return 10*np.log10(power)

def from_dB(dB: float) -> float:
    '''Convert power from dB to Watts.
    '''
    return np.power(10, dB/10)