'''Submodule providing interface for Measurement Computing (MCC) DAQ hardware.
'''

from mcculw import ul
from mcculw.device_info import DaqDeviceInfo
from mcculw.enums import InterfaceType

def config_first_detected_device(board_num, dev_id_list=None):
    """Adds the first available device to the UL.  If a types_list is specified,
    the first available device in the types list will be add to the UL.

    Parameters
    ----------
    board_num : int
        The board number to assign to the board when configuring the device.

    dev_id_list : list[int], optional
        A list of product IDs used to filter the results. Default is None.
        See UL documentation for device IDs.
    """
    ul.ignore_instacal()
    devices = ul.get_daq_device_inventory(InterfaceType.ANY)
    if not devices:
        raise Exception('Error: No DAQ devices found')

    print('Found', len(devices), 'DAQ device(s):')
    for device in devices:
        print('  ', device.product_name, ' (', device.unique_id, ') - ',
              'Device ID = ', device.product_id, sep='')

    device = devices[0]
    if dev_id_list:
        device = next((device for device in devices
                       if device.product_id in dev_id_list), None)
        if not device:
            err_str = 'Error: No DAQ device found in device ID list: '
            err_str += ','.join(str(dev_id) for dev_id in dev_id_list)
            raise Exception(err_str)
            
            

    # Add the first DAQ device to the UL with the specified board number
    ul.create_daq_device(0, devices[0])
    ul.create_daq_device(1, devices[1])
    ul.create_daq_device(2, devices[2])
    #ul.create_daq_device(3, devices[3])
    
# DAC initialization
def DAC_Init():
    use_device_detection = True
    dev_id_list = []
    board_num = 0
     
    try:
        if use_device_detection:
            config_first_detected_device(board_num, dev_id_list)
    
        daq_dev_info = DaqDeviceInfo(board_num)
        if not daq_dev_info.supports_analog_output:
            raise Exception('Error: The DAQ device does not support '
                            'analog output')
    
        print('\nActive DAQ device: ', daq_dev_info.product_name, ' (',
              daq_dev_info.unique_id, ')\n', sep='')
    
        ao_info = daq_dev_info.get_ao_info()
        ao_range = ao_info.supported_ranges[0]
        return ao_range
    except Exception as e:
        print('\n', e)

ao_range = DAC_Init()