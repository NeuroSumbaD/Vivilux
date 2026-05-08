'''Functional forms for the operations required by the Layer class for its
    forward pass.
'''

from vivilux.functional.activations import NoisyXX1
from vivilux.functional.processes import StepFFFB

from jax import numpy as jnp

def UpdateConductance(GeRaw: jnp.ndarray,
                      Ge: jnp.ndarray,
                      Act: jnp.ndarray,
                      GiRaw: jnp.ndarray, # TODO: handle inhibitory inputs
                      GiSyn: jnp.ndarray,
                      fbi: float,
                      DtParams_GDt: float,
                      DtParams_Integ: float,
                      StepFFFB: callable = StepFFFB,
                      ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    # Update conductances from raw inputs
    Ge = Ge + (DtParams_Integ *
               DtParams_GDt * 
               (GeRaw - Ge)
               )
    
    # Call FFFB to update GiRaw
    fbi, Gi_FFFB = StepFFFB(Ge, Act, fbi)

    GiSyn = GiSyn + (DtParams_Integ *
                     DtParams_GDt * 
                     (GiRaw - GiSyn)
                     )
    Gi = GiSyn + Gi_FFFB # Add synaptic Gi to FFFB contribution

    return Ge, GiSyn, Gi, fbi

def UpdateActivity(Vm: jnp.ndarray,
                   Act: jnp.ndarray,
                   Ge: jnp.ndarray,
                   Gi: jnp.ndarray,
                   Gbar_E: float,
                   Gbar_I: float,
                   Gbar_L: float,
                   Erev_E: float,
                   Erev_I: float,
                   Erev_L: float,
                   DtParams_VmDt: float,
                   Thr: float,
                   VmActThr: float,
                   actFn: callable = NoisyXX1,
                   ) -> tuple[jnp.ndarray, jnp.ndarray]:
    '''Functional form of the activity update for a layer, to be called each
        time step after all the conductance operations have been updated.
        Note that if non-default parameters are desired for the activation
        function, it must be decorated with partial and passed in as the
        actFn argument.
    '''
    # Update layer potentials
    Inet = (Ge * Gbar_E * (Erev_E - Vm) +
            Gbar_L * (Erev_L - Vm) +
            Gi * Gbar_I * (Erev_I - Vm)
            )
    Vm = Vm + DtParams_VmDt * Inet

    # Calculate conductance threshold
    geThr = (Gi * Gbar_I * (Erev_I - Thr) +
                Gbar_L * (Erev_L - Thr)
            )
    geThr = geThr/(Thr - Erev_E)

    # Firing rate above threshold governed by conductance-based rate coding
    newAct = actFn(Ge*Gbar_E - geThr)
    
    # Activity below threshold is nearly zero
    mask = jnp.logical_and(
        Act < VmActThr,
        Vm <= Thr
        )
    newAct = jnp.where(mask, actFn(Vm - Thr), newAct)

    # Update layer activities
    Act = Act + DtParams_VmDt * (newAct - Act)

    return Vm, Act