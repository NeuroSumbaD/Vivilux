'''This submodule contains the major mechanisms of the Leabra framework re-written
    in a more functional computing paradigm, using Jax for efficient SIMD
    acceleration and to force functional programming patterns. The submodule
    should help to bettter understand the data path and show how the components
    could be modularized for hardware description languages.
'''

import jax
jax.config.update("jax_enable_x64", True)