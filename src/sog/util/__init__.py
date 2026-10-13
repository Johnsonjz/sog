from .grad import grad
from .scatter import scatter_sum

# deepmd_ref is intentionally NOT re-exported here: it is a test/reference
# bridge that would otherwise ride the always-imported path (`import sog` ->
# module.bec -> util).  Import it explicitly as `sog.util.deepmd_ref`.

