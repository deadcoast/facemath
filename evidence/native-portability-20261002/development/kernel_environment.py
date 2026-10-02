"""Record actual numerical-kernel builds for the negative transfer control."""
import contextlib
import io
import json
import platform
import sys
import numpy as np
import scipy
from threadpoolctl import threadpool_info

configuration={}
for name,action in [('numpy_build',np.show_config),('numpy_runtime',np.show_runtime),('scipy_build',scipy.show_config)]:
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):action()
    configuration[name]=buffer.getvalue()
print(json.dumps(dict(platform=platform.platform(),python=sys.version,
    numpy=np.__version__,scipy=scipy.__version__,threadpools=threadpool_info(),
    configuration=configuration),indent=2))
