import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from dezero.core_simple import Variable



x = Variable(np.array(2.0))
print(x)