import numpy as np 
import sys
import math
import time

from fluc_vol import main

start = time.time()

main()

end = time.time() - start
print "Total time is " +str(end)

