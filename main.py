import imports as import_manager

from Outdoor_Object_Recognition_Engine.grid_based_probability_detection import GBPD

import time
start_time = time.time()

GBPD = GBPD(import_manager)
GBPD.main()

print("GBPD algo: ", time.time() - start_time)