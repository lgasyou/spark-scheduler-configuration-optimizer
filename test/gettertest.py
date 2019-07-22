from optimizer.environment.stateobtaining.regulartimedelaygetter import RegularTimeDelayGetter
from optimizer.environment.stateobtaining.statebuilder import StateBuilder

sb = StateBuilder(
    'http://omnisky:8088/',
    'http://omnisky:18080/',
    None
)
getter = RegularTimeDelayGetter(sb)
print(getter.get_interval_average_time_delay())
