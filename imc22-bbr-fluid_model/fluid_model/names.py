RENO  = 'RENO'
CUBIC = 'CUBIC'
VEGAS = 'VEGAS'
BBR   = 'BBR'
BBR2  = 'BBR2'
PCC   = 'PCC'
PCCRENO = 'PCCRENO'
PCCVEGAS = 'PCCVEGAS'
PCCFLEX = 'PCCFLEX'

RED = 'RED'
DROPTAIL = 'Drop-tail'

# -----------------------------------------------------------
# Parameters
const = 'const'
vary = 'vary'
stp  = 'stp'
T = 'T'
start = 'start'
end = 'end'

N   = 'senders'
C   = 'link_capacity'
AQM = 'qdisc'
CC  = 'cc_combination'
B  = 'switch_buffer'
D   = 'D'
D0  = 'd0'
DL  = 'link_latency'
SLR = 'source_latency_range'
UFP = 'ufp'

# -----------------------------------------------------------
# State
q    = 'q'    # Queue length
w    = 'w'    # Cwnd size
v    = 'v'    # Inflight volume
tau  = 'tau'  # RTT
x    = 'x'    # Rate
y    = 'y'    # Total rate
p    = 'p'    # Loss probability
pc   = 'pc'   # Continous loss probability

wmax_last = 'wmax_last'
wmax = 'wmax' # Cwnd size at time of last loss (CUBIC)
s    = 's'    # Time since last loss (CUBIC)    

tstr = 'tstr' # Start time of period
xpcg = 'xpcg' # Pacing rate
xmax = 'xmax' # Maximum rate measured within one cycle (BBR)
xbtl = 'xbtl' # Estimated bottleneck bandwidth (BBR)
xdel = 'xdel' # Delivery rate
mcrs = 'mcrs' # Cwnd-limited mode
tmin = 'tmin' # RTprop
tprt = 'tprt' # ProbeRT timer (Time until next evaluation)
mprt = 'mprt' # ProbeRT mode
wlo  = 'wlo'  # inflight_lo
whi  = 'whi'  # inflight_hi
mdwn = 'mdwn'  # Probe-bandwidth phase
util = 'util'
utl1 = 'u1'
utl2 = 'u2'
gmma = 'gmma'