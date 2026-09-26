! *************
! * Well data *
! *************

WELDATA BEGIN

WELL INJECTOR myinj1
myinj1 STATUS = ON
! myinj1 PUMP_LOC = [5, 57, 3, NF]
myinj1 PW_MAX = 420*BAR
myinj1 INJ_CMP [H2O]   = 1.0
myinj1 SURF_DNS [WATER] = PVTDATA.STDNS [WATER]
CONSTRAINT myinj1 q_myin1;
q_myin1 SURF_VOL [WATER] = 79.5 * M3/DAY;

WELL INJECTOR myinj2
myinj2 STATUS = ON
! myinj2 PUMP_LOC = [30, 53, 3, NF]
myinj2 PW_MAX = 420*BAR
myinj2 INJ_CMP [H2O]   = 1.0
myinj2 SURF_DNS [WATER] = PVTDATA.STDNS [WATER]
CONSTRAINT myinj2 q_myin2;
q_myin2 SURF_VOL [WATER] = 79.5 * M3/DAY;

WELL INJECTOR myinj3
myinj3 STATUS = ON
! myinj3 PUMP_LOC = [2, 35, 3, NF]
myinj3 PW_MAX = 420*BAR
myinj3 INJ_CMP [H2O]   = 1.0
myinj3 SURF_DNS [WATER] = PVTDATA.STDNS [WATER]
CONSTRAINT myinj3 q_myin3;
q_myin3 SURF_VOL [WATER] = 79.5 * M3/DAY;

WELL INJECTOR myinj4
myinj4 STATUS = ON
! myinj4 PUMP_LOC = [27, 29, 3, NF]
myinj4 PW_MAX = 420*BAR
myinj4 INJ_CMP [H2O]   = 1.0
myinj4 SURF_DNS [WATER] = PVTDATA.STDNS [WATER]
CONSTRAINT myinj4 q_myin4;
q_myin4 SURF_VOL [WATER] = 79.5 * M3/DAY;

WELL INJECTOR myinj5
myinj5 STATUS = ON
! myinj5 PUMP_LOC = [50, 35, 3, NF]
myinj5 PW_MAX = 420*BAR
myinj5 INJ_CMP [H2O]   = 1.0
myinj5 SURF_DNS [WATER] = PVTDATA.STDNS [WATER]
CONSTRAINT myinj5 q_myin5;
q_myin5 SURF_VOL [WATER] = 79.5 * M3/DAY;

WELL INJECTOR myinj6
myinj6 STATUS = ON
! myinj6 PUMP_LOC = [8, 9, 3, NF]
myinj6 PW_MAX = 420*BAR
myinj6 INJ_CMP [H2O]   = 1.0
myinj6 SURF_DNS [WATER] = PVTDATA.STDNS [WATER]
CONSTRAINT myinj6 q_myin6;
q_myin6 SURF_VOL [WATER] = 79.5 * M3/DAY;

WELL INJECTOR myinj7
myinj7 STATUS = ON
! myinj7 PUMP_LOC = [32, 2, 3, NF]
myinj7 PW_MAX = 420*BAR
myinj7 INJ_CMP [H2O]   = 1.0
myinj7 SURF_DNS [WATER] = PVTDATA.STDNS [WATER]
CONSTRAINT myinj7 q_myin7;
q_myin7 SURF_VOL [WATER] = 79.5 * M3/DAY;

WELL INJECTOR myinj8
myinj8 STATUS = ON
! myinj8 PUMP_LOC = [57, 6, 3, NF]
myinj8 PW_MAX = 420*BAR
myinj8 INJ_CMP [H2O]   = 1.0
myinj8 SURF_DNS [WATER] = PVTDATA.STDNS [WATER]
CONSTRAINT myinj8 q_myin8;
q_myin8 SURF_VOL [WATER] = 79.5 * M3/DAY;


WELL PRODUCER myprod1
myprod1 STATUS = ON
! myprod1 PUMP_LOC = [16, 43, 3, NF]
myprod1 PW_MIN = 395*BAR
myprod1 SURF_DNS [WATER] = PVTDATA.STDNS [WATER]
myprod1 SURF_DNS [OIL] = PVTDATA.STDNS [OIL]

WELL PRODUCER myprod2
myprod2 STATUS = ON
! myprod2 PUMP_LOC = [35, 40, 3, NF]
myprod2 PW_MIN = 395*BAR
myprod2 SURF_DNS [WATER] = PVTDATA.STDNS [WATER]
myprod2 SURF_DNS [OIL] = PVTDATA.STDNS [OIL]

WELL PRODUCER myprod3
myprod3 STATUS = ON
! myprod3 PUMP_LOC = [23, 16, 3, NF]
myprod3 PW_MIN = 395*BAR
myprod3 SURF_DNS [WATER] = PVTDATA.STDNS [WATER]
myprod3 SURF_DNS [OIL] = PVTDATA.STDNS [OIL]

WELL PRODUCER myprod4
myprod4 STATUS = ON
! myprod4 POC = [43, 18, 3, NF]
myprod4 PW_MIN = 395*BAR
myprod4 SURF_DNS [WATER] = PVTDATA.STDNS [WATER]
myprod4 SURF_DNS [OIL] = PVTDATA.STDNS [OIL]

! *****************
! * Interval data *
! *****************

X=1
WHILE ( X LE NZ) DO \ 
(\ 
myinj1 INTERVAL [5, 57, X, 1] ;\ 
myinj2 INTERVAL [30, 53, X, 1] ;\ 
myinj3 INTERVAL [2, 35, X, 1] ;\ 
myinj4 INTERVAL [27, 29, X, 1] ;\ 
myinj5 INTERVAL [50, 35, X, 1] ;\ 
myinj6 INTERVAL [8, 9, X, 1] ;\ 
myinj7 INTERVAL [32, 2, X, 1] ;\ 
myinj8 INTERVAL [57, 6, X, 1] ;\ 

X+=1;\ 
)

X=1
WHILE ( X LE NZ) DO \ 
(\ 
myprod1 INTERVAL [16, 43, X, 1] ;\ 
myprod2 INTERVAL [35, 40, X, 1] ;\ 
myprod3 INTERVAL [23, 16, X, 1] ;\ 
myprod4 INTERVAL [43, 18, X, 1] ;\ 

X+=1;\ 
)

WELDATA END
