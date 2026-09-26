! ************
! * PVT data *
! ************

PVTDATA BEGIN
   ! The definition template for this pvt model:
   PVTMODEL pvtmodel_1
   pvtmodel_1 TRACED[ H2O ]   = OFF;
   pvtmodel_1 TRACED[ HEAVY ] = OFF;

   ! The supplied stock tank density variables:
   pvtmodel_1 STDNS WATER = 1000.0000 * KG/M3;
   pvtmodel_1 STDNS OIL   =  900.0000 * KG/M3;

   !The supplied reference pressure variable:
   pvtmodel_1 PREF = 400 * BAR;

   ! The supplied pure density variables:
   pvtmodel_1 DNS WATER = 1000.0000 * KG/M3;
   pvtmodel_1 DNS OIL   =  900.0000 * KG/M3;

   ! The supplied fixed compressibility variables:
   pvtmodel_1 COMPR WATER = 1.0000e-05 * 1/BAR;
   pvtmodel_1 COMPR OIL   = 1.0000e-05 * 1/BAR;

   ! The supplied pure phase viscosity variables:
   pvtmodel_1 VSC WATER = 1.0000e-03 * PA*S;
   pvtmodel_1 VSC OIL   = 5.0000e-03 * PA*S;

   STDNS [WATER] = pvtmodel_1 STDNS WATER;
   STDNS [OIL]   = pvtmodel_1 STDNS OIL;

PVTDATA END

! ******************************
! * Relative permeability data *
! ******************************

RLPDATA BEGIN

   RLPINTERPOL = LINEAR;

   ! Corey-type relative permeabilities for an unfractured reservoir model
   ! The user can modify or replace them by his own tables if required.

   ! M A T R I X  O N L Y --> CHECK !!

   VARIABLE VOLFRAC SCW SORG SORW SRG SRL
   VARIABLE REAL    KRW KROW KRG KROG NW NO NG
   VARIABLE INTEGER NPOINTS

   SCW = 0.10; SORW = 0.10;
   KRW = 1.0 ; KROW = 1.0 ;
   NW  = 1   ; NO   = 1   ;

   ! Import RLP data from Imagine2 field
   INCLUDE Rel_Perm.inc

   ! Create a relative permeability model for the matrix
   RLPMODEL relpmat
   relpmat TWOPHASETABLE WATER OIL : Rel_Perm_TaUn_rlp_mat_watoil

   ! Assign the model to matrix 
   relpmat LOC XYZFRPD

RLPDATA END

! ***************************
! * Capillary pressure data *
! ***************************

CPRDATA BEGIN

   ! Zero capillary pressure input
   CPRMODEL cprzeroma

   ! Create a water capillary pressure table
   ! which will ensure initial water saturation is at Scw = 0.10
   ! If minimum water saturation in table is 0. then 
   ! MoReS will initialise with initial Swi = 0.

   TABLE PCTABLEDEF cprwat
   ! Fill the water table only if the water phase is present
   IF ( EXISTS(WATER) ) THEN                            \
   (                                                    \
     cprwat,  0.10 ,   0.0 *PSI;                        \
     cprwat,  1.00,    0.0 *PSI;                        \
     EXEC_STRING("cprzeroma PCTABLE WATER  : cprwat")   \
   )

   !  Model to be used for matrix only if deck is fractured
   cprzeroma LOC [1:NX, 1:NY, 1:NZ, MAT]

   !  Second, empty model is created which sets Pc = 0 everywhere, i.e.
   !  also connate saturations at 0
   CPRMODEL cprzerofr

   ! Assign only if fractures present
   IF ( NF EQ 2) THEN                           \
   (                                            \
      cprzerofr LOC [1:NX, 1:NY, 1:NZ, FRC]     \
   )

CPRDATA END