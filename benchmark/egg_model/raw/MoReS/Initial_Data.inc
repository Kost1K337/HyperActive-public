! ******************
! * Initialisation *
! ******************

INIDATA BEGIN

   ! Make sure IniModel settings are flags only
   BEFORE_V2_3 = OFF;

   ! Create the model
   INIMODEL model2

   ! Set P_SAT initialisation flag
   model2 P_SAT

   ! Define initialisation region
   model2 LOC [1:NX, 1:NY, 1:NZ, 1:NF]

   ! Hydrostatic initialisation
   model2 INIDATUM_HGT = GEODATA.ORIGINZ
   model2 INIDATUM_PRF = 400 * BAR
   model2 HYDROSTATIC

   ! Activate P_SAT initialisation
   model2 PREPARE

INIDATA END
