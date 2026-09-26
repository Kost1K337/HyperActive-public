%%%
%%%       IMPLEMENTATION OF EGG MODEL IN MRST (Fully implicit solvers)
%%%
%%%  Implementation of Egg model in MRST 
%%%
%%%  Last update:   Nov 2013
%%%  Owner:         M.Mohsin Siraj (PhD - TU/e) & R.M. Fonseca(TU Delft)

%
% This file is an implementation of the Egg model in
% MRST and will be used as real reservoir model for further research.

clc, clear all, close all

%% Set up gravity and add deckformat.
% gravity reset
% gravity off

mrstModule add ad-fi deckformat

%% Initialize the model properties (Grids, paraemters etc.) of EGGs Model

% We read in the deck file, convert the units to SI and then use the 
% resulting deck variable to create grid, rock, fluid and well 
% configurations.

%%%
%%%     Add the directly of the Eclipse deck here!!!
%%%

fn    = fullfile('D:', 'PHD', 'Egg_Model','Standard_Egg_Model', 'MRST', ...
    'Egg_Model_ECL.DATA');

if ~exist(fn, 'file'),
   error('Egg model data is not available.')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% 
%%%     Reading the input deck
%%%
deck  = readEclipseDeck(fn);      
deck = convertDeckUnits(deck);  % Convert to MRST units (SI)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%
%%%     Reading grid structure
%%%
G     = initEclipseGrid(deck); 
G     = removeCells(G, ~deck.GRID.ACTNUM);
G     = computeGeometry(G);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%
%%%     Defining fluid properties
%%%
fluid = initDeckADIFluid(deck);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%
%%%     Reading rock properties (permeability and porosity fields)
%%%
      
rock  = initEclipseRock(deck);
rock  = compressRock(rock, G.cells.indexMap);


gravity on

% Get schedule
schedule = deck.SCHEDULE;

% Enable this to get convergence reports when solving schedules
verbose = false;
actnum  = deck.GRID.ACTNUM; 
%% 
%                     Introduce wells
W = processWells(G, rock, deck.SCHEDULE.control(1)); % Introduce wells


%% Initialize schedule and system before solving for all timesteps
% Change of pressure initializaton!!
po(1) = 400*barsa;

% for i = 2:7
%     po(i) = po(i-1) + 0.3*barsa;
% end
% 
% Po = repmat(po(1),3600,1);
% for i =2:7
%     Po = [Po;repmat(po(i),3600,1)];
% end

rSol         = initResSol(G, po(1),0.1);

initpress    = convertTo(rSol.pressure(1:G.cells.num), barsa);
 
%% Main loop
system = initADISystem({'Water', 'Oil'}, G, rock, fluid);
timer = tic;
[wellSols rSol iter] = runScheduleADI(rSol, G, rock, system, schedule)
toc(timer)

%% Put the well solution data into a format more suitable for plotting
[qWs, qOs, qGs, bhp] = wellSolToVector(wellSols);



