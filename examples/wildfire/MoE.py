from enum import Enum
import json
from examples.wildfire.paths import SCENARIOS_DIR
import os
cwd = os.getcwd()
print(cwd)

class ExampleName(Enum):
    SALAMIS = 0
    PYRENEES = 1
    PALISADES = 2

class ExampleParameters:
    def __init__(self, BurntArea, CostArea, EmissionArea, FleetAcqCost, FleetOpsCost):
        self.BurntArea = BurntArea
        self.CostArea = CostArea
        self.EmissionArea = EmissionArea
        self.FleetAcqCost = FleetAcqCost
        self.FleetOpsCost = FleetOpsCost

def main():
    # Change to example you are running
    # Made some assumptions about the 
    currentExample = ExampleName.PALISADES
    simOutputPath= 'examples/wildfire/data/scenarios/outputs/Palisades copy_out_simulation.json'
    #"Users/ibrahimoguz/Desktop/FirefightingSoS/SoSID_X-Challenge/examples/wildfire/data/scenarios/Salamis_out_simulation.json"
    agentOutputPath="examples/wildfire/data/scenarios/outputs/Palisades copy_out_agents_SuppressionUAV.json"
    fuelUSDPricePerKg = 0.82
    pricePerAircraft = 50000000
    availCost = 100
    flightCostPerHour = 100
    energyCostPerHour = 100

    # Define constant parameters
    w1 = 0.2
    w2 = 0.2
    w3 = 0.2
    w4 = 0.3
    w5 = 0.1
    salamis = ExampleParameters(4146,139929,7140,100000000,268000)
    pyrenees = ExampleParameters(9938,175089,23641,100000000,167000)
    palisades = ExampleParameters(9087,1911058,13122,100000000,250000)

    # Choose the specific parameters for the current example
    exampleParameters = [salamis, pyrenees, palisades]
    p = exampleParameters[currentExample.value]

    # Get outputs from the simulation
    with open(simOutputPath, 'r') as file:
        simData = json.load(file)

    with open(agentOutputPath, 'r') as file:
        agentData = json.load(file)

    AverageAgentFlightTime = simData['fleet_average_cumulative_flight_time']
    FleetSize = simData['n_agents']
    TotalFuelPrice = (agentData['SuppressionUAV_0_dhc_515_total_propellant_mass_consumed'] + 
                         agentData['SuppressionUAV_1_dhc_515_total_propellant_mass_consumed']) * fuelUSDPricePerKg
    
    FleetAcqCost = pricePerAircraft * FleetSize
    OpsCost = (availCost + (flightCostPerHour + energyCostPerHour) * AverageAgentFlightTime + TotalFuelPrice) * FleetSize

    # Define the output parameters
    o = ExampleParameters(simData['burnt_area'],
                          simData['total_fire_cost'],
                          simData['total_fire_emissions'],
                          FleetAcqCost,
                          OpsCost)

    # Calculate and Display the MoE
    MoE = 1 - (w1*o.BurntArea/p.BurntArea + 
       w2*o.CostArea/p.CostArea + 
       w3*o.EmissionArea/p.EmissionArea + 
       w4*o.FleetAcqCost/p.FleetAcqCost + 
       w5*o.FleetOpsCost/p.FleetOpsCost)
    
    print("Burnt Area:\t\t" + str(o.BurntArea) + "\t\t\t~ Term 1: " + str(w1*o.BurntArea/p.BurntArea))
    print("Cost Area:\t\t" + str(o.CostArea) + "\t\t~ Term 2: " + str(w2*o.CostArea/p.CostArea))
    print("Emissions Area:\t\t" + str(o.EmissionArea) + "\t\t~ Term 3: " + str(w3*o.EmissionArea/p.EmissionArea))
    print("Fleet Acquisition Cost:\t" + str(o.FleetAcqCost) + "\t\t\t~ Term 4: " + str(w4*o.FleetAcqCost/p.FleetAcqCost))
    print("Fleet Operational Cost:\t" + str(o.FleetOpsCost) + "\t~ Term 5: " + str(w5*o.FleetOpsCost/p.FleetOpsCost))
    print("Final MOE:\t\t" + str(MoE))

if __name__ == "__main__":
    main()