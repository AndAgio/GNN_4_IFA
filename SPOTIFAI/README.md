## SPOTIFAI
Available here: https://tinyurl.com/7zp4z5zv

SPOTIFAI is a dataset of Interest Flooding Attacks (IFAs) simulated using ndnSIM (https://ndnsim.net/current/).

***

### Structure
The dataset's repo includes the following submodules:
* Topologies
* Normal
* IFA_4_Existing
* IFA_4_Non_Existing

*Note: the two considered attacking scenarios are requests for existing content -- i.e., IFA_4_Existing --
and for non existing contents---i.e., IFA_4_Non_Existing*

The *Topologies* submodule includes the three main topologies we have considered for the simulations:
* Small Topology - composed of 18 nodes including 8 users (consumers and attackers), 9 routers and 1 producer
* DFN Topology - composed of a set of 29 nodes including 12 users (consumers and attackers), 11 routers and 6 producers
* Large-scale Topology - composed of 163 nodes

The *Normal* submodule contains the IFA-free simulations for the three considered topologies.
Here, SPOTIFAI simulates normal scenarios without the presence of attackers in the network.

In the *IFA_4_Existing* submodule, for each topology, SPOTIFAI contains two main attacking cases:
* fixed_attackers: the number and the position of attackers is kept fixed for all the 5 runs.
  - Consumers' frequency is uniformly selected in [50, 150] requests/second
  - Attackers' frequency is uniformly selected in ranges 4x[50, 150],8x[50, 150], 16x[50, 150], 32x[50, 150] and 64x[50, 150]  
  - Number of attackers is 4 both *Small topology* and *DFN topology*.
* variable_attackers: the number of attackers is fixed and their position changes from one run to the other (max of 5 runs)
  - Consumers' frequency is uniformly selected in [50, 150] requests/second
  - Attackers' frequency is uniformly selected in ranges 4x[50, 150],8x[50, 150], 16x[50, 150], 32x[50, 150] and 64x[50, 150]
  - Number of attackers for *small topology* is 4, 5, 6, 7
  - Number of attackers for *DFN topology* is 4, 8, 11

Lastly, the *IFA_4_Non_Existing* contains only some samples as we are continuing to populate such submodule.
