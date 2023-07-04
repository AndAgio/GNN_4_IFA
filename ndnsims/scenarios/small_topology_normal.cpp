/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 University of California, Los Angeles
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Spyridon (Spyros) Mastorakis <mastorakis@cs.ucla.edu>
 *          Alexander Afanasyev <alexander.afanasyev@ucla.edu>
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/ndnSIM-module.h"
#include <fstream>
#include <iostream>
#include <random>

//ofstream MyFile("pitsize.txt");
namespace ns3 {

  void printPitCSSize(int sample_sim_time){
    //auto currentNode = ns3::NodeList::GetNode(ns3::Simulator::GetContext());
    //auto& CurrNodePit = currentNode->GetObject<ndn::L3Protocol>()->getForwarder()->getPit();
    //auto& CurrNodeCS = currentNode->GetObject<ndn::L3Protocol>()->getForwarder()->getCs();

    auto& pit_1 = Names::Find<Node>("Rout1")->GetObject<ndn::L3Protocol>()->getForwarder()->getPit();
    auto& pit_2 = Names::Find<Node>("Rout2")->GetObject<ndn::L3Protocol>()->getForwarder()->getPit();
    auto& pit_3 = Names::Find<Node>("Rout3")->GetObject<ndn::L3Protocol>()->getForwarder()->getPit();
    auto& pit_4 = Names::Find<Node>("Rout4")->GetObject<ndn::L3Protocol>()->getForwarder()->getPit();
    auto& pit_5 = Names::Find<Node>("Rout5")->GetObject<ndn::L3Protocol>()->getForwarder()->getPit();
    auto& pit_6 = Names::Find<Node>("Rout6")->GetObject<ndn::L3Protocol>()->getForwarder()->getPit();
    auto& pit_7 = Names::Find<Node>("Rout7")->GetObject<ndn::L3Protocol>()->getForwarder()->getPit();
    auto& pit_8 = Names::Find<Node>("Rout8")->GetObject<ndn::L3Protocol>()->getForwarder()->getPit();
    auto& pit_9 = Names::Find<Node>("Rout9")->GetObject<ndn::L3Protocol>()->getForwarder()->getPit();

    //std::cout << "Simulation time: " << i << endl;
    std::cout << "PIT_1 number of entries: " << pit_1.size() << endl; //<< " | CS number of stored packets: " <<  CurrNodeCS.size();
    std::cout << "PIT_2 number of entries: " << pit_2.size() << endl;
    std::cout << "PIT_3 number of entries: " << pit_3.size() << endl;
    std::cout << "PIT_4 number of entries: " << pit_4.size() << endl;
    std::cout << "PIT_5 number of entries: " << pit_5.size() << endl;
    std::cout << "PIT_6 number of entries: " << pit_6.size() << endl;
    std::cout << "PIT_7 number of entries: " << pit_7.size() << endl;
    std::cout << "PIT_8 number of entries: " << pit_8.size() << endl;
    std::cout << "PIT_9 number of entries: " << pit_9.size() << endl;

    fstream log;
    log.open("results/dataset/IFA_4_Existing/small_topology/normal/pitsize_5.txt", fstream::app);
    if (log.is_open()){
      log << "Simulation time (s): " << sample_sim_time << endl;
      log << "PIT_1 number of entries: " << pit_1.size() << endl;
      log << "PIT_2 number of entries: " << pit_2.size() << endl;
      log << "PIT_3 number of entries: " << pit_3.size() << endl;
      log << "PIT_4 number of entries: " << pit_4.size() << endl;
      log << "PIT_5 number of entries: " << pit_5.size() << endl;
      log << "PIT_6 number of entries: " << pit_6.size() << endl;
      log << "PIT_7 number of entries: " << pit_7.size() << endl;
      log << "PIT_8 number of entries: " << pit_8.size() << endl;
      log << "PIT_9 number of entries: " << pit_9.size() << endl;
    }
    log.close();
  }

  int random(int min, int max) //range : [min, max]
  {
     static bool first = true;
     if (first)
     {
        srand( time(NULL) ); //seeding for the first time only!
        first = false;
     }
     return min + rand() % (( max + 1 ) - min);
  }

int main(int argc, char *argv[])
{

  // Read optional command-line parameters (e.g., enable visualizer with ./waf --run=<> --     visualize
  CommandLine cmd;
  cmd.Parse(argc, argv);

  // Read and parse topology with annotated topology reader
  AnnotatedTopologyReader topologyReader("", 1);
  topologyReader.SetFileName("/home/enkeleda/ndnSIM/scenario-trials/scenarios/topologies/small_topology.txt");
  topologyReader.Read();

  // Install NDN stack on all nodes
  ndn::StackHelper ndnHelper;
  //ndnHelper.SetPit("ns3::ndn::pit::Persistent");
  ndnHelper.InstallAll();

  // Set BestRoute Strategy
  ndn::StrategyChoiceHelper::InstallAll("/", "/localhost/nfd/strategy/best-route");

  // Installing global routing interface on all nodes
  ndn::GlobalRoutingHelper ndnGlobalRoutingHelper;
  ndnGlobalRoutingHelper.InstallAll();

  // Getting containers for the consumer/producer/attackers/routers
  Ptr<Node> producers = Names::Find<Node>("Prod1");
  Ptr<Node> consumers[4] = {Names::Find<Node>("Cons1"), Names::Find<Node>("Cons2"), Names::Find<Node>("Cons3"), Names::Find<Node>("Cons4")};
  Ptr<Node> attackers[4] = {Names::Find<Node>("Atta1"), Names::Find<Node>("Atta2"), Names::Find<Node>("Atta3"), Names::Find<Node>("Atta4")};
  Ptr<Node> routers[9] = {Names::Find<Node>("Rout1"), Names::Find<Node>("Rout2"), Names::Find<Node>("Rout3"), Names::Find<Node>("Rout4"), Names::Find<Node>("Rout5"), Names::Find<Node>("Rout6"), Names::Find<Node>("Rout7"),
                          Names::Find<Node>("Rout8"), Names::Find<Node>("Rout9")};

  // Installing applications
  // Consumers have a random distribution following a uniform
  for (int i = 0; i < 4; i++) {
    int freq1 = random(50, 150);
    int freq2 = random(50, 150);
    std::cout << freq1 <<"\n" << freq2 << "\n";
    ndn::AppHelper consumerHelper("ns3::ndn::ConsumerZipfMandelbrot");
    consumerHelper.SetPrefix("/good/" + Names::FindName(consumers[i]));
    //std::string pref = "/good/" + Names::FindName(consumers[i]);
    //printf("%s\n", pref.c_str());
    consumerHelper.SetAttribute("Frequency", DoubleValue(freq1));
    //consumerHelper.SetAttribute("Frequency", StringValue("100"));
    //consumerHelper.SetAttribute("Randomize", StringValue("exponential"));
    consumerHelper.SetAttribute("NumberOfContents", StringValue("1000")); //this is the size of popularity ranking list
    consumerHelper.SetAttribute("q", StringValue("0.5")); // parameter q for zipf distribution
    consumerHelper.SetAttribute("s", StringValue("0.9")); // parameter s for zipf distribution
    consumerHelper.Install(consumers[i]);

    ndn::AppHelper attackerHelper("ns3::ndn::ConsumerZipfMandelbrot");
    attackerHelper.SetPrefix("/good/" + Names::FindName(attackers[i]));
    //std::string pref1 = "/good/" + Names::FindName(attackers[i]);
    attackerHelper.SetAttribute("Frequency", DoubleValue(freq2));
    //attackerHelper.SetAttribute("Randomize", StringValue("exponential")); // uniform requests for the attacker
    attackerHelper.SetAttribute("NumberOfContents", StringValue("1000")); //this is the size of popularity ranking list
    attackerHelper.SetAttribute("q", StringValue("0.5")); // parameter q for zipf distribution
    attackerHelper.SetAttribute("s", StringValue("0.9")); // parameter s for zipf distribution
    attackerHelper.Install(attackers[i]);
  }
  printf("Consmuers installed\n");

  // Producer
  ndn::AppHelper producerHelper("ns3::ndn::Producer");
  std::string prefix = "/good";
  producerHelper.SetPrefix(prefix);
  producerHelper.SetAttribute("PayloadSize", StringValue("1024"));
  producerHelper.Install(producers);
  ndnGlobalRoutingHelper.AddOrigins(prefix, producers);
  //ndnGlobalRoutingHelper.AddOrigins("/line/forged", producer);

  printf("ProducerHelper installed\n");

  //Ptr<ns3::ndn::nfd::Pit> pit = (*(Names::Find<Node>("Rout1"))

  // Calculate and install FIBs
  ndn::GlobalRoutingHelper::CalculateRoutes();

  ndn::L3RateTracer::InstallAll("results/dataset/IFA_4_Existing/small_topology/normal/rate-trace_5.txt", Seconds(1));
  L2RateTracer::InstallAll("results/dataset/IFA_4_Existing/small_topology/normal/drop-trace_5.txt", Seconds(1));


  for (int i = 1; i<=300; i++){
    //std::cout << ns3::Simulator::Now().ToDouble(Time::S) << endl;
    ns3::Simulator::Schedule(Seconds(i), printPitCSSize, i);
  }

  Simulator::Stop(Seconds(300.0)); // simulation time
  Simulator::Run();

  Simulator::Destroy();

  return 0;
}

} // namespace ns3

int
main(int argc, char* argv[])
{
  return ns3::main(argc, argv);
}
