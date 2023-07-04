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
using namespace std;
//ofstream MyFile("pitsize.txt");
namespace ns3 {

  void printPitCSSize(int sample_sim_time, int run, string file_path){
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
    auto& pit_10 = Names::Find<Node>("Rout10")->GetObject<ndn::L3Protocol>()->getForwarder()->getPit();
    auto& pit_11 = Names::Find<Node>("Rout11")->GetObject<ndn::L3Protocol>()->getForwarder()->getPit();

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
    std::cout << "PIT_10 number of entries: " << pit_10.size() << endl;
    std::cout << "PIT_11 number of entries: " << pit_11.size() << endl;

    fstream log;
    std::stringstream ss;
    ss << file_path << "pit-size-" << run << ".txt";
    std::string path = ss.str();
    log.open(path, fstream::app);

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
      log << "PIT_10 number of entries: " << pit_10.size() << endl;
      log << "PIT_11 number of entries: " << pit_11.size() << endl;
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

int main(int argc, char** argv)
{
  int run=1;
  int attacker_frequency=64;
  uint32_t badNodes = 11;

  std::stringstream main_path;
  main_path << "results/dataset/IFA_4_Existing/DFN_topology/variable_attackers/" << attacker_frequency << "x/" << badNodes << "_attackers/"; //pit-size-" << run << ".txt";

  std::stringstream sub_path;
  sub_path << main_path.rdbuf() << "rate-trace-" << run << ".txt";

  std::stringstream drop;
  drop << main_path.str() << "drop-trace-" << run << ".txt";

  std::stringstream topo;
  topo << main_path.str() << run << "_topology.txt";

  std::string results_path = main_path.str();
  std::string rate_results = sub_path.str();
  std::string drop_results = drop.str();
  std::string topo_results = topo.str();

  // Read optional command-line parameters (e.g., enable visualizer with ./waf --run=<> --     visualize
  CommandLine cmd;
  //cmd.AddValue("badNodes", "Number of attackers", badNodes);
  //cmd.AddValue("goodNodes", "Number of legitimate consumers", goodNodes);
  cmd.Parse(argc, argv);

  // Read and parse topology with annotated topology reader
  AnnotatedTopologyReader topologyReader("", 1);
  topologyReader.SetFileName("/home/enkeleda/ndnSIM/scenario-trials/scenarios/DFN_topology_general.txt");
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

  NodeContainer leaves;
  NodeContainer attackers;
  NodeContainer consumers;
  NodeContainer producers;
  NodeContainer routers;

  for_each (NodeList::Begin(), NodeList::End(), [&] (Ptr<Node> node){
    // cout << Names::FindName(node) << endl;
    if (Names::FindName(node).compare(0, 4, "User")==0)
      {
        leaves.Add(node);
      }

    if (Names::FindName(node).compare(0, 4, "Prod")==0)
      {
        producers.Add(node);
      }

    if (Names::FindName(node).compare(0, 4, "Rout")==0)
      {
        routers.Add(node);
      }
  });

  cout << "Total number of nodes: " << leaves.GetN()+producers.GetN()+routers.GetN() << endl;
  cout << "Number of users: " << leaves.GetN() << endl;
  //cout << "Number of producers: " << producers.GetN() << endl;
  //cout << "Number of routers: " << routers.GetN() << endl;

  uint32_t goodNodes = leaves.GetN()-badNodes;

  if (goodNodes < 1){
    NS_FATAL_ERROR("Number of legitimate nodes not correct");
    exit(1);
  }

  if (goodNodes == 0){
    goodNodes = leaves.GetN()-badNodes;
  }

  if (leaves.GetN() < goodNodes+badNodes){
    NS_FATAL_ERROR("Number of legitimate and attacker nodes must sum up to number of users in topology");
    exit(1);
  }

  cout << "Number of attacker nodes: " << badNodes << endl;
  cout << "Number of legitimate nodes: " << goodNodes << endl;

  set< Ptr<Node> > bads;
  set< Ptr<Node> > goods;

  while (bads.size () < badNodes)
  {
    int randVar = random(0, leaves.GetN()-1);
    cout << randVar << endl;
    Ptr<Node> node = leaves.Get(randVar);

    if (bads.find (node) != bads.end ())
      continue;
    bads.insert (node);

    string name = Names::FindName (node);
    string postfix = to_string(randVar+1);
    Names::Rename (name, "Atta"+postfix);
  }

  while (goods.size () < goodNodes)
    {
      int randVar = random(0, leaves.GetN()-1);
      Ptr<Node> node = leaves.Get (randVar);
      if (goods.find (node) != goods.end () ||
          bads.find (node) != bads.end ())
        continue;

      goods.insert (node);
      string name = Names::FindName(node);
      string postfix = to_string(randVar+1);
      Names::Rename (name, "Cons"+postfix);
    }

  for_each (NodeList::Begin(), NodeList::End(), [&] (Ptr<Node> node){
     cout << Names::FindName(node) << endl;
  });

  auto assignNodes = [&cout](NodeContainer &aset, const string &str) {
    return [&cout, &aset, &str] (Ptr<Node> node)
    {
      string name = Names::FindName (node);
      cout << name << " ";
      aset.Add (node);
    };
  };
  cout << endl;

  cout << "Attackers: ";
  std::for_each (bads.begin (), bads.end (), assignNodes (attackers, "Atta"));
  cout << "\nConsumers: ";
  std::for_each (goods.begin (), goods.end (), assignNodes (consumers, "Cons"));
  cout << "\n";


  for (NodeContainer::Iterator node = consumers.Begin (); node != consumers.End (); node++) {
    int freq1 = random(50, 150);
    //int freq2 = random(50, 150);
    std::cout << freq1 << "\n";
    ndn::AppHelper consumerHelper("ns3::ndn::ConsumerZipfMandelbrot");
    consumerHelper.SetPrefix("/good/" + Names::FindName(*node));
    std::string pref = "/good/" + Names::FindName(*node);
    //printf("%s\n", pref.c_str());
    consumerHelper.SetAttribute("Frequency", DoubleValue(freq1));
    //consumerHelper.SetAttribute("Randomize", StringValue("exponential"));
    consumerHelper.SetAttribute("NumberOfContents", StringValue("1000")); //this is the size of popularity ranking list
    consumerHelper.SetAttribute("q", StringValue("0.5")); // parameter q for zipf distribution
    consumerHelper.SetAttribute("s", StringValue("0.9")); // parameter s for zipf distribution
    consumerHelper.Install(*node);
  }
  printf("ConsumerHelper installed\n");

  // Attackers
  for (NodeContainer::Iterator node = attackers.Begin (); node != attackers.End (); node++) {
    //int freq1 = random(50, 150);
    int freq2 = random(3200, 9600);
    std::cout << freq2 << "\n";
    ndn::AppHelper attackerHelper("ns3::ndn::ConsumerZipfMandelbrot");
    attackerHelper.SetPrefix("/good/" + Names::FindName(*node));
    std::string pref1 = "/good/" + Names::FindName(*node);
    //printf("%s\n", pref1.c_str());
    attackerHelper.SetAttribute("Frequency", DoubleValue(freq2));
    //attackerHelper.SetAttribute("Randomize", StringValue("exponential")); // uniform requests for the attacker
    attackerHelper.SetAttribute("NumberOfContents", StringValue("1000")); //this is the size of popularity ranking list
    attackerHelper.SetAttribute("q", StringValue("0.5")); // parameter q for zipf distribution
    attackerHelper.SetAttribute("s", StringValue("0.9")); // parameter s for zipf distribution
    ApplicationContainer attacker = attackerHelper.Install(*node);
    attacker.Start(Seconds(50.0)); // start attackers at 20s
    attacker.Stop(Seconds(300.0)); // stop attackers at the end of simulation
  }
  printf("AttackersHelper installed\n");

  // Producer
  for (NodeContainer::Iterator node = producers.Begin (); node != producers.End (); node++){
    ndn::AppHelper producerHelper("ns3::ndn::Producer");
    std::string prefix = "/good";
    producerHelper.SetPrefix(prefix);
    producerHelper.SetAttribute("PayloadSize", StringValue("1024"));
    producerHelper.Install(*node);
    ndnGlobalRoutingHelper.AddOrigins(prefix, *node);
    //ndnGlobalRoutingHelper.AddOrigins("/line/forged", producer);
  }
  printf("ProducerHelper installed\n");
  // Calculate and install FIBs
  ndn::GlobalRoutingHelper::CalculateRoutes();

  ndn::L3RateTracer::InstallAll(rate_results, Seconds(1));
  L2RateTracer::InstallAll(drop_results, Seconds(1));
  topologyReader.SaveTopology(topo_results);
  
  for (int i = 1; i<=300; i++){
    //std::cout << ns3::Simulator::Now().ToDouble(Time::S) << endl;
    ns3::Simulator::Schedule(Seconds(i), printPitCSSize, i, run, results_path);
  }

  Simulator::Stop(Seconds(300.0)); // simulation time
  Simulator::Run();
  Simulator::Destroy();
  
  return 0;
}

} // namespace ns3

int main(int argc, char** argv)
{
  return ns3::main(argc, argv);
}
