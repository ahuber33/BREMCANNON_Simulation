/// BREMCANNONSimRunAction.cc
//// Auteur: Arnaud HUBER for ENL group <huber@cenbg.in2p3.fr>
//// Copyright: 2022 (C) Projet RATP - ENL [LP2IB] - CELIA

#include "BREMCANNONSimRunAction.hh"

using namespace CLHEP;

G4Mutex BREMCANNONSimRunAction::fileMutex = G4MUTEX_INITIALIZER;
const G4String BREMCANNONSimRunAction::path_bin = "../bin/";

BREMCANNONSimRunAction::BREMCANNONSimRunAction(BREMCANNONSimGeometry *Geom, const char *suff, G4bool pMT)
    : detector (Geom),
    suffixe(suff),
      flag_MT(pMT),
      f(nullptr)
{
}
BREMCANNONSimRunAction::~BREMCANNONSimRunAction() {}

void BREMCANNONSimRunAction::DrawMemoryInformations() {
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("VmRSS:") != std::string::npos) { // Mémoire réellement utilisée
            std::cout << "[Memory used] " << line << std::endl;
            break;
        }
    }
}

void BREMCANNONSimRunAction::CreateROOTFile(G4String suffixe)
{
  f = new TFile(suffixe.c_str(), "RECREATE");

  IP = new TTree("IP", "Tree_Information");

  // create the branch for each event.
  // Be careful of the data structure here!  /F is a float  /I is an integer
  RunBranch = IP->Branch("IncidentE", &Stats.IncidentE, "IncidentE/F");
  // RunBranch = IP->Branch("Compteur", &Stats.compteur, "Compteur/I");
  // RunBranch = IP->Branch("EBremCreation", "vector<float>", &Stats.EBremCreation);
  // RunBranch = IP->Branch("EBremExit", "vector<float>", &Stats.EBremExit);

  hist = new TH1D("incident_energy", "incident_energy", 500, 0, 5000);
  hist->GetXaxis()->SetTitle("Energy [keV]");
  hist->GetYaxis()->SetTitle("N");

  // InitialiseHistos();
  for (G4int i = 0; i < Nombre_IP; i++)
  {
    E_dep_electrons[i] = 0.;
    E_dep_gammas[i] = 0.;
    E_dep_tot[i] = 0.;
    E_dep_electrons_eff[i] = 0.;
    E_dep_gammas_eff[i] = 0.;
    E_dep_tot_eff[i] = 0.;
  }
}

void BREMCANNONSimRunAction::WriteROOTFile(TFile *f)
{
  f->cd();

  auto createHistogram = [](const char *name, const char *title)
  {
    auto hist = new TH1D(name, title, 15, 0.5, 15.5);
    hist->GetXaxis()->SetTitle("numero de l'IP");
    hist->GetYaxis()->SetTitle("Energie deposee (keV)");
    return hist;
  };

  auto histo_edep_electrons = createHistogram("edep_electrons", "energy deposited by electrons");
  auto histo_edep_electrons_eff = createHistogram("edep_electrons_eff", "energy effective deposited by electrons");
  auto histo_edep_gammas = createHistogram("edep_gammas", "energy deposited by gammas");
  auto histo_edep_gammas_eff = createHistogram("edep_gammas_eff", "energy effective deposited by gammas");
  auto histo_edep_tot = createHistogram("edep_totale", "energy totale deposited");
  auto histo_edep_tot_eff = createHistogram("edep_totale_eff", "energy effective totale deposited");
  auto histo_PSL = createHistogram("PSL vs IP", "PSL vs IP");
  histo_PSL->GetXaxis()->SetTitle("IP number");
  histo_PSL->GetYaxis()->SetTitle("PSL");

  for (G4int i = 0; i < Nombre_IP; i++)
  {
    //    G4cout<<"depot alpha_electron "<<i<<" "<<E_dep_electrons[i]<<"\t depot alpha_gamma "<<i<<" "<<E_dep_gammas[i]<<G4endl;
    histo_edep_electrons->SetBinContent(i + 1, E_dep_electrons[i]);
    histo_edep_gammas->SetBinContent(i + 1, E_dep_gammas[i]);
    histo_edep_tot->SetBinContent(i + 1, E_dep_tot[i]);
    histo_edep_electrons_eff->SetBinContent(i + 1, E_dep_electrons_eff[i]);
    histo_edep_gammas_eff->SetBinContent(i + 1, E_dep_gammas_eff[i]);
    histo_edep_tot_eff->SetBinContent(i + 1, E_dep_tot_eff[i]);
    histo_PSL->SetBinContent(i + 1, E_dep_tot_eff[i] * detector->GetCoefPSL());
    //G4cout << "Edeptot[" << i << "] = " << E_dep_tot[i] << G4endl;
  }

  f->Write();
  // IP->Write();

  f->Close();
  delete f;
  f = nullptr;

  G4cout << "Write ROOT file" << G4endl;
}


void BREMCANNONSimRunAction::DisplayRunTime(time_t start, time_t end)
{
  G4int elapsed = end - start;
  G4cout << "Run Completed in " << elapsed / 3600
         << ":" << (elapsed % 3600) / 60 << ":"
         << ((elapsed % 3600) % 60) << G4endl;

  // Output the time in the file Runtime.out
  std::ofstream timeout;
  timeout.open("Runtime.out", std::ios::app);
  timeout << "Run "
          << ": " << elapsed / 3600
          << ":" << (elapsed % 3600) / 60
          << ":" << ((elapsed % 3600) % 60) << G4endl;
  timeout.close();
}

//-----------------------------------------------------
//  BeginOfRunAction:  used to calculate the start time and
//  to set up information in the run tree.
//-----------------------------------------------------
void BREMCANNONSimRunAction::BeginOfRunAction(const G4Run *aRun)
{

  threadID = G4Threading::G4GetThreadId();

  if (flag_MT == false)
  {
    start = time(NULL); // start the timer clock to calculate run times
    seed = start;
    G4Random::setTheSeed(seed);
    G4cout << "seed = " << seed << G4endl;
    fileName = suffixe + ".root";
    CreateROOTFile(fileName);
    G4cout << "### Run " << aRun->GetRunID() << " start." << G4endl;
  }

  else
  {
    G4AutoLock lock(&fileMutex); // Verrouillage automatique du mutex
    if (G4Threading::IsMasterThread())
    {
      start = time(NULL); // start the timer clock to calculate run times
      seed = start;
      G4Random::setTheSeed(seed);
    }

    else
    {
      G4String s = std::to_string(threadID);
      fileName = suffixe + "_" + s + ".root";
      CreateROOTFile(fileName);
      G4cout << "filename = " << fileName << G4endl;
    }
  }

  if (G4VVisManager::GetConcreteInstance())
  {
    G4UImanager *UI = G4UImanager::GetUIpointer();
    UI->ApplyCommand("/vis/scene/notifyHandlers");
  }

} // end BeginOfRunAction

//-----------------------------------------------------
//  EndOfRunAction:  used to calculate the end time and
//  to write information to the run tree.
//-----------------------------------------------------
void BREMCANNONSimRunAction::EndOfRunAction(const G4Run *aRun)
{

  G4AutoLock lock(&fileMutex); // Verrouillage automatique du mutex

  if (flag_MT == false)
  {
    end = time(NULL);
    WriteROOTFile(f);
    DisplayRunTime(start, end);
  }

  else
  {
    if (G4Threading::IsMasterThread())
    {
      end = time(NULL);
      DisplayRunTime(start, end);
      // WriteROOTFile(f);
    }

    else
    {
      WriteROOTFile(f);
    }
  }

  if (G4VVisManager::GetConcreteInstance())
  {
    G4UImanager::GetUIpointer()->ApplyCommand("/vis/viewer/update");
  }

  G4cout << "Leaving Run Action" << G4endl;
  DrawMemoryInformations();
}

//---------------------------------------------------------
//  For each event update the statistics in the Run tree
//---------------------------------------------------------

void BREMCANNONSimRunAction::UpdateStatistics(RunTally aRunTally)
{
  //G4AutoLock lock(&fileMutex); // Verrouillage automatique du mutex
  Stats = aRunTally;
  IP->Fill();
}
